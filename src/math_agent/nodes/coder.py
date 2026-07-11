from __future__ import annotations
from pathlib import Path
import tempfile
import json as _json

from pydantic import BaseModel
from math_agent.llm import complete
from math_agent.config import MAX_CODE_RETRIES, MODEL_ROUTING
from math_agent.prompts.coder import SYSTEM, build_prompt  # noqa: F401  (build_prompt kept for backward compat)
from math_agent.prompts.coder_figure_one import build_prompt_figure_one
from math_agent.prompts.coder_baseline import BASELINE_SPECS, build_baseline_prompt
from math_agent.state import MathModelingState, CodeArtifact
from math_agent.tools.runner import run_python


class CoderDraft(BaseModel):
    purpose: str
    code: str


def coder_node(state: MathModelingState) -> dict:
    model = state.latest_model()
    workdir = Path(state.output_dir) if state.output_dir else Path(tempfile.mkdtemp(prefix="math_agent_"))
    workdir.mkdir(parents=True, exist_ok=True)

    # Plan D Phase 3：按 modeler.figure_purposes 拆成 N 个单图调用；
    # 旧 state 无 figure_purposes -> 退化为单次调用（用 description 当 purpose），向后兼容。
    purposes = model.figure_purposes or [model.description]

    # batch 递增：每次 coder 运行产生新批次，一致性审查只看最新 batch
    max_batch = max((a.batch for a in state.code_artifacts), default=0)
    current_batch = max_batch + 1

    checkpoint_path = workdir / "_checkpoint.json"

    artifacts: list[CodeArtifact] = []
    for i, purpose in enumerate(purposes):
        prev_err: str | None = None
        prev_kind: str = ""
        for attempt in range(MAX_CODE_RETRIES + 1):
            draft: CoderDraft = complete(
                build_prompt_figure_one(model, purpose, prev_err, prev_kind,
                                        blueprint=state.problem_blueprint,
                                        data_dir=state.data_dir,
                                        data_files=state.data_files),
                schema=CoderDraft,
                system=SYSTEM,
                model=MODEL_ROUTING["coder"],
            )
            result = run_python(draft.code, workdir=workdir / f"fig_{i}_attempt_{attempt}", timeout=300)
            artifacts.append(
                CodeArtifact(
                    purpose=draft.purpose,
                    code=draft.code,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    success=result.success,
                    artifact_paths=result.artifact_paths,
                    batch=current_batch,
                )
            )
            # 每 attempt 立即写 checkpoint + 进度日志
            checkpoint_path.write_text(
                _json.dumps(
                    [{"purpose": a.purpose, "success": a.success,
                      "stdout": a.stdout[:500], "stderr": a.stderr[:500],
                      "artifact_paths": a.artifact_paths, "batch": a.batch}
                     for a in artifacts],
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[coder] purpose={i+1}/{len(purposes)} attempt={attempt} "
                  f"success={result.success} error_kind={result.error_kind}", flush=True)
            if result.success:
                break
            prev_err = result.stderr
            prev_kind = result.error_kind

    # ---- Phase 2: 对照方案运行 ----
    # 取最后一个成功的主方案代码作为基础
    main_code = next((a.code for a in reversed(artifacts) if a.success), "")
    if main_code:
        for name, category, instruction in BASELINE_SPECS:
            try:
                baseline_draft: CoderDraft = complete(
                    build_baseline_prompt(state.problem, main_code, name, category, instruction),
                    schema=CoderDraft,
                    system=SYSTEM,
                    model=MODEL_ROUTING["coder"],
                )
                baseline_result = run_python(
                    baseline_draft.code,
                    workdir=workdir / f"baseline_{category}",
                    timeout=300,
                )
                artifacts.append(
                    CodeArtifact(
                        purpose=baseline_draft.purpose,
                        code=baseline_draft.code,
                        stdout=baseline_result.stdout,
                        stderr=baseline_result.stderr,
                        success=baseline_result.success,
                        artifact_paths=baseline_result.artifact_paths,
                        category=f"baseline:{category}",
                        batch=current_batch,
                    )
                )
            except Exception as e:
                # 对照方案失败不阻断主流程
                artifacts.append(
                    CodeArtifact(
                        purpose=f"{name}对照方案（失败）",
                        code="",
                        stdout="",
                        stderr=str(e)[:500],
                        success=False,
                        category=f"baseline:{category}",
                        batch=current_batch,
                    )
                )

    delta: dict = {"code_artifacts": artifacts}
    if not any(a.success for a in artifacts):
        # 所有图的所有 attempts 都失败：显式写 error 让 writer / paper_critic 看到，
        # 避免 IRON RULE 1（"数字必须可追溯"）被失败 stdout 偷偷绕过。
        # 前缀与 sensitivity_node 的 "sensitivity: ..." 约定对齐。
        last_stderr = artifacts[-1].stderr if artifacts else ""
        delta["errors"] = [
            f"coder: 所有 {len(artifacts)} 次尝试均失败；最后一次 stderr 节选：{last_stderr[:300]}"
        ]
    return delta
