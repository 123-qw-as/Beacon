from __future__ import annotations
from pathlib import Path
import tempfile

from pydantic import BaseModel
from math_agent.llm import complete
from math_agent.config import MODEL_ROUTING
from math_agent.prompts.coder import SYSTEM, build_prompt  # noqa: F401  (build_prompt kept for backward compat)
from math_agent.prompts.coder_figure_one import build_prompt_figure_one
from math_agent.state import MathModelingState, CodeArtifact
from math_agent.tools.runner import run_python


class CoderDraft(BaseModel):
    purpose: str
    code: str


MAX_CODE_RETRIES = 1  # 一次失败后再给一次机会，避免成本失控


def coder_node(state: MathModelingState) -> dict:
    model = state.latest_model()
    workdir = Path(state.output_dir) if state.output_dir else Path(tempfile.mkdtemp(prefix="math_agent_"))
    workdir.mkdir(parents=True, exist_ok=True)

    # Plan D Phase 3：按 modeler.figure_purposes 拆成 N 个单图调用；
    # 旧 state 无 figure_purposes → 退化为单次调用（用 description 当 purpose），向后兼容。
    purposes = model.figure_purposes or [model.description]

    artifacts: list[CodeArtifact] = []
    for i, purpose in enumerate(purposes):
        prev_err: str | None = None
        prev_kind: str = ""
        for attempt in range(MAX_CODE_RETRIES + 1):
            draft: CoderDraft = complete(
                build_prompt_figure_one(model, purpose, prev_err, prev_kind),
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
                )
            )
            if result.success:
                break
            prev_err = result.stderr
            prev_kind = result.error_kind

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
