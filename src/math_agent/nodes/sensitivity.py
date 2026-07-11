"""Sensitivity 节点：作为'必经'节点存在。

流程：
  1. PLAN：LLM 选参数 + 给出每个参数的扫值。
  2. CODE：LLM 写一段扫参 Python，沙箱执行；解析每行 `RESULT: ...`。
  3. INTERPRET：把数值结果回灌给 LLM 生成每个 run 的解读段。
失败策略：
  - 任何一步失败 → 记录 errors，**不写入半成品 sensitivity_runs**。
  - 调用方（graph）在敏感性失败时不应阻塞流水线，但 PaperCritic / Evaluation 会因 sensitivity_runs 为空而扣分。
"""
from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from math_agent.config import MAX_CODE_RETRIES, MODEL_ROUTING
from math_agent.llm import complete
from math_agent.prompts.sensitivity import (
    PLAN_SYSTEM, CODE_SYSTEM, INTERPRET_SYSTEM,
    build_plan_prompt, build_code_prompt, build_interpret_prompt,
)
from math_agent.state import MathModelingState, SensitivityRun
from math_agent.tools.runner import run_python


class _PlanRun(BaseModel):
    parameter: str
    values: list[float]
    metric: str
    rationale: str = ""


class SensitivityPlan(BaseModel):
    runs: list[_PlanRun]


class SensitivityCode(BaseModel):
    code: str


class Interpretations(BaseModel):
    interpretations: list[str]


_RESULT_RE = re.compile(r"RESULT:\s*parameter=(\S+)\s+values=(\[[^\]]+\])\s+results=(\[[^\]]+\])")
# 只匹配“前面不是字母/数字/下划线/小数点”的数字，避免抓到 np.float64 里的 64
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])-?\d+\.?\d*(?:[eE][+-]?\d+)?")


def _parse_results(stdout: str) -> list[tuple[str, list[float], list[float]]]:
    """从 stdout 中抽 RESULT 行。

    用正则抓 list 里的所有数字，不依赖 ast.literal_eval——后者无法解析
    `[np.float64(1.0), np.float64(2.0)]` 这类带函数调用的 repr（实际遇过）。
    """
    out = []
    for line in stdout.splitlines():
        m = _RESULT_RE.search(line)
        if not m:
            continue
        param = m.group(1)
        values = [float(x) for x in _NUMBER_RE.findall(m.group(2))]
        results = [float(x) for x in _NUMBER_RE.findall(m.group(3))]
        if values and results:
            out.append((param, values, results))
    return out


def sensitivity_node(state: MathModelingState) -> dict:
    final = next((m for m in reversed(state.model_versions) if m.stage == "final"), None)
    if final is None:
        return {"errors": ["sensitivity: 缺少 final 阶段模型，跳过敏感性分析"]}

    workdir = Path(state.output_dir or ".") / "sensitivity"
    workdir.mkdir(parents=True, exist_ok=True)

    # 1) PLAN
    plan: SensitivityPlan = complete(
        build_plan_prompt(final, state.assumptions),
        schema=SensitivityPlan, system=PLAN_SYSTEM,
        model=MODEL_ROUTING.get("modeler"),
    )
    if not plan.runs:
        return {"errors": ["sensitivity: LLM 未给出可执行的 runs"]}

    # 2) CODE+RUN（最多重试 MAX_CODE_RETRIES 次，与 coder 一致：失败按 error_kind 分流）
    sandbox_result = None
    prev_err: str | None = None
    prev_kind: str = ""
    cp_path = workdir / "_sensitivity_checkpoint.json"
    for attempt in range(MAX_CODE_RETRIES + 1):
        code_out: SensitivityCode = complete(
            build_code_prompt(final, [r.model_dump() for r in plan.runs], prev_err, prev_kind),
            schema=SensitivityCode, system=CODE_SYSTEM,
            model=MODEL_ROUTING.get("coder"),
        )
        sandbox_result = run_python(
            code_out.code, workdir=workdir / f"attempt_{attempt}", timeout=300,
        )
        # 每 attempt 立即写 checkpoint + 进度日志
        import json as _json
        cp_path.write_text(
            _json.dumps({
                "attempt": attempt,
                "success": sandbox_result.success,
                "stdout": sandbox_result.stdout[:500],
                "stderr": sandbox_result.stderr[:500],
                "error_kind": sandbox_result.error_kind,
                "artifact_paths": sandbox_result.artifact_paths,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[sensitivity] attempt={attempt} success={sandbox_result.success} "
              f"error_kind={sandbox_result.error_kind}", flush=True)
        if sandbox_result.success:
            break
        prev_err = sandbox_result.stderr
        prev_kind = sandbox_result.error_kind
    if not sandbox_result.success:
        return {"errors": [f"sensitivity: 扫参代码执行失败：{sandbox_result.stderr[:500]}"]}

    parsed = _parse_results(sandbox_result.stdout)
    if not parsed:
        return {"errors": ["sensitivity: 未在 stdout 中解析到任何 `RESULT:` 行"]}

    # 把 parsed 与 plan.runs 对齐（按 parameter 名匹配；缺失的剔除）
    by_name = {p.parameter: p for p in plan.runs}
    aligned: list[SensitivityRun] = []
    for param, vals, res in parsed:
        plan_entry = by_name.get(param)
        if plan_entry is None:
            continue
        if len(vals) != len(res):
            continue
        fig = next((p for p in sandbox_result.artifact_paths if Path(p).stem == param), None)
        aligned.append(SensitivityRun(
            parameter=param, values=vals, metric=plan_entry.metric,
            results=res, figure_path=fig,
        ))
    if not aligned:
        return {"errors": ["sensitivity: 解析结果与计划无法对齐"]}

    # 3) INTERPRET
    interp: Interpretations = complete(
        build_interpret_prompt(aligned),
        schema=Interpretations, system=INTERPRET_SYSTEM,
        model=MODEL_ROUTING.get("writer"),
    )
    if len(interp.interpretations) != len(aligned):
        return {"errors": [
            "sensitivity: 解读数量与数值结果数量不一致，拒绝写入不完整结果"
        ]}
    for r, text in zip(aligned, interp.interpretations):
        r.interpretation = text

    return {"sensitivity_runs": aligned}
