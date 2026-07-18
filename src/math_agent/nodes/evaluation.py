"""Evaluation Module：与 PaperCritic 解耦的独立量化打分。

为避免 LLM 在 overall 上算错，节点最终用确定性公式重算 overall。
"""
from __future__ import annotations

from math_agent.config import MODEL_ROUTING
from math_agent.llm import complete
from math_agent.prompts.evaluation import SYSTEM, build_prompt
from math_agent.state import EvaluationReport, MathModelingState


_WEIGHTS = {
    "assumption_reasonableness": 0.20,
    "modeling_creativity": 0.25,
    "result_correctness": 0.25,
    "writing_clarity": 0.20,
    "extra_depth": 0.10,
}


def _compute_overall(r: EvaluationReport) -> float:
    total = sum(getattr(r, k) * w for k, w in _WEIGHTS.items())
    return round(total, 2)


def _unwrap_scores(raw: EvaluationReport) -> EvaluationReport:
    """如果 LLM 返回嵌套 scores 而非顶层字段，解包并重新赋值。"""
    scores_raw = getattr(raw, "_scores", None) or getattr(raw, "scores", None)
    if scores_raw is None:
        return raw
    # 尝试从嵌套 scores dict 里提取各维度分
    for dim in _WEIGHTS:
        if getattr(raw, dim, None) is None:
            val = scores_raw.get(dim)
            if val is not None:
                setattr(raw, dim, val)
    return raw


def evaluation_node(state: MathModelingState) -> dict:
    p = state.paper
    if not any([p.abstract, p.model_section, p.solution]):
        return {"errors": ["evaluation: 论文初稿为空，跳过评估"]}

    paper_critic = state.latest_critic("paper")
    depth_labels = (
        "ALGORITHM_SEARCH", "ROBUSTNESS", "SERVICE_DIAGNOSTICS", "DYNAMIC_EVENTS",
    )
    depth_signals = {
        label: any(
            artifact.success and artifact.evidence_role == "primary"
            and f"{label}:" in (artifact.stdout or "")
            for artifact in state.latest_code_artifacts()
        )
        for label in depth_labels
    }
    out: EvaluationReport = complete(
        build_prompt(
            p, state.figures, state.sensitivity_runs, paper_critic,
            state.table_warnings, depth_signals=depth_signals,
        ),
        schema=EvaluationReport, system=SYSTEM,
        model=MODEL_ROUTING["evaluation"],
    )
    out = _unwrap_scores(out)
    out.overall = _compute_overall(out)  # 确定性自校正
    return {"evaluation": out}
