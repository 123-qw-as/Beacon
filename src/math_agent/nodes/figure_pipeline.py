"""figure_pipeline：扫描 code_artifacts/sensitivity_runs 里的 PNG，
对每张图做 Critic 评分（最多重试 1 次）+ Analyst 写图说。

不重新生成图（重生成的成本/收益不划算）；只评分、解读，
低质量图保留但 quality_score 反映在 Evaluation 中。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from math_agent.config import MODEL_ROUTING
from math_agent.llm import complete
from math_agent.prompts.figure_critic import (
    SYSTEM as FC_SYSTEM, build_prompt as fc_prompt,
)
from math_agent.prompts.figure_analyst import (
    SYSTEM as FA_SYSTEM, build_prompt as fa_prompt,
)
from math_agent.state import FigureArtifact, MathModelingState
from math_agent.tools.image import inspect_image, encode_image_to_data_url
from math_agent.tools.runner import extract_numeric_results


class FigureCriticOut(BaseModel):
    score: int = Field(ge=0, le=10)
    issues: list[str] = []
    suggestions: list[str] = []
    approved: bool = False


class FigureAnalysisOut(BaseModel):
    analysis: str


_MAX_CRITIC_RETRIES = 1  # critic 不通过时，最多再问一次（不重新生成图）

_FIGURE_EVIDENCE_KEYS = (
    "total_cost", "total_carbon", "vehicles", "service_rate",
    "avg_delivery_time", "timewin_rate", "fuel_ratio",
)


def _matches_primary_evidence(stdout: str, primary: dict[str, float]) -> bool:
    """拒绝仍携带旧主方案口径的补充图，避免图说污染正式论文。"""
    emitted = extract_numeric_results(stdout).get("ours", {})
    if not primary or not emitted:
        return True
    for key in _FIGURE_EVIDENCE_KEYS:
        if key not in primary or key not in emitted:
            continue
        expected = primary[key]
        observed = emitted[key]
        tolerance = max(abs(expected) * 0.2, 0.1 if "rate" in key or "ratio" in key else 1e-6)
        if abs(observed - expected) > tolerance:
            return False
    return True


def _collect_pngs(state: MathModelingState) -> list[tuple[str, str, str]]:
    """返回 [(path, purpose, context_text), ...]"""
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    primary_artifact = next(
        (
            artifact for artifact in reversed(state.latest_code_artifacts())
            if artifact.success and artifact.evidence_role == "primary"
        ),
        None,
    )
    primary_metrics = (
        extract_numeric_results(primary_artifact.stdout).get("ours", {})
        if primary_artifact is not None else {}
    )
    for art in state.latest_code_artifacts():
        if not art.success or not _matches_primary_evidence(art.stdout, primary_metrics):
            continue
        for p in art.artifact_paths:
            if p.lower().endswith(".png") and p not in seen:
                seen.add(p)
                out.append((p, art.purpose, art.stdout[:500]))
    for r in state.sensitivity_runs:
        if (r.figure_path and r.figure_path.lower().endswith(".png")
                and r.figure_path not in seen):
            seen.add(r.figure_path)
            ctx = f"parameter={r.parameter} values={r.values} {r.metric}={r.results}"
            out.append((r.figure_path, f"敏感性分析: {r.parameter}", ctx))
    return out


def figure_prepare_node(state: MathModelingState) -> dict:
    """创建逐图审查队列，不把图片 base64 写进 checkpoint。"""
    queue = [
        {"id": f"figure:{i}", "path": path, "purpose": purpose, "context": context}
        for i, (path, purpose, context) in enumerate(_collect_pngs(state))
    ]
    return {
        "figure_work_queue": queue, "figure_work_results": [],
        "figure_current_critic": {}, "figure_critic_attempt": 0,
        "figure_phase": "critic" if queue else "done",
    }


def figure_critic_node(state: MathModelingState) -> dict:
    """对当前图片执行一次 critic 调用。"""
    queue = list(state.figure_work_queue)
    if not queue:
        return {"figure_phase": "done"}
    item = queue[0]
    try:
        info = inspect_image(item["path"])
        url = encode_image_to_data_url(item["path"])
    except (OSError, ValueError) as exc:
        queue.pop(0)
        delta = {
            "errors": [f"figure_pipeline: 无法读取图像 {item['path']}: {exc}"],
            "figure_work_queue": queue,
            "figure_phase": "critic" if queue else "done",
        }
        if not queue and state.figure_work_results:
            delta["figures"] = list(state.figure_work_results)
            delta["figure_work_results"] = []
        return delta
    meta = f"{info.width}x{info.height}px, dpi={info.dpi}"
    critic: FigureCriticOut = complete(
        fc_prompt(item["purpose"], meta), schema=FigureCriticOut,
        system=FC_SYSTEM, model=MODEL_ROUTING["figure_critic"],
        images=[url], profile="vision",
    )
    attempt = state.figure_critic_attempt
    if not critic.approved and attempt < _MAX_CRITIC_RETRIES:
        return {
            "figure_current_critic": critic.model_dump(),
            "figure_critic_attempt": attempt + 1, "figure_phase": "critic",
        }
    return {
        "figure_current_critic": critic.model_dump(),
        "figure_critic_attempt": attempt, "figure_phase": "analysis",
    }


def figure_analysis_node(state: MathModelingState) -> dict:
    """解释当前图片并完成该工作项；下一张图片从新 checkpoint 开始。"""
    queue = list(state.figure_work_queue)
    if not queue:
        return {"figure_phase": "done"}
    item = queue.pop(0)
    url = encode_image_to_data_url(item["path"])
    analysis: FigureAnalysisOut = complete(
        fa_prompt(item["purpose"], item["context"]), schema=FigureAnalysisOut,
        system=FA_SYSTEM, model=MODEL_ROUTING["figure_analyst"],
        images=[url], profile="vision",
    )
    critic = FigureCriticOut.model_validate(state.figure_current_critic)
    results = [*state.figure_work_results, FigureArtifact(
        path=item["path"], purpose=item["purpose"], caption=analysis.analysis[:60],
        quality_score=critic.score, quality_issues=list(critic.issues),
        analysis=analysis.analysis,
    )]
    if queue:
        return {
            "figure_work_queue": queue, "figure_work_results": results,
            "figure_current_critic": {}, "figure_critic_attempt": 0,
            "figure_phase": "critic",
        }
    return {
        "figures": results, "figure_work_queue": [], "figure_work_results": [],
        "figure_current_critic": {}, "figure_critic_attempt": 0,
        "figure_phase": "done",
    }


def figure_pipeline_node(state: MathModelingState) -> dict:
    figures: list[FigureArtifact] = []
    errors: list[str] = []
    for path, purpose, context in _collect_pngs(state):
        try:
            info = inspect_image(path)
            url = encode_image_to_data_url(path)
        except (OSError, ValueError) as exc:
            errors.append(f"figure_pipeline: 无法读取图像 {path}: {exc}")
            continue
        meta = f"{info.width}x{info.height}px, dpi={info.dpi}"

        critic: FigureCriticOut | None = None
        for _ in range(_MAX_CRITIC_RETRIES + 1):
            critic = complete(
                fc_prompt(purpose, meta),
                schema=FigureCriticOut, system=FC_SYSTEM,
                model=MODEL_ROUTING["figure_critic"],
                images=[url],
                profile="vision",
            )
            if critic.approved:
                break

        analysis: FigureAnalysisOut = complete(
            fa_prompt(purpose, context),
            schema=FigureAnalysisOut, system=FA_SYSTEM,
            model=MODEL_ROUTING["figure_analyst"],
            images=[url],
            profile="vision",
        )

        figures.append(FigureArtifact(
            path=path, purpose=purpose,
            caption=analysis.analysis[:60],   # 简短题注（正文用）
            quality_score=critic.score if critic else 0,
            quality_issues=list(critic.issues) if critic else [],
            analysis=analysis.analysis,
        ))

    delta: dict = {}
    if figures:
        delta["figures"] = figures
    if errors:
        delta["errors"] = errors
    return delta
