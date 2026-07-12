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


class FigureCriticOut(BaseModel):
    score: int = Field(ge=0, le=10)
    issues: list[str] = []
    suggestions: list[str] = []
    approved: bool = False


class FigureAnalysisOut(BaseModel):
    analysis: str


_MAX_CRITIC_RETRIES = 1  # critic 不通过时，最多再问一次（不重新生成图）


def _collect_pngs(state: MathModelingState) -> list[tuple[str, str, str]]:
    """返回 [(path, purpose, context_text), ...]"""
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for art in state.latest_code_artifacts():
        if not art.success:
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
            )
            if critic.approved:
                break

        analysis: FigureAnalysisOut = complete(
            fa_prompt(purpose, context),
            schema=FigureAnalysisOut, system=FA_SYSTEM,
            model=MODEL_ROUTING["figure_analyst"],
            images=[url],
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
