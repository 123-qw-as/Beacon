from math_agent.config import MODEL_ROUTING
from math_agent.llm import complete
from math_agent.prompts.paper_critic import SYSTEM, build_prompt
from math_agent.state import CriticIssue, CriticReport, MathModelingState


def _last_successful_stdout(state: MathModelingState) -> str:
    """最后一个 success=True 的 code_artifact.stdout。没有则空串。"""
    for art in reversed(state.latest_code_artifacts()):
        if art.success:
            return art.stdout
    return ""


def paper_critic_node(state: MathModelingState) -> dict:
    p = state.paper
    if not any([p.abstract, p.model_section, p.solution]):
        report = CriticReport(
            target="paper", score=0, approved=False,
            issues=[CriticIssue(section="general", problem="论文初稿为空")],
            suggestions=["重新生成全部论文章节后再评审"],
        )
        return {
            "critic_reports": [report],
            "errors": ["paper_critic: 论文初稿为空，无法进入最终环节"],
        }

    out: CriticReport = complete(
        build_prompt(p, len(state.figures), len(state.sensitivity_runs),
                     _last_successful_stdout(state)),
        schema=CriticReport, system=SYSTEM,
        model=MODEL_ROUTING["paper_critic"],
    )
    out.target = "paper"
    return {"critic_reports": [out]}
