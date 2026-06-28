from math_agent.config import MODEL_ROUTING
from math_agent.llm import complete
from math_agent.prompts.paper_critic import SYSTEM, build_prompt
from math_agent.state import CriticReport, MathModelingState


def paper_critic_node(state: MathModelingState) -> dict:
    p = state.paper
    if not any([p.abstract, p.model_section, p.solution]):
        return {"errors": ["paper_critic: 论文初稿为空，跳过整体评审"]}

    out: CriticReport = complete(
        build_prompt(p, len(state.figures), len(state.sensitivity_runs)),
        schema=CriticReport, system=SYSTEM,
        model=MODEL_ROUTING["paper_critic"],
    )
    out.target = "paper"
    return {"critic_reports": [out]}
