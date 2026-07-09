from math_agent.llm import complete
from math_agent.config import MODEL_ROUTING
from math_agent.prompts.model_critic import SYSTEM, build_prompt
from math_agent.state import MathModelingState, CriticReport


def model_critic_node(state: MathModelingState) -> dict:
    model = state.latest_model()
    if model is None:
        raise ValueError("ModelCritic: 当前 state 中没有 model_versions")

    prompt = build_prompt(state.problem, state.assumptions, model,
                          blueprint=state.problem_blueprint)
    out: CriticReport = complete(
        prompt, schema=CriticReport, system=SYSTEM, model=MODEL_ROUTING["model_critic"]
    )
    out.target = "modeler"        # 防篡改
    out.stage = model.stage       # 标记 critic 针对的阶段，避免跨阶段污染（见 modeler_node）
    return {"critic_reports": [out]}
