from math_agent.llm import complete
from math_agent.config import MODEL_ROUTING
from math_agent.prompts.blueprint_critic import SYSTEM, build_prompt
from math_agent.state import MathModelingState, CriticReport


def blueprint_critic_node(state: MathModelingState) -> dict:
    blueprint = state.problem_blueprint
    if blueprint is None:
        # 没有蓝图无法审查；直接返回未通过，触发 retry
        report = CriticReport(
            target="analyst", score=0, approved=False,
            critic_type="blueprint",
            issues=[],
            suggestions=["analyst 未产出 problem_blueprint，请重试"],
        )
        return {"critic_reports": [report], "blueprint_iteration": state.blueprint_iteration + 1}

    blueprint_json = blueprint.model_dump_json(indent=2)
    prompt = build_prompt(blueprint_json, state.problem, state.questions)
    out: CriticReport = complete(
        prompt, schema=CriticReport, system=SYSTEM, model=MODEL_ROUTING["model_critic"]
    )
    out.target = "analyst"  # 防篡改
    out.critic_type = "blueprint"  # 区分同 target 的不同 reviewer
    return {
        "critic_reports": [out],
        "blueprint_iteration": state.blueprint_iteration + 1,
    }
