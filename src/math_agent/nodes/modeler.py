from math_agent.llm import complete
from math_agent.config import MODEL_ROUTING
from math_agent.prompts.modeler import SYSTEM, build_prompt
from math_agent.state import MathModelingState, ModelVersion


def modeler_node(state: MathModelingState) -> dict:
    # 只关心针对**当前阶段**的上一版模型与上一份 critic，避免跨阶段污染。
    same_stage_prev = next(
        (m for m in reversed(state.model_versions) if m.stage == state.stage_target),
        None,
    )
    # 没有同阶段的上一版时，把上一阶段的最终版作为参考（用于 improved 起步）
    prev_for_stage = same_stage_prev or (state.model_versions[-1] if state.model_versions else None)

    critic_fb = next(
        (r for r in reversed(state.critic_reports)
         if r.target == "modeler" and r.stage == state.stage_target),
        None,
    )
    # 当前阶段已 approved，不再回灌反馈
    if critic_fb and critic_fb.approved:
        critic_fb = None

    prompt = build_prompt(
        state.problem, state.assumptions, prev_for_stage, state.stage_target, critic_fb
    )
    out: ModelVersion = complete(
        prompt, schema=ModelVersion, system=SYSTEM, model=MODEL_ROUTING["modeler"]
    )
    # 保证 stage 与请求一致（防 LLM 篡改）
    out.stage = state.stage_target
    return {"model_versions": [out], "iteration": state.iteration + 1}
