from math_agent.llm import complete
from math_agent.config import (
    MODEL_ROUTING, RAG_ENABLED, RAG_DB_PATH, RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_DIM, RAG_TOPK, RAG_CTX_MAX_CHARS_MODELER,
)
from math_agent.prompts.modeler import SYSTEM, build_prompt
from math_agent.rag.retrieve import search, format_snippets
from math_agent.state import MathModelingState, ModelVersion


def modeler_node(state: MathModelingState) -> dict:
    # 只关心针对**当前阶段**的上一版模型与上一份 critic，避免跨阶段污染。
    same_stage_prev = next(
        (m for m in reversed(state.model_versions) if m.stage == state.stage_target),
        None,
    )
    # 没有同阶段的上一版时，把上一阶段的最终版作为参考（用于 improved 起步）
    prev_for_stage = same_stage_prev or (state.model_versions[-1] if state.model_versions else None)

    critic_fb = state.latest_critic_for_stage("modeler", state.stage_target)
    # 当前阶段已 approved，不再回灌反馈
    if critic_fb and critic_fb.approved:
        critic_fb = None

    ctx = ""
    if RAG_ENABLED:
        prev_desc = prev_for_stage.description if prev_for_stage else ""
        query = (state.problem + " " + state.stage_target + " " + prev_desc).strip()
        snippets = search(
            query,
            db_path=RAG_DB_PATH, k=RAG_TOPK,
            embedding_model=RAG_EMBEDDING_MODEL, dim=RAG_EMBEDDING_DIM,
        )
        ctx = format_snippets(snippets, max_chars=RAG_CTX_MAX_CHARS_MODELER)

    prompt = build_prompt(
        state.problem, state.assumptions, prev_for_stage, state.stage_target,
        critic_fb, retrieved_context=ctx,
    )
    out: ModelVersion = complete(
        prompt, schema=ModelVersion, system=SYSTEM, model=MODEL_ROUTING["modeler"]
    )
    # 保证 stage 与请求一致（防 LLM 篡改）
    out.stage = state.stage_target
    return {"model_versions": [out], "iteration": state.iteration + 1}
