from math_agent.llm import complete
from math_agent.config import (
    MODEL_ROUTING, RAG_ENABLED, RAG_DB_PATH, RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_DIM, RAG_TOPK, RAG_CTX_MAX_CHARS_ANALYST,
)
from math_agent.prompts.analyst import SYSTEM, build_prompt
from math_agent.rag.retrieve import search, format_snippets
from math_agent.state import MathModelingState, ProblemBlueprint

# 向后兼容别名：旧测试和导入引用 AnalystOutput。实际输出已改为 ProblemBlueprint。
AnalystOutput = ProblemBlueprint


def analyst_node(state: MathModelingState) -> dict:
    ctx = ""
    if RAG_ENABLED:
        snippets = search(
            state.problem,
            db_path=RAG_DB_PATH, k=RAG_TOPK,
            embedding_model=RAG_EMBEDDING_MODEL, dim=RAG_EMBEDDING_DIM,
        )
        ctx = format_snippets(snippets, max_chars=RAG_CTX_MAX_CHARS_ANALYST)

    # 如果上一轮 blueprint_critic 未通过，把 issues/suggestions 注入 prompt
    critic_fb = state.latest_critic("analyst", critic_type="blueprint")
    if critic_fb is not None and critic_fb.approved:
        critic_fb = None

    prompt = build_prompt(state.problem, state.background, state.questions,
                          retrieved_context=ctx, critic_feedback=critic_fb)
    blueprint: ProblemBlueprint = complete(
        prompt,
        schema=ProblemBlueprint,
        system=SYSTEM,
        model=MODEL_ROUTING["analyst"],
    )

    delta: dict = {"problem_blueprint": blueprint}
    # problem_domains 是覆盖语义字段，每次 analyst 输出都同步更新
    # （题目理解调整后，参考文献方向也应随之更新 -- 这是预期行为）
    delta["problem_domains"] = blueprint.problem_domains
    # assumptions 是追加字段。仅在首次（blueprint_iteration == 0）同步旧字段，
    # 避免 analyst retry 时重复追加。
    if state.blueprint_iteration == 0:
        delta["assumptions"] = blueprint.assumptions
    return delta
