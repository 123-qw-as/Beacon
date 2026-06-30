from pydantic import BaseModel
from math_agent.llm import complete
from math_agent.config import (
    MODEL_ROUTING, RAG_ENABLED, RAG_DB_PATH, RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_DIM, RAG_TOPK, RAG_CTX_MAX_CHARS_ANALYST,
)
from math_agent.prompts.analyst import SYSTEM, build_prompt
from math_agent.rag.retrieve import search, format_snippets
from math_agent.state import Assumption, MathModelingState


class AnalystOutput(BaseModel):
    assumptions: list[Assumption]


def analyst_node(state: MathModelingState) -> dict:
    ctx = ""
    if RAG_ENABLED:
        snippets = search(
            state.problem,
            db_path=RAG_DB_PATH, k=RAG_TOPK,
            embedding_model=RAG_EMBEDDING_MODEL, dim=RAG_EMBEDDING_DIM,
        )
        ctx = format_snippets(snippets, max_chars=RAG_CTX_MAX_CHARS_ANALYST)
    prompt = build_prompt(state.problem, state.background, state.questions,
                          retrieved_context=ctx)
    out: AnalystOutput = complete(
        prompt,
        schema=AnalystOutput,
        system=SYSTEM,
        model=MODEL_ROUTING["analyst"],
    )
    return {"assumptions": out.assumptions}
