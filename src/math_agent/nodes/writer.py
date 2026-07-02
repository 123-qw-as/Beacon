from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from math_agent.llm import complete
from math_agent.config import (
    MODEL_ROUTING, RAG_ENABLED, RAG_DB_PATH, RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_DIM, RAG_TOPK, RAG_CTX_MAX_CHARS_WRITER,
)
from math_agent.prompts.writer import SYSTEM, build_prompt
from math_agent.rag.retrieve import search, format_snippets
from math_agent.state import MathModelingState, PaperSections


def writer_node(state: MathModelingState) -> dict:
    ctx = ""
    if RAG_ENABLED:
        # 拿当前 paper 的模型部分（若有）做查询补全；首轮 paper 为 None 时仅用 problem
        prev_paper_hint = ""
        if state.paper is not None and state.paper.model_section:
            prev_paper_hint = state.paper.model_section[:500]
        query = (state.problem + " " + prev_paper_hint).strip()
        snippets = search(
            query,
            db_path=RAG_DB_PATH, k=RAG_TOPK,
            embedding_model=RAG_EMBEDDING_MODEL, dim=RAG_EMBEDDING_DIM,
            source_type="paper",
        )
        ctx = format_snippets(snippets, max_chars=RAG_CTX_MAX_CHARS_WRITER)

    out: PaperSections = complete(
        build_prompt(state, retrieved_context=ctx),
        schema=PaperSections,
        system=SYSTEM,
        model=MODEL_ROUTING["writer"],
    )
    return {
        "paper": out,
        "writer_iteration": state.writer_iteration + 1,
    }


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=select_autoescape([]))


def render_markdown(state: MathModelingState) -> str:
    tmpl = _env.get_template("paper.md.j2")
    return tmpl.render(
        problem=state.problem,
        paper=state.paper,
        code_artifacts=state.code_artifacts,
        figures=state.figures,
        sensitivity_runs=state.sensitivity_runs,
    )
