from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from math_agent.llm import complete
from math_agent.config import (
    MODEL_ROUTING, RAG_ENABLED, RAG_DB_PATH, RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_DIM, RAG_TOPK, RAG_CTX_MAX_CHARS_WRITER,
)
from math_agent.prompts.writer import SYSTEM, build_prompt  # noqa: F401  (build_prompt 保留向后兼容)
from math_agent.prompts.writer_section import (
    WriterOutline,
    _sections_to_rewrite,
    build_outline_prompt,
    build_section_prompt,
    schema_for_group,
    writer_sections,
)
from math_agent.rag.retrieve import search, format_snippets
from math_agent.state import MathModelingState, PaperSections


def writer_node(state: MathModelingState) -> dict:
    """两阶段撰写：Pass 1 大纲 → Pass 2 逐章填充。

    - RAG 只在开头查一次，复用于大纲与各分组。
    - 首轮（writer_iteration == 0）：跑大纲 + 全部 7 分组（共 8 次 LLM 调用）。
    - 重试轮（writer_iteration > 0）：跳过大纲，只重跑 paper_critic 标记的分组；
      未标记分组继承上一轮 state.paper 的文本。
    """
    # ---- RAG：查询一次，全组复用 ----
    ctx = ""
    if RAG_ENABLED:
        # 拿当前 paper 的模型部分（若有）做查询补全；首轮 paper 为空时仅用 problem
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

    # ---- 起点：重试轮从已有 paper 续写，未重写的分组原样保留 ----
    if state.writer_iteration > 0 and state.paper is not None:
        paper = state.paper.model_copy(deep=True)
    else:
        paper = PaperSections()

    # ---- Pass 1：大纲（仅首轮）----
    prior_critic = state.latest_critic("paper")
    if state.writer_iteration == 0:
        outline = complete(
            build_outline_prompt(state, retrieved_context=ctx),
            schema=WriterOutline,
            system=SYSTEM,
            model=MODEL_ROUTING["writer"],
        )
    else:
        # 重试轮跳过大纲：章节已有文本，critic 只想要局部修正。
        # 用空 WriterOutline 作锚点（锚点仅起引导作用，非强制）。
        outline = WriterOutline()

    # ---- 决定本轮要重跑的分组 ----
    if state.writer_iteration > 0 and prior_critic is not None and prior_critic.issues:
        groups_to_run = set(_sections_to_rewrite(prior_critic.issues))
    else:
        groups_to_run = {g.name for g in writer_sections()}

    # ---- Plan D: 检索参考文献一次，references 分组复用 ----
    references = None
    if "references" in groups_to_run:
        from math_agent.tools.references import select_references
        references = select_references(state.problem, state.problem_domains)

    # ---- Pass 2：逐章填充 ----
    for group in writer_sections():
        if group.name not in groups_to_run:
            continue  # 保留 paper 中已有的该组文本
        refs_for_group = references if group.name == "references" else None
        section_out = complete(
            build_section_prompt(
                group.name, state, outline,
                prior_critic=prior_critic,
                retrieved_context=ctx,
                references_list=refs_for_group,
            ),
            schema=schema_for_group(group.name),
            system=SYSTEM,
            model=MODEL_ROUTING["writer"],
        )
        for field in group.fields:
            setattr(paper, field, getattr(section_out, field))

    return {
        "paper": paper,
        "writer_iteration": state.writer_iteration + 1,
    }


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=select_autoescape([]))


def render_markdown(state: MathModelingState) -> str:
    from math_agent.nodes.latex import _curate_code, _curate_stdout
    tmpl = _env.get_template("paper.md.j2")
    curated = [
        {
            "purpose": a.purpose, "code": a.code, "stdout": a.stdout,
            "success": a.success, "artifact_paths": a.artifact_paths,
            "curated_code": _curate_code(a.code),
            "curated_stdout": _curate_stdout(a.stdout),
        }
        for a in state.code_artifacts if a.success
    ]
    return tmpl.render(
        problem=state.problem,
        paper=state.paper,
        code_artifacts=curated,
        figures=state.figures,
        sensitivity_runs=state.sensitivity_runs,
    )
