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
    """writer prep：RAG 检索 + 大纲（首轮）+ 决定本轮写哪些 group + 填充队列。

    拆成 prep + section 循环后，每次 writer_section_node 完成即一个 LangGraph
    checkpoint，单节卡住/崩溃不丢已完成节（recover 从队列断点续跑）。
    """
    # ---- RAG：查询一次，全组复用 ----
    ctx = ""
    if RAG_ENABLED:
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

    # ---- 大纲（仅首轮）----
    prior_critic = state.latest_critic("paper")
    if state.writer_iteration == 0:
        outline = complete(
            build_outline_prompt(state, retrieved_context=ctx),
            schema=WriterOutline,
            system=SYSTEM,
            model=MODEL_ROUTING["writer"],
        )
    else:
        outline = WriterOutline()

    # ---- 决定本轮要重跑的分组 ----
    if state.writer_iteration > 0 and prior_critic is not None and prior_critic.issues:
        run_set = set(_sections_to_rewrite(prior_critic.issues))
    else:
        run_set = {g.name for g in writer_sections()}

    queue = [g.name for g in writer_sections() if g.name in run_set]

    return {
        "writer_section_queue": queue,
        "writer_outline_dump": outline.model_dump(),
        "writer_retrieved_context": ctx,
        # writer_iteration 在 prep（而非完整 writer->critic 循环后）递增：
        # routing 的 after_paper_critic 用 writer_iteration >= MAX_WRITER_ITERATIONS 判定
        # "是否允许再 retry 回 writer"。prep 进入即代表一轮 writer 开始，
        # 所以 prep 递增保证：iteration=1 首轮写完后 critic 可 retry 回 writer(iteration=2)，
        # iteration=2 写完后 critic 若仍不通过则 advance（>= MAX=2）。
        "writer_iteration": state.writer_iteration + 1,
    }


def writer_section_node(state: MathModelingState) -> dict:
    """写队首一节，弹出队列。每次完成 = 一个 LangGraph checkpoint。"""
    queue = list(state.writer_section_queue)
    group_name = queue.pop(0)
    print(f"[writer] writing section: {group_name} ({len(queue)} remaining)", flush=True)

    outline = WriterOutline(**state.writer_outline_dump)
    prior_critic = state.latest_critic("paper")

    # references 分组专用：检索参考文献一次
    refs = None
    if group_name == "references":
        from math_agent.tools.references import select_references
        refs = select_references(state.problem, state.problem_domains)

    section_out = complete(
        build_section_prompt(
            group_name, state, outline,
            prior_critic=prior_critic,
            retrieved_context=state.writer_retrieved_context,
            references_list=refs,
        ),
        schema=schema_for_group(group_name),
        system=SYSTEM,
        model=MODEL_ROUTING["writer"],
    )

    # 累积到 paper（读-改-写，覆盖语义）
    paper = state.paper.model_copy(deep=True)
    for group in writer_sections():
        if group.name == group_name:
            for field in group.fields:
                setattr(paper, field, getattr(section_out, field))
            break

    return {"paper": paper, "writer_section_queue": queue}


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
