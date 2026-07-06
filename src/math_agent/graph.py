"""LangGraph 图构建。"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from math_agent.state import MathModelingState
from math_agent.nodes.analyst import analyst_node
from math_agent.nodes.modeler import modeler_node
from math_agent.nodes.model_critic import model_critic_node
from math_agent.nodes.coder import coder_node
from math_agent.nodes.sensitivity import sensitivity_node
from math_agent.nodes.figure_pipeline import figure_pipeline_node
from math_agent.nodes.writer import writer_node, writer_section_node
from math_agent.nodes.paper_critic import paper_critic_node
from math_agent.nodes.evaluation import evaluation_node
from math_agent.nodes.human_review import human_review_node
from math_agent.nodes.latex import latex_node
from math_agent.nodes.table_assembler import table_assembler_node
from math_agent.routing import after_model_critic, after_paper_critic, after_writer_step


def _advance_stage(state: MathModelingState) -> dict:
    return {"stage_target": {"basic": "improved", "improved": "final"}[state.stage_target], "iteration": 0}


def _wrap(fn, name: str):
    """如当前 contextvar 上有 Tracer，按节点名打点；没有则原样调用。
    同时设置 _last_node_name contextvar 供 CLI 报错时显示当前节点。"""
    def _inner(s):
        from math_agent.tracing import get_current, set_last_node, reset_last_node
        tok = set_last_node(name)
        try:
            tracer = get_current()
            if tracer is not None:
                with tracer.node(name):
                    return fn(s)
            return fn(s)
        finally:
            reset_last_node(tok)
    _inner.__name__ = fn.__name__
    return _inner


def build_graph(
    *,
    checkpointer=None,
    interrupt_before: list[str] | None = None,
):
    g = StateGraph(MathModelingState)
    g.add_node("analyst", _wrap(analyst_node, "analyst"))
    g.add_node("modeler", _wrap(modeler_node, "modeler"))
    g.add_node("model_critic", _wrap(model_critic_node, "model_critic"))
    g.add_node("advance_stage", _wrap(_advance_stage, "advance_stage"))
    g.add_node("coder", _wrap(coder_node, "coder"))
    g.add_node("sensitivity", _wrap(sensitivity_node, "sensitivity"))
    g.add_node("figure_pipeline", _wrap(figure_pipeline_node, "figure_pipeline"))
    g.add_node("writer", _wrap(writer_node, "writer"))
    g.add_node("writer_section", _wrap(writer_section_node, "writer_section"))
    g.add_node("paper_critic", _wrap(paper_critic_node, "paper_critic"))
    g.add_node("evaluation", _wrap(evaluation_node, "evaluation"))
    g.add_node("human_review", _wrap(human_review_node, "human_review"))
    g.add_node("latex", _wrap(latex_node, "latex"))
    g.add_node("table_assembler", _wrap(table_assembler_node, "table_assembler"))

    g.set_entry_point("analyst")
    g.add_edge("analyst", "modeler")
    g.add_edge("modeler", "model_critic")
    g.add_conditional_edges(
        "model_critic",
        after_model_critic,
        {"retry": "modeler", "advance": "advance_stage", "to_coder": "coder"},
    )
    g.add_edge("advance_stage", "modeler")
    g.add_edge("coder", "sensitivity")
    g.add_edge("sensitivity", "figure_pipeline")
    g.add_edge("figure_pipeline", "writer")
    # writer prep → section 循环 → paper_critic。每次 section 完成 = 一个 checkpoint。
    g.add_conditional_edges(
        "writer",
        after_writer_step,
        {"section": "writer_section", "done": "paper_critic"},
    )
    g.add_conditional_edges(
        "writer_section",
        after_writer_step,
        {"section": "writer_section", "done": "paper_critic"},
    )
    g.add_conditional_edges(
        "paper_critic",
        after_paper_critic,
        {"retry": "writer", "advance": "table_assembler"},
    )
    g.add_edge("table_assembler", "evaluation")
    g.add_edge("evaluation", "human_review")
    g.add_edge("human_review", "latex")
    g.add_edge("latex", END)
    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or [],
    )
