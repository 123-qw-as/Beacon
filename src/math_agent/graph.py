"""LangGraph 图构建。"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from math_agent.state import MathModelingState
from math_agent.nodes.analyst import analyst_node
from math_agent.nodes.blueprint_critic import blueprint_critic_node
from math_agent.nodes.modeler import modeler_node
from math_agent.nodes.model_critic import model_critic_node
from math_agent.nodes.coder import coder_node
from math_agent.nodes.model_code_consistency import model_code_consistency_node
from math_agent.nodes.sensitivity import sensitivity_node
from math_agent.nodes.figure_pipeline import figure_pipeline_node
from math_agent.nodes.writer import writer_node, writer_section_node
from math_agent.nodes.paper_critic import paper_critic_node
from math_agent.nodes.evaluation import evaluation_node
from math_agent.nodes.human_review import human_review_node
from math_agent.nodes.latex_node import latex_node
from math_agent.nodes.table_assembler import table_assembler_node
from math_agent.routing import (
    after_blueprint_critic, after_model_critic, after_paper_critic,
    after_writer_step, after_model_code_consistency, after_human_review,
)


def _advance_stage(state: MathModelingState) -> dict:
    return {"stage_target": {"basic": "improved", "improved": "final"}[state.stage_target], "iteration": 0}


def _wrap(fn, name: str):
    """如当前 contextvar 上有 Tracer，按节点名打点；没有则原样调用。
    同时设置 _last_node_name contextvar 供 CLI 报错时显示当前节点。"""
    def _inner(s):
        from math_agent.tracing import (
            get_current, set_last_node, reset_last_node, record_failed_node,
        )
        tok = set_last_node(name)
        try:
            print(f"[pipeline] node: {name}", flush=True)
            tracer = get_current()
            if tracer is not None:
                with tracer.node(name):
                    return fn(s)
            return fn(s)
        except BaseException:
            # 活跃节点 contextvar 会在 finally 中恢复；另存失败节点供 CLI 报错。
            record_failed_node(name)
            raise
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
    g.add_node("blueprint_critic", _wrap(blueprint_critic_node, "blueprint_critic"))
    g.add_node("modeler", _wrap(modeler_node, "modeler"))
    g.add_node("model_critic", _wrap(model_critic_node, "model_critic"))
    g.add_node("advance_stage", _wrap(_advance_stage, "advance_stage"))
    g.add_node("coder", _wrap(coder_node, "coder"))
    g.add_node("model_code_consistency", _wrap(model_code_consistency_node, "model_code_consistency"))
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
    # analyst -> blueprint_critic -> (retry analyst / advance / advance_with_warning) -> modeler
    g.add_edge("analyst", "blueprint_critic")
    g.add_conditional_edges(
        "blueprint_critic",
        after_blueprint_critic,
        {"retry": "analyst", "advance": "modeler", "advance_with_warning": "modeler"},
    )
    g.add_edge("modeler", "model_critic")
    g.add_conditional_edges(
        "model_critic",
        after_model_critic,
        {"retry": "modeler", "advance": "advance_stage", "to_coder": "coder"},
    )
    g.add_edge("advance_stage", "modeler")
    # coder -> model_code_consistency -> (retry coder / advance / advance_with_warning) -> sensitivity
    g.add_edge("coder", "model_code_consistency")
    g.add_conditional_edges(
        "model_code_consistency",
        after_model_code_consistency,
        {"retry_coder": "coder", "advance": "sensitivity", "advance_with_warning": "sensitivity"},
    )
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
        {"retry": "writer", "advance": "table_assembler", "stop": END},
    )
    g.add_edge("table_assembler", "evaluation")
    g.add_edge("evaluation", "human_review")
    g.add_conditional_edges(
        "human_review",
        after_human_review,
        {"finalize": "latex", "stop": END},
    )
    g.add_edge("latex", END)
    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or [],
    )
