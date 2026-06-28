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
from math_agent.nodes.writer import writer_node
from math_agent.nodes.paper_critic import paper_critic_node
from math_agent.nodes.evaluation import evaluation_node
from math_agent.nodes.human_review import human_review_node
from math_agent.nodes.latex import latex_node
from math_agent.routing import after_model_critic


def _advance_stage(state: MathModelingState) -> dict:
    return {"stage_target": {"basic": "improved", "improved": "final"}[state.stage_target], "iteration": 0}


def build_graph(
    *,
    checkpointer=None,
    interrupt_before: list[str] | None = None,
):
    g = StateGraph(MathModelingState)
    g.add_node("analyst", analyst_node)
    g.add_node("modeler", modeler_node)
    g.add_node("model_critic", model_critic_node)
    g.add_node("advance_stage", _advance_stage)
    g.add_node("coder", coder_node)
    g.add_node("sensitivity", sensitivity_node)
    g.add_node("figure_pipeline", figure_pipeline_node)
    g.add_node("writer", writer_node)
    g.add_node("paper_critic", paper_critic_node)
    g.add_node("evaluation", evaluation_node)
    g.add_node("human_review", human_review_node)
    g.add_node("latex", latex_node)

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
    g.add_edge("writer", "paper_critic")
    g.add_edge("paper_critic", "evaluation")
    g.add_edge("evaluation", "human_review")
    g.add_edge("human_review", "latex")
    g.add_edge("latex", END)
    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or [],
    )
