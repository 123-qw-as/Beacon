"""LangGraph 图构建。"""
from __future__ import annotations

from pathlib import Path

from langgraph.graph import StateGraph, END

from math_agent.state import MathModelingState
from math_agent.nodes.analyst import analyst_node
from math_agent.nodes.modeler import modeler_node
from math_agent.nodes.model_critic import model_critic_node
from math_agent.nodes.coder import coder_node
from math_agent.nodes.writer import writer_node, render_markdown
from math_agent.routing import after_model_critic


def _advance_stage(state: MathModelingState) -> dict:
    return {"stage_target": {"basic": "improved", "improved": "final"}[state.stage_target], "iteration": 0}


def _finalize(state: MathModelingState) -> dict:
    if state.output_dir:
        out = Path(state.output_dir) / "paper.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(state), encoding="utf-8")
    return {}


def build_graph():
    g = StateGraph(MathModelingState)
    g.add_node("analyst", analyst_node)
    g.add_node("modeler", modeler_node)
    g.add_node("model_critic", model_critic_node)
    g.add_node("advance_stage", _advance_stage)
    g.add_node("coder", coder_node)
    g.add_node("writer", writer_node)
    g.add_node("finalize", _finalize)

    g.set_entry_point("analyst")
    g.add_edge("analyst", "modeler")
    g.add_edge("modeler", "model_critic")
    g.add_conditional_edges(
        "model_critic",
        after_model_critic,
        {"retry": "modeler", "advance": "advance_stage", "to_coder": "coder"},
    )
    g.add_edge("advance_stage", "modeler")
    g.add_edge("coder", "writer")
    g.add_edge("writer", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
