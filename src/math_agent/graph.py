"""LangGraph 图构建。"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from math_agent.state import MathModelingState
from math_agent.nodes.analyst import analyst_node
from math_agent.nodes.blueprint_critic import blueprint_critic_node
from math_agent.nodes.modeler import (
    modeler_prepare_node, modeler_derivation_node, modeler_consistency_node,
)
from math_agent.nodes.model_critic import model_critic_node
from math_agent.nodes.coder import coder_prepare_node, coder_generate_node, coder_execute_node
from math_agent.nodes.model_code_consistency import model_code_consistency_node
from math_agent.nodes.sensitivity import (
    sensitivity_plan_node, sensitivity_code_generate_node,
    sensitivity_code_execute_node, sensitivity_interpret_node,
)
from math_agent.nodes.figure_pipeline import (
    figure_prepare_node, figure_critic_node, figure_analysis_node,
)
from math_agent.nodes.writer import writer_node, writer_section_node
from math_agent.nodes.paper_critic import paper_critic_node
from math_agent.nodes.evaluation import evaluation_node
from math_agent.nodes.human_review import human_review_node
from math_agent.nodes.latex_node import latex_node
from math_agent.nodes.finalizer import finalizer_node
from math_agent.nodes.table_assembler import table_assembler_node
from math_agent.routing import (
    after_blueprint_critic, after_model_critic, after_paper_critic,
    after_writer_step, after_model_code_consistency, after_human_review,
    after_modeler_work, after_coder_work, after_sensitivity_work, after_figure_work,
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
        except BaseException as exc:
            # 活跃节点 contextvar 会在 finally 中恢复；另存失败节点供 CLI 报错。
            record_failed_node(name)
            # LangGraph 可能在线程池上下文中执行节点，contextvar 的写入不会反向
            # 传播到调用线程。把节点名附到异常对象上，跨线程重新抛出后仍可读取。
            try:
                setattr(exc, "_math_agent_failed_node", name)
            except Exception:
                pass
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
    g.add_node("modeler", _wrap(modeler_prepare_node, "modeler"))
    g.add_node("modeler_derivation", _wrap(modeler_derivation_node, "modeler_derivation"))
    g.add_node("modeler_consistency", _wrap(modeler_consistency_node, "modeler_consistency"))
    g.add_node("model_critic", _wrap(model_critic_node, "model_critic"))
    g.add_node("advance_stage", _wrap(_advance_stage, "advance_stage"))
    g.add_node("coder", _wrap(coder_prepare_node, "coder"))
    g.add_node("coder_generate", _wrap(coder_generate_node, "coder_generate"))
    g.add_node("coder_execute", _wrap(coder_execute_node, "coder_execute"))
    g.add_node("model_code_consistency", _wrap(model_code_consistency_node, "model_code_consistency"))
    g.add_node("sensitivity", _wrap(sensitivity_plan_node, "sensitivity"))
    g.add_node("sensitivity_code_generate", _wrap(sensitivity_code_generate_node, "sensitivity_code_generate"))
    g.add_node("sensitivity_code_execute", _wrap(sensitivity_code_execute_node, "sensitivity_code_execute"))
    g.add_node("sensitivity_interpret", _wrap(sensitivity_interpret_node, "sensitivity_interpret"))
    g.add_node("figure_pipeline", _wrap(figure_prepare_node, "figure_pipeline"))
    g.add_node("figure_critic", _wrap(figure_critic_node, "figure_critic"))
    g.add_node("figure_analysis", _wrap(figure_analysis_node, "figure_analysis"))
    g.add_node("writer", _wrap(writer_node, "writer"))
    g.add_node("writer_section", _wrap(writer_section_node, "writer_section"))
    g.add_node("paper_critic", _wrap(paper_critic_node, "paper_critic"))
    g.add_node("evaluation", _wrap(evaluation_node, "evaluation"))
    g.add_node("human_review", _wrap(human_review_node, "human_review"))
    g.add_node("latex", _wrap(latex_node, "latex"))
    g.add_node("finalizer", _wrap(finalizer_node, "finalizer"))
    g.add_node("table_assembler", _wrap(table_assembler_node, "table_assembler"))

    g.set_entry_point("analyst")
    # analyst -> blueprint_critic -> (retry analyst / advance / advance_with_warning) -> modeler
    g.add_edge("analyst", "blueprint_critic")
    g.add_conditional_edges(
        "blueprint_critic",
        after_blueprint_critic,
        {"retry": "analyst", "advance": "modeler", "advance_with_warning": "modeler"},
    )
    g.add_conditional_edges(
        "modeler", after_modeler_work,
        {"derive": "modeler_derivation", "check": "modeler_consistency", "done": "model_critic"},
    )
    g.add_conditional_edges(
        "modeler_derivation", after_modeler_work,
        {"derive": "modeler_derivation", "check": "modeler_consistency", "done": "model_critic"},
    )
    g.add_edge("modeler_consistency", "model_critic")
    g.add_conditional_edges(
        "model_critic",
        after_model_critic,
        {"retry": "modeler", "advance": "advance_stage", "to_coder": "coder"},
    )
    g.add_edge("advance_stage", "modeler")
    # coder -> model_code_consistency -> (retry coder / advance / advance_with_warning) -> sensitivity
    g.add_conditional_edges(
        "coder", after_coder_work,
        {"generate": "coder_generate", "execute": "coder_execute", "done": "model_code_consistency"},
    )
    g.add_conditional_edges(
        "coder_generate", after_coder_work,
        {"generate": "coder_generate", "execute": "coder_execute", "done": "model_code_consistency"},
    )
    g.add_conditional_edges(
        "coder_execute", after_coder_work,
        {"generate": "coder_generate", "execute": "coder_execute", "done": "model_code_consistency"},
    )
    g.add_conditional_edges(
        "model_code_consistency",
        after_model_code_consistency,
        {"retry_coder": "coder", "advance": "sensitivity", "advance_with_warning": "sensitivity"},
    )
    g.add_conditional_edges(
        "sensitivity", after_sensitivity_work,
        {"code_generate": "sensitivity_code_generate", "code_execute": "sensitivity_code_execute",
         "interpret": "sensitivity_interpret", "done": "figure_pipeline"},
    )
    g.add_conditional_edges(
        "sensitivity_code_generate", after_sensitivity_work,
        {"code_generate": "sensitivity_code_generate", "code_execute": "sensitivity_code_execute",
         "interpret": "sensitivity_interpret", "done": "figure_pipeline"},
    )
    g.add_conditional_edges(
        "sensitivity_code_execute", after_sensitivity_work,
        {"code_generate": "sensitivity_code_generate", "code_execute": "sensitivity_code_execute",
         "interpret": "sensitivity_interpret", "done": "figure_pipeline"},
    )
    g.add_edge("sensitivity_interpret", "figure_pipeline")
    g.add_conditional_edges(
        "figure_pipeline", after_figure_work,
        {"critic": "figure_critic", "analysis": "figure_analysis", "done": "writer"},
    )
    g.add_conditional_edges(
        "figure_critic", after_figure_work,
        {"critic": "figure_critic", "analysis": "figure_analysis", "done": "writer"},
    )
    g.add_conditional_edges(
        "figure_analysis", after_figure_work,
        {"critic": "figure_critic", "analysis": "figure_analysis", "done": "writer"},
    )
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
    g.add_edge("latex", "finalizer")
    g.add_edge("finalizer", END)
    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or [],
    )
