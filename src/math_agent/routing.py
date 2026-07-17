"""集中放图的条件边路由函数，便于单元测试。

约定的返回值是字符串字面量，graph.py 会把它映射到具体节点名。
"""
from math_agent.config import (
    MAX_MODEL_ITERATIONS, MAX_WRITER_ITERATIONS,
    MAX_BLUEPRINT_ITERATIONS, MAX_CODE_VERIFY_ITERATIONS,
)
from math_agent.state import MathModelingState


def after_blueprint_critic(state: MathModelingState) -> str:
    """blueprint_critic 审查完后的去向。

    blueprint_iteration 语义：blueprint_critic_node 返回时已递增。
    - == 0：尚未审查（不应到此，但防御性返回 retry）
    - == 1：analyst 首轮输出已被审查一次。若未通过，允许一次 retry。
    - >= 2：第二次审查仍未通过，则带警告前进（最多一次 retry）。
    """
    report = state.latest_critic("analyst", critic_type="blueprint")
    if report is None:
        return "retry"
    if report.approved:
        return "advance"
    if state.blueprint_iteration >= MAX_BLUEPRINT_ITERATIONS:
        return "advance_with_warning"
    return "retry"


def after_model_critic(state: MathModelingState) -> str:
    """basic/improved/final 任一阶段评审完后的去向。"""
    critic = state.latest_critic("modeler")
    if critic is None:
        return "retry"

    if state.stage_target == "final":
        if critic.approved or state.iteration >= MAX_MODEL_ITERATIONS:
            return "to_coder"
        return "retry"

    # basic / improved
    if critic.approved or state.iteration >= MAX_MODEL_ITERATIONS:
        return "advance"
    return "retry"


def after_paper_critic(state: MathModelingState) -> str:
    """writer 闭环：critic 通过或迭代用尽 → advance；否则 retry 回 writer。"""
    paper_has_content = any([
        state.paper.abstract, state.paper.model_section, state.paper.solution,
    ])
    if not paper_has_content:
        return "stop" if state.writer_iteration >= MAX_WRITER_ITERATIONS else "retry"
    critic = state.latest_critic("paper")
    if critic is None:
        return "stop" if state.writer_iteration >= MAX_WRITER_ITERATIONS else "retry"
    if critic.approved or state.writer_iteration >= MAX_WRITER_ITERATIONS:
        return "advance"
    return "retry"


def after_writer_step(state: MathModelingState) -> str:
    """writer 子流程：队列还有章节 -> 继续写 section；空 -> 进 paper_critic。

    prep（writer_node）和 section（writer_section_node）共用此路由。
    """
    return "section" if state.writer_section_queue else "done"


def after_modeler_work(state: MathModelingState) -> str:
    """模型子流程：每个推导步骤和一致性检查分别形成 checkpoint。"""
    return state.modeler_phase if state.modeler_phase in {"derive", "check"} else "done"


def after_coder_work(state: MathModelingState) -> str:
    if state.coder_work_queue and state.coder_phase in {"generate", "execute"}:
        return state.coder_phase
    return "done"


def after_sensitivity_work(state: MathModelingState) -> str:
    valid = {"code_generate", "code_execute", "interpret"}
    return state.sensitivity_phase if state.sensitivity_phase in valid else "done"


def after_figure_work(state: MathModelingState) -> str:
    return state.figure_phase if state.figure_phase in {"critic", "analysis"} else "done"


def after_model_code_consistency(state: MathModelingState) -> str:
    """model_code_consistency 审查完后的去向。

    code_verify_iteration 语义：model_code_consistency_node 返回时已递增。
    approved 且 score >= 7 -> advance；未通过且迭代用尽 -> advance_with_warning；否则 retry_coder。
    """
    if not state.model_code_reports:
        return "retry_coder"

    report = state.model_code_reports[-1]
    if report.approved and report.score >= 7:
        return "advance"
    latest = state.latest_code_artifacts()
    has_primary = any(
        artifact.success and artifact.evidence_role == "primary"
        for artifact in latest
    )
    # 没有主证据不是“低分但可继续”的软问题，任何重试上限都不能把它放行到
    # sensitivity/writer。下一 coder 批次可使用确定性安全求解器或明确失败。
    if not has_primary:
        return "retry_coder"
    if state.code_verify_iteration >= MAX_CODE_VERIFY_ITERATIONS:
        return "advance_with_warning"
    return "retry_coder"


def after_human_review(state: MathModelingState) -> str:
    """只有明确批准才进入最终 LaTeX；拒绝或缺少决定都停止。"""
    decision = state.human_decision
    return "finalize" if decision is not None and decision.approved else "stop"
