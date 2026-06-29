"""集中放图的条件边路由函数，便于单元测试。

约定的返回值是字符串字面量，graph.py 会把它映射到具体节点名。
"""
from math_agent.config import MAX_MODEL_ITERATIONS, MAX_WRITER_ITERATIONS
from math_agent.state import MathModelingState


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
    critic = state.latest_critic("paper")
    if critic is None:
        return "advance"
    if critic.approved or state.writer_iteration >= MAX_WRITER_ITERATIONS:
        return "advance"
    return "retry"
