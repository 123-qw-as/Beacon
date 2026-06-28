"""human_review 节点：被 graph 在 interrupt 之后运行；
预期 state.human_decision 已被外部填充。
"""
from math_agent.state import MathModelingState


def human_review_node(state: MathModelingState) -> dict:
    if state.human_decision is None:
        return {"errors": ["human_review: 恢复后未发现 human_decision，请填入后再恢复"]}
    return {}
