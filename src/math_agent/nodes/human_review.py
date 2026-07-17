"""human_review 节点：支持显式人工决策和无中断恢复的节点内自动批准。"""
import os

from math_agent.state import HumanDecision, MathModelingState


def human_review_node(state: MathModelingState) -> dict:
    if state.human_decision is None:
        if os.getenv("MATH_AGENT_AUTO_APPROVE_HUMAN_REVIEW", "").strip() == "1":
            return {
                "human_decision": HumanDecision(
                    approved=True,
                    notes="--no-interrupt recovery",
                )
            }
        return {"errors": ["human_review: 恢复后未发现 human_decision，请填入后再恢复"]}
    return {}
