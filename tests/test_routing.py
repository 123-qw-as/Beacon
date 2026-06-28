from math_agent.state import MathModelingState, ModelVersion, CriticReport
from math_agent.routing import after_model_critic


def _state_with(stage, score, iteration):
    s = MathModelingState(problem="x", iteration=iteration, stage_target=stage)
    s.model_versions.append(ModelVersion(stage=stage, description="d"))
    s.critic_reports.append(
        CriticReport(target="modeler", score=score, approved=score >= 8, stage=stage)
    )
    return s


def test_routing_approved_basic_goes_to_improved():
    assert after_model_critic(_state_with("basic", 9, 0)) == "advance"


def test_routing_low_score_retries():
    assert after_model_critic(_state_with("basic", 4, 0)) == "retry"


def test_routing_caps_retries():
    # 即使分数低，达到迭代上限也必须前进
    assert after_model_critic(_state_with("basic", 4, 3)) == "advance"


def test_routing_after_final_goes_to_coder():
    assert after_model_critic(_state_with("final", 9, 0)) == "to_coder"
