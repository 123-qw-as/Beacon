from math_agent.state import MathModelingState, ModelVersion, CriticReport
from math_agent.routing import after_model_critic, after_paper_critic


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


def _state_with_paper_critic(score: int, approved: bool, writer_iter: int):
    s = MathModelingState(problem="p")
    s.writer_iteration = writer_iter
    s.critic_reports.append(CriticReport(
        target="paper", score=score, approved=approved,
        issues=["编造数字"], suggestions=["核对附录"],
    ))
    return s


def test_after_paper_critic_advances_when_approved():
    s = _state_with_paper_critic(score=9, approved=True, writer_iter=0)
    assert after_paper_critic(s) == "advance"


def test_after_paper_critic_retries_when_below_threshold_and_iter_left():
    s = _state_with_paper_critic(score=4, approved=False, writer_iter=0)
    assert after_paper_critic(s) == "retry"


def test_after_paper_critic_advances_when_iter_cap_hit():
    from math_agent.config import MAX_WRITER_ITERATIONS
    s = _state_with_paper_critic(score=4, approved=False, writer_iter=MAX_WRITER_ITERATIONS)
    assert after_paper_critic(s) == "advance"


def test_after_paper_critic_advances_when_no_critic():
    s = MathModelingState(problem="p")
    assert after_paper_critic(s) == "advance"
