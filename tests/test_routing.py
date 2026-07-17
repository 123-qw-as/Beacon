from math_agent.state import MathModelingState, ModelVersion, CriticReport, CriticIssue, ModelCodeConsistencyReport
from math_agent.routing import (
    after_model_critic, after_paper_critic,
    after_blueprint_critic, after_model_code_consistency,
)


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
    s.paper.abstract = "非空论文"
    s.writer_iteration = writer_iter
    s.critic_reports.append(CriticReport(
        target="paper", score=score, approved=approved,
        issues=[CriticIssue(section="general", problem="编造数字")], suggestions=["核对附录"],
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


def test_after_paper_critic_retries_when_no_critic():
    s = MathModelingState(problem="p")
    s.paper.abstract = "非空论文"
    assert after_paper_critic(s) == "retry"


def test_after_paper_critic_stops_empty_paper_at_iteration_cap():
    from math_agent.config import MAX_WRITER_ITERATIONS
    s = MathModelingState(problem="p", writer_iteration=MAX_WRITER_ITERATIONS)
    assert after_paper_critic(s) == "stop"


# ---------------------------------------------------------------------------
# after_blueprint_critic
# ---------------------------------------------------------------------------

def _state_with_blueprint_critic(approved: bool, iteration: int):
    s = MathModelingState(problem="p", blueprint_iteration=iteration)
    s.critic_reports.append(CriticReport(
        target="analyst", score=9 if approved else 4, approved=approved,
        critic_type="blueprint",
    ))
    return s


def test_blueprint_critic_advances_when_approved():
    s = _state_with_blueprint_critic(approved=True, iteration=1)
    assert after_blueprint_critic(s) == "advance"


def test_blueprint_critic_retries_when_not_approved_and_under_cap():
    s = _state_with_blueprint_critic(approved=False, iteration=1)
    assert after_blueprint_critic(s) == "retry"


def test_blueprint_critic_advances_with_warning_at_cap():
    from math_agent.config import MAX_BLUEPRINT_ITERATIONS
    s = _state_with_blueprint_critic(approved=False, iteration=MAX_BLUEPRINT_ITERATIONS)
    assert after_blueprint_critic(s) == "advance_with_warning"


def test_blueprint_critic_retries_when_no_report():
    s = MathModelingState(problem="p", blueprint_iteration=0)
    assert after_blueprint_critic(s) == "retry"


# ---------------------------------------------------------------------------
# after_model_code_consistency
# ---------------------------------------------------------------------------

def _state_with_consistency_report(approved: bool, score: int, iteration: int):
    s = MathModelingState(problem="p", code_verify_iteration=iteration)
    s.model_code_reports.append(ModelCodeConsistencyReport(score=score, approved=approved))
    return s


def test_consistency_advances_when_approved_and_high_score():
    s = _state_with_consistency_report(approved=True, score=8, iteration=1)
    assert after_model_code_consistency(s) == "advance"


def test_consistency_retries_coder_when_not_approved():
    s = _state_with_consistency_report(approved=False, score=4, iteration=1)
    assert after_model_code_consistency(s) == "retry_coder"


def test_consistency_retries_when_approved_but_low_score():
    """approved=True 但 score < 7 仍不 advance。"""
    s = _state_with_consistency_report(approved=True, score=5, iteration=1)
    assert after_model_code_consistency(s) == "retry_coder"


def test_consistency_advances_with_warning_at_cap():
    from math_agent.config import MAX_CODE_VERIFY_ITERATIONS
    from math_agent.state import CodeArtifact
    s = _state_with_consistency_report(approved=False, score=4, iteration=MAX_CODE_VERIFY_ITERATIONS)
    s.code_artifacts.append(CodeArtifact(
        purpose="primary", code="print(1)", success=True,
        evidence_role="primary", batch=1,
    ))
    assert after_model_code_consistency(s) == "advance_with_warning"


def test_consistency_never_advances_without_primary_even_at_cap():
    from math_agent.config import MAX_CODE_VERIFY_ITERATIONS
    s = _state_with_consistency_report(
        approved=False, score=0, iteration=MAX_CODE_VERIFY_ITERATIONS,
    )
    assert after_model_code_consistency(s) == "retry_coder"


def test_consistency_retries_coder_when_no_reports():
    s = MathModelingState(problem="p", code_verify_iteration=0)
    assert after_model_code_consistency(s) == "retry_coder"


def test_human_review_routes_rejection_to_stop():
    from math_agent.routing import after_human_review
    from math_agent.state import HumanDecision
    s = MathModelingState(problem="p", human_decision=HumanDecision(approved=False))
    assert after_human_review(s) == "stop"


def test_human_review_routes_approval_to_finalize():
    from math_agent.routing import after_human_review
    from math_agent.state import HumanDecision
    s = MathModelingState(problem="p", human_decision=HumanDecision(approved=True))
    assert after_human_review(s) == "finalize"
