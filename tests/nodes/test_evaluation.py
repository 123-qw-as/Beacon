import pytest
from math_agent.state import (
    MathModelingState, PaperSections, FigureArtifact, SensitivityRun,
    CriticReport, EvaluationReport,
)
from math_agent.nodes.evaluation import evaluation_node


def _full_state():
    s = MathModelingState(problem="p")
    s.paper = PaperSections(
        abstract="a"*200, problem_restatement="x"*200, assumptions="x"*200,
        notation="x"*200, model_section="x"*200, solution="x"*200,
        sensitivity="x"*200, conclusion="x"*200, references="-",
    )
    s.figures.append(FigureArtifact(path="a.png", purpose="t", quality_score=8))
    s.sensitivity_runs.append(SensitivityRun(parameter="a", values=[1], metric="m", results=[1]))
    s.critic_reports.append(CriticReport(target="paper", score=8, approved=True))
    return s


def test_evaluation_returns_report(mocker):
    fake = EvaluationReport(
        assumption_reasonableness=8, modeling_creativity=8,
        result_correctness=8, writing_clarity=8, extra_depth=8, overall=8.0,
        issues=[], suggestions=[],
    )
    mocker.patch("math_agent.nodes.evaluation.complete", return_value=fake)
    delta = evaluation_node(_full_state())
    assert isinstance(delta["evaluation"], EvaluationReport)
    assert delta["evaluation"].overall == 8.0


def test_evaluation_recomputes_overall_if_llm_wrong(mocker):
    fake = EvaluationReport(
        assumption_reasonableness=8, modeling_creativity=8,
        result_correctness=8, writing_clarity=8, extra_depth=8, overall=10.0,
    )
    mocker.patch("math_agent.nodes.evaluation.complete", return_value=fake)
    delta = evaluation_node(_full_state())
    assert delta["evaluation"].overall == pytest.approx(8.0, abs=0.01)


def test_evaluation_skips_without_paper(mocker):
    s = MathModelingState(problem="p")
    delta = evaluation_node(s)
    assert delta["errors"]
    assert delta.get("evaluation") is None
