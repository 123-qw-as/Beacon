from math_agent.state import (
    MathModelingState, PaperSections, FigureArtifact, SensitivityRun, CriticReport,
)
from math_agent.nodes.paper_critic import paper_critic_node


def test_paper_critic_appends_report(mocker):
    fake = CriticReport(target="paper", score=8, issues=[], suggestions=[], approved=True)
    mocker.patch("math_agent.nodes.paper_critic.complete", return_value=fake)
    s = MathModelingState(problem="p")
    s.paper = PaperSections(
        abstract="a"*200, problem_restatement="b"*200, assumptions="c"*200,
        notation="d"*200, model_section="e"*200, solution="f"*200,
        sensitivity="g"*200, conclusion="h"*200, references="-",
    )
    s.figures.append(FigureArtifact(path="x.png", purpose="t"))
    s.sensitivity_runs.append(SensitivityRun(
        parameter="a", values=[1], metric="m", results=[1],
    ))
    delta = paper_critic_node(s)
    assert delta["critic_reports"][0].target == "paper"
    assert delta["critic_reports"][0].approved is True


def test_paper_critic_handles_missing_paper(mocker):
    s = MathModelingState(problem="p")
    delta = paper_critic_node(s)
    assert delta["errors"]
    assert delta.get("critic_reports", []) == []
