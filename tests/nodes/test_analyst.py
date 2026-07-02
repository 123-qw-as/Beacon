from math_agent.state import MathModelingState
from math_agent.nodes.analyst import analyst_node, AnalystOutput


def test_analyst_appends_assumptions(mocker):
    fake = AnalystOutput(
        assumptions=[
            {"statement": "需求服从泊松", "rationale": "短时间高频独立到达"},
            {"statement": "调度车恒速", "rationale": "市区均速近似"},
        ]
    )
    mocker.patch("math_agent.nodes.analyst.complete", return_value=fake)

    state = MathModelingState(problem="共享单车调度", questions=["预测", "调度"])
    delta = analyst_node(state)

    assert "assumptions" in delta
    assert len(delta["assumptions"]) == 2
    assert delta["assumptions"][0].statement == "需求服从泊松"


def test_analyst_does_not_query_rag_when_disabled(mocker):
    mocker.patch("math_agent.nodes.analyst.RAG_ENABLED", False)
    spy = mocker.patch("math_agent.nodes.analyst.search")
    mocker.patch(
        "math_agent.nodes.analyst.complete",
        return_value=AnalystOutput(assumptions=[]),
    )
    analyst_node(MathModelingState(problem="p"))
    spy.assert_not_called()


def test_analyst_queries_rag_when_enabled(mocker):
    mocker.patch("math_agent.nodes.analyst.RAG_ENABLED", True)
    mocker.patch("math_agent.nodes.analyst.RAG_DB_PATH", "/tmp/nonexistent.db")
    spy = mocker.patch("math_agent.nodes.analyst.search", return_value=[])
    mocker.patch(
        "math_agent.nodes.analyst.complete",
        return_value=AnalystOutput(assumptions=[]),
    )
    analyst_node(MathModelingState(problem="p"))
    spy.assert_called_once()


def test_analyst_outputs_problem_domains(mocker):
    from math_agent.state import Assumption
    mocker.patch("math_agent.nodes.analyst.complete",
                 return_value=AnalystOutput(
                     assumptions=[Assumption(statement="a", rationale="r")],
                     problem_domains=["optimization", "queueing"],
                 ))
    s = MathModelingState(problem="p")
    delta = analyst_node(s)
    assert delta["problem_domains"] == ["optimization", "queueing"]
