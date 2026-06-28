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
