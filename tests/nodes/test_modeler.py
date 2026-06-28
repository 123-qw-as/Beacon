from math_agent.state import MathModelingState, Assumption, ModelVersion, CriticReport
from math_agent.nodes.modeler import modeler_node


def test_modeler_produces_basic_first(mocker):
    fake = ModelVersion(stage="basic", description="d" * 200, equations=["x=1"], variables={"x": "ok"})
    mocker.patch("math_agent.nodes.modeler.complete", return_value=fake)

    s = MathModelingState(problem="p", stage_target="basic")
    s.assumptions.append(Assumption(statement="a", rationale="r"))
    delta = modeler_node(s)
    assert delta["model_versions"][0].stage == "basic"
    assert delta["iteration"] == 1


def test_modeler_passes_critic_feedback(mocker):
    spy = mocker.patch(
        "math_agent.nodes.modeler.complete",
        return_value=ModelVersion(stage="basic", description="d"*200, equations=["y=2"]),
    )
    s = MathModelingState(problem="p", stage_target="basic", iteration=1)
    s.assumptions.append(Assumption(statement="a", rationale="r"))
    s.model_versions.append(ModelVersion(stage="basic", description="old"))
    s.critic_reports.append(
        CriticReport(target="modeler", score=4, issues=["弱"], suggestions=["改"], stage="basic")
    )
    modeler_node(s)
    prompt_arg = spy.call_args.args[0]
    assert "弱" in prompt_arg and "改" in prompt_arg


def test_modeler_ignores_other_stage_critic(mocker):
    """basic 阶段未通过的 critic 不应污染 improved 阶段的 prompt。"""
    spy = mocker.patch(
        "math_agent.nodes.modeler.complete",
        return_value=ModelVersion(stage="improved", description="d"*200),
    )
    s = MathModelingState(problem="p", stage_target="improved", iteration=0)
    s.assumptions.append(Assumption(statement="a", rationale="r"))
    s.model_versions.append(ModelVersion(stage="basic", description="basic-final"))
    s.critic_reports.append(
        CriticReport(target="modeler", score=4, issues=["basic-issue"], suggestions=["basic-fix"], stage="basic")
    )
    modeler_node(s)
    prompt_arg = spy.call_args.args[0]
    assert "basic-issue" not in prompt_arg
    assert "basic-fix" not in prompt_arg
