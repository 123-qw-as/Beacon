from math_agent.state import MathModelingState, Assumption, ModelVersion, CriticReport, CriticIssue
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
        CriticReport(target="modeler", score=4, issues=[CriticIssue(problem="弱")], suggestions=["改"], stage="basic")
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
        CriticReport(target="modeler", score=4, issues=[CriticIssue(problem="basic-issue")], suggestions=["basic-fix"], stage="basic")
    )
    modeler_node(s)
    prompt_arg = spy.call_args.args[0]
    assert "basic-issue" not in prompt_arg
    assert "basic-fix" not in prompt_arg


def test_modeler_does_not_query_rag_when_disabled(mocker):
    mocker.patch("math_agent.nodes.modeler.RAG_ENABLED", False)
    spy = mocker.patch("math_agent.nodes.modeler.search")
    mocker.patch(
        "math_agent.nodes.modeler.complete",
        return_value=ModelVersion(stage="basic", description="d"*200),
    )
    s = MathModelingState(problem="p", stage_target="basic")
    s.assumptions.append(Assumption(statement="a", rationale="r"))
    modeler_node(s)
    spy.assert_not_called()


def test_modeler_queries_rag_when_enabled(mocker):
    mocker.patch("math_agent.nodes.modeler.RAG_ENABLED", True)
    mocker.patch("math_agent.nodes.modeler.RAG_DB_PATH", "/tmp/nonexistent.db")
    spy = mocker.patch("math_agent.nodes.modeler.search", return_value=[])
    mocker.patch(
        "math_agent.nodes.modeler.complete",
        return_value=ModelVersion(stage="basic", description="d"*200),
    )
    s = MathModelingState(problem="p", stage_target="basic")
    s.assumptions.append(Assumption(statement="a", rationale="r"))
    modeler_node(s)
    spy.assert_called_once()


def test_modeler_does_not_filter_source_type(mocker):
    """modeler 两类语料都需，不传 source_type（防回归）。"""
    mocker.patch("math_agent.nodes.modeler.RAG_ENABLED", True)
    mocker.patch("math_agent.nodes.modeler.RAG_DB_PATH", "/tmp/nonexistent.db")
    spy = mocker.patch("math_agent.nodes.modeler.search", return_value=[])
    mocker.patch(
        "math_agent.nodes.modeler.complete",
        return_value=ModelVersion(stage="basic", description="d"*200),
    )
    s = MathModelingState(problem="p", stage_target="basic")
    s.assumptions.append(Assumption(statement="a", rationale="r"))
    modeler_node(s)
    assert spy.call_args.kwargs.get("source_type") is None


def test_modeler_prompt_asks_figure_purposes_for_final_stage():
    """final 阶段的 modeler prompt 应要求 LLM 输出 figure_purposes（Plan D Phase 3）。"""
    from math_agent.prompts.modeler import build_prompt
    asum = [Assumption(statement="a", rationale="r")]
    prompt = build_prompt("problem", asum, None, "final")
    assert "figure_purposes" in prompt


def test_modeler_prompt_omits_figure_purposes_for_basic_stage():
    """basic 阶段不需要图，prompt 不应出现 figure_purposes 指令。"""
    from math_agent.prompts.modeler import build_prompt
    asum = [Assumption(statement="a", rationale="r")]
    prompt = build_prompt("problem", asum, None, "basic")
    assert "figure_purposes" not in prompt
