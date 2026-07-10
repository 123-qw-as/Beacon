from math_agent.state import MathModelingState, Assumption, ProblemBlueprint
from math_agent.nodes.analyst import analyst_node, AnalystOutput


def _blueprint(**overrides):
    """构造测试用 ProblemBlueprint，默认带 2 条假设。"""
    defaults = dict(
        core_task="共享单车调度",
        assumptions=[
            Assumption(statement="需求服从泊松", rationale="短时间高频独立到达"),
            Assumption(statement="调度车恒速", rationale="市区均速近似"),
        ],
        problem_domains=["optimization", "queueing"],
    )
    defaults.update(overrides)
    return ProblemBlueprint(**defaults)


def test_analyst_returns_problem_blueprint(mocker):
    fake = _blueprint()
    mocker.patch("math_agent.nodes.analyst.complete", return_value=fake)

    state = MathModelingState(problem="共享单车调度", questions=["预测", "调度"])
    delta = analyst_node(state)

    assert delta["problem_blueprint"].core_task == "共享单车调度"
    # 首次（blueprint_iteration==0）同步 assumptions
    assert "assumptions" in delta
    assert len(delta["assumptions"]) == 2
    assert delta["assumptions"][0].statement == "需求服从泊松"
    # problem_domains 覆盖同步
    assert delta["problem_domains"] == ["optimization", "queueing"]


def test_analyst_does_not_query_rag_when_disabled(mocker):
    mocker.patch("math_agent.nodes.analyst.RAG_ENABLED", False)
    spy = mocker.patch("math_agent.nodes.analyst.search")
    mocker.patch(
        "math_agent.nodes.analyst.complete",
        return_value=_blueprint(assumptions=[]),
    )
    analyst_node(MathModelingState(problem="p"))
    spy.assert_not_called()


def test_analyst_queries_rag_when_enabled(mocker):
    mocker.patch("math_agent.nodes.analyst.RAG_ENABLED", True)
    mocker.patch("math_agent.nodes.analyst.RAG_DB_PATH", "/tmp/nonexistent.db")
    spy = mocker.patch("math_agent.nodes.analyst.search", return_value=[])
    mocker.patch(
        "math_agent.nodes.analyst.complete",
        return_value=_blueprint(assumptions=[]),
    )
    analyst_node(MathModelingState(problem="p"))
    spy.assert_called_once()


def test_analyst_outputs_problem_domains(mocker):
    mocker.patch("math_agent.nodes.analyst.complete",
                 return_value=_blueprint(
                     problem_domains=["optimization", "queueing"],
                 ))
    s = MathModelingState(problem="p")
    delta = analyst_node(s)
    assert delta["problem_domains"] == ["optimization", "queueing"]


def test_analyst_skips_assumptions_on_retry(mocker):
    """blueprint_iteration > 0 时不追加 assumptions，避免重复。"""
    mocker.patch("math_agent.nodes.analyst.complete",
                 return_value=_blueprint())
    s = MathModelingState(problem="p", blueprint_iteration=1)
    delta = analyst_node(s)
    # retry 时不应同步 assumptions（避免追加语义导致重复）
    assert "assumptions" not in delta
    # 但 problem_blueprint 和 problem_domains 仍应覆盖
    assert "problem_blueprint" in delta
    assert "problem_domains" in delta


def test_analyst_injects_critic_feedback(mocker):
    """blueprint_critic 未通过时，issues/suggestions 应注入 prompt。"""
    from math_agent.state import CriticReport, CriticIssue
    spy = mocker.patch("math_agent.nodes.analyst.complete", return_value=_blueprint())
    s = MathModelingState(problem="p")
    s.critic_reports.append(CriticReport(
        target="analyst", score=4, approved=False, critic_type="blueprint",
        issues=[CriticIssue(problem="遗漏小问2")], suggestions=["补充小问2"],
    ))
    analyst_node(s)
    prompt_arg = spy.call_args.args[0]
    assert "遗漏小问2" in prompt_arg
    assert "补充小问2" in prompt_arg
