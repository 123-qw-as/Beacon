"""BlueprintCritic 节点测试。"""
from math_agent.state import (
    MathModelingState, ProblemBlueprint, CriticReport, CriticIssue, Assumption,
)
from math_agent.nodes.blueprint_critic import blueprint_critic_node


def _blueprint():
    return ProblemBlueprint(
        core_task="调度优化",
        subquestions=[],
        assumptions=[Assumption(statement="A", rationale="r")],
        problem_domains=["optimization"],
    )


def test_blueprint_critic_generates_analyst_target_report(mocker):
    fake = CriticReport(target="analyst", score=9, approved=True, critic_type="blueprint")
    mocker.patch("math_agent.nodes.blueprint_critic.complete", return_value=fake)

    s = MathModelingState(problem="p", blueprint_iteration=0)
    s.problem_blueprint = _blueprint()
    delta = blueprint_critic_node(s)

    assert "critic_reports" in delta
    report = delta["critic_reports"][0]
    assert report.target == "analyst"
    assert report.approved is True
    # blueprint_iteration 应递增
    assert delta["blueprint_iteration"] == 1


def test_blueprint_critic_increments_iteration(mocker):
    mocker.patch("math_agent.nodes.blueprint_critic.complete",
                 return_value=CriticReport(target="analyst", score=4, approved=False, critic_type="blueprint"))

    s = MathModelingState(problem="p", blueprint_iteration=1)
    s.problem_blueprint = _blueprint()
    delta = blueprint_critic_node(s)
    assert delta["blueprint_iteration"] == 2


def test_blueprint_critic_handles_missing_blueprint(mocker):
    """没有 blueprint 时应返回未通过报告并递增 iteration。"""
    spy = mocker.patch("math_agent.nodes.blueprint_critic.complete")
    s = MathModelingState(problem="p", blueprint_iteration=0)
    s.problem_blueprint = None
    delta = blueprint_critic_node(s)

    # 不应调用 LLM
    spy.assert_not_called()
    report = delta["critic_reports"][0]
    assert report.target == "analyst"
    assert report.approved is False
    assert delta["blueprint_iteration"] == 1


def test_blueprint_critic_injects_blueprint_into_prompt(mocker):
    spy = mocker.patch("math_agent.nodes.blueprint_critic.complete",
                       return_value=CriticReport(target="analyst", score=9, approved=True, critic_type="blueprint"))
    s = MathModelingState(problem="调度问题", questions=["预测", "调度"])
    s.problem_blueprint = _blueprint()
    blueprint_critic_node(s)

    prompt_arg = spy.call_args.args[0]
    # prompt 应包含 blueprint JSON 和题目小问
    assert "调度优化" in prompt_arg
    assert "预测" in prompt_arg
    assert "调度" in prompt_arg
