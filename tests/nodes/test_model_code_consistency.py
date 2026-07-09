"""ModelCodeConsistency 节点测试。"""
from math_agent.state import (
    MathModelingState, ModelVersion, CodeArtifact, ProblemBlueprint,
    ModelCodeConsistencyReport, MetricSpec,
)
from math_agent.nodes.model_code_consistency import model_code_consistency_node


def _state_with_model_and_code(*, main_success=True, batch=1):
    s = MathModelingState(problem="p")
    s.problem_blueprint = ProblemBlueprint(
        core_task="test",
        metrics=[MetricSpec(name="total_cost", meaning="成本", direction="lower_better")],
    )
    s.model_versions.append(ModelVersion(
        stage="final", description="d" * 200, equations=["x=1"],
        variables={"x": "决策变量", "total_cost": "总成本"},
    ))
    s.code_artifacts.append(CodeArtifact(
        purpose="主方案", code="x=1\nprint('RESULT: baseline=ours total_cost=100')",
        stdout="RESULT: baseline=ours total_cost=100",
        success=main_success, category="figure", batch=batch,
    ))
    s.code_artifacts.append(CodeArtifact(
        purpose="贪心对照", code="print('RESULT: baseline=greedy total_cost=200')",
        stdout="RESULT: baseline=greedy total_cost=200",
        success=True, category="baseline:greedy", batch=batch,
    ))
    return s


def test_consistency_fails_without_final_model(mocker):
    mocker.patch("math_agent.nodes.model_code_consistency.complete",
                 return_value=ModelCodeConsistencyReport(score=9, approved=True))
    s = MathModelingState(problem="p")
    s.problem_blueprint = ProblemBlueprint(core_task="test")
    # 没有 model_versions
    delta = model_code_consistency_node(s)
    report = delta["model_code_reports"][0]
    assert report.approved is False
    assert delta["code_verify_iteration"] == 1


def test_consistency_fails_without_successful_main_code(mocker):
    """没有成功主方案代码时直接未通过。"""
    spy = mocker.patch("math_agent.nodes.model_code_consistency.complete")
    s = _state_with_model_and_code(main_success=False)
    delta = model_code_consistency_node(s)
    # 不应调用 LLM
    spy.assert_not_called()
    report = delta["model_code_reports"][0]
    assert report.approved is False
    assert "没有成功的主方案代码" in report.issues[0]
    # missing_variables 应包含模型变量
    assert "x" in report.missing_variables


def test_consistency_approves_when_aligned(mocker):
    fake = ModelCodeConsistencyReport(
        score=9, approved=True,
        implemented_variables=["x", "total_cost"],
        implemented_objectives=["minimize cost"],
        implemented_constraints=["supply=demand"],
        output_metric_alignment=["total_cost"],
    )
    mocker.patch("math_agent.nodes.model_code_consistency.complete", return_value=fake)
    s = _state_with_model_and_code()
    delta = model_code_consistency_node(s)
    report = delta["model_code_reports"][0]
    assert report.approved is True
    assert report.score == 9
    assert "x" in report.implemented_variables
    assert delta["code_verify_iteration"] == 1


def test_consistency_only_checks_latest_batch(mocker):
    """一致性审查只看最新 batch 的 artifact，不看旧 batch。"""
    spy = mocker.patch("math_agent.nodes.model_code_consistency.complete",
                       return_value=ModelCodeConsistencyReport(score=9, approved=True))
    s = _state_with_model_and_code(batch=2)
    # 添加一批旧的 artifact（batch=1）
    s.code_artifacts.insert(0, CodeArtifact(
        purpose="旧主方案", code="old code",
        stdout="RESULT: baseline=ours old=999",
        success=True, category="figure", batch=1,
    ))
    model_code_consistency_node(s)
    prompt_arg = spy.call_args.args[0]
    # 最新 batch 的代码应在 prompt 中
    assert "主方案" in prompt_arg
    # 旧 batch 的代码不应在 prompt 中
    assert "旧主方案" not in prompt_arg
    assert "old code" not in prompt_arg


def test_consistency_increments_iteration(mocker):
    mocker.patch("math_agent.nodes.model_code_consistency.complete",
                 return_value=ModelCodeConsistencyReport(score=9, approved=True))
    s = _state_with_model_and_code()
    s.code_verify_iteration = 1
    delta = model_code_consistency_node(s)
    assert delta["code_verify_iteration"] == 2
