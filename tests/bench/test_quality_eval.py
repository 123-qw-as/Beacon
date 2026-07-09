"""P2 §8.3: quality_eval 测试。

重点断言：
- rubric 能加载。
- 评分结果包含四类分数。
- 高质量 state 通过门槛，低质量 state 不通过。
"""
from math_agent.bench.quality_eval import evaluate_quality, load_rubric, QualityScore
from math_agent.state import (
    MathModelingState, ProblemBlueprint, SubQuestionBlueprint, ObjectiveSpec,
    ConstraintSpec, MetricSpec, RecommendedRoute, ValidationPlanItem,
    ModelVersion, ModelQuestionCoverage, DerivationStep, CodeArtifact,
    EvaluationReport, PaperSections, CriticReport, ModelCodeConsistencyReport,
)


def test_rubric_loads():
    r = load_rubric("2022_A")
    assert r["problem_id"] == "2022_A"
    assert "blueprint" in r
    assert "model" in r
    assert "code" in r
    assert "paper" in r
    assert "thresholds" in r


def test_rubric_loads_all():
    for pid in ("2022_A", "2023_B"):
        r = load_rubric(pid)
        assert r["thresholds"]["min_overall"] == 7.2


def _passing_state():
    s = MathModelingState(problem="test")
    s.problem_blueprint = ProblemBlueprint(
        core_task="coverage",
        subquestions=[
            SubQuestionBlueprint(id="q1", original_text="x", task_type="optimization"),
            SubQuestionBlueprint(id="q2", original_text="y", task_type="prediction"),
        ],
        objectives=[ObjectiveSpec(direction="minimize", description="cost")],
        constraints=[ConstraintSpec(description="balance")],
        metrics=[MetricSpec(name="cost", meaning="total")],
        recommended_route=RecommendedRoute(route="MILP", reason="ok"),
        validation_plan=[ValidationPlanItem(target="acc", method="cv")],
    )
    s.model_versions.append(ModelVersion(
        stage="final", description="model with baseline",
        question_coverage=[ModelQuestionCoverage(question_id="q1", how_answered="eq1")],
        derivation_steps=[DerivationStep(title="s", motivation="m", statement="st")
                          for _ in range(5)],
    ))
    s.code_artifacts.append(CodeArtifact(
        purpose="main", code="print(1)",
        stdout="RESULT: baseline=ours cost=100", success=True, category="figure"))
    s.model_code_reports.append(ModelCodeConsistencyReport(score=9, approved=True))
    s.evaluation = EvaluationReport(
        assumption_reasonableness=8, modeling_creativity=8,
        result_correctness=8, writing_clarity=8, extra_depth=8, overall=8.0)
    s.paper = PaperSections(
        abstract="覆盖 无人机 鲁棒", model_section="覆盖",
        solution="无人机", conclusion="鲁棒")
    s.critic_reports.append(CriticReport(target="modeler", score=9, approved=True))
    s.critic_reports.append(CriticReport(target="paper", score=9, approved=True))
    return s


def test_quality_eval_returns_four_scores():
    q = evaluate_quality(_passing_state(), "2022_A")
    assert isinstance(q, QualityScore)
    assert hasattr(q, "blueprint")
    assert hasattr(q, "model")
    assert hasattr(q, "code")
    assert hasattr(q, "paper")
    assert hasattr(q, "overall")


def test_quality_eval_passes_with_good_state():
    q = evaluate_quality(_passing_state(), "2022_A")
    assert q.passed, f"expected pass, got failures: {q.failures}"
    assert q.blueprint >= 7
    assert q.model >= 7
    assert q.code >= 7
    assert q.paper >= 7
    assert q.overall >= 7.2


def test_quality_eval_fails_with_no_blueprint():
    s = _passing_state()
    s.problem_blueprint = None
    q = evaluate_quality(s, "2022_A")
    assert not q.passed
    assert q.blueprint == 0
    assert any("BlueprintScore" in f for f in q.failures)


def test_quality_eval_fails_with_no_result_lines():
    s = _passing_state()
    s.code_artifacts[0].stdout = "just text, no RESULT"
    q = evaluate_quality(s, "2022_A")
    assert not q.passed
    assert q.code < 7


def test_quality_eval_fails_with_consistency_not_approved():
    s = _passing_state()
    s.model_code_reports[-1] = ModelCodeConsistencyReport(score=3, approved=False)
    q = evaluate_quality(s, "2022_A")
    assert q.code < 7


def test_quality_eval_fails_with_missing_keywords():
    s = _passing_state()
    s.paper = PaperSections(abstract="nothing relevant here")
    q = evaluate_quality(s, "2022_A")
    assert not q.passed
    assert q.paper < 7


def test_quality_eval_question_coverage():
    s = _passing_state()
    q = evaluate_quality(s, "2022_A")
    # 2 subquestions, 1 covered
    assert q.question_coverage == "1/2"


def test_quality_eval_overall_is_arithmetic_mean():
    s = _passing_state()
    q = evaluate_quality(s, "2022_A")
    expected = (q.blueprint + q.model + q.code + q.paper) / 4.0
    assert abs(q.overall - round(expected, 2)) < 0.01


def test_quality_eval_failing_state_all_below_threshold():
    s = MathModelingState(problem="test")
    s.model_versions.append(ModelVersion(stage="final", description="d"))
    s.code_artifacts.append(CodeArtifact(
        purpose="main", code="print(1)", stdout="done", success=True))
    s.evaluation = EvaluationReport(
        assumption_reasonableness=3, modeling_creativity=3,
        result_correctness=3, writing_clarity=3, extra_depth=3, overall=3.0)
    s.paper = PaperSections(abstract="nothing")
    q = evaluate_quality(s, "2022_A")
    assert not q.passed
    assert q.overall < 7.2
    assert len(q.failures) >= 4  # all four dimensions below threshold
