"""ProblemBlueprint schema 测试：完整 JSON 解析 + 缺失可选字段的默认值。"""
import json

from math_agent.state import (
    ProblemBlueprint, SubQuestionBlueprint, DecisionVariable, ObjectiveSpec,
    ConstraintSpec, MetricSpec, DataRequirement, Assumption, ModelingCandidate,
    RecommendedRoute, ValidationPlanItem, ModelQuestionCoverage, ModelVersion,
    CodeArtifact, ModelCodeConsistencyReport,
)


def test_blueprint_parses_full_json():
    raw = {
        "core_task": "共享单车调度优化",
        "subquestions": [
            {"id": "q1", "original_text": "预测需求", "task_type": "prediction",
             "depends_on": [], "expected_output": "需求预测值", "success_criteria": ["RMSE < 0.1"]},
            {"id": "q2", "original_text": "调度方案", "task_type": "optimization",
             "depends_on": ["q1"], "expected_output": "调度路径", "success_criteria": ["成本最低"]},
        ],
        "decision_variables": [
            {"name": "x_i", "meaning": "站点i调度量", "domain": ">=0", "unit": "辆"},
        ],
        "objectives": [
            {"direction": "minimize", "description": "总调度成本", "math_hint": "sum(c_i * x_i)"},
        ],
        "constraints": [
            {"description": "供需平衡", "math_hint": "sum(x_i) = D", "source": "given"},
        ],
        "metrics": [
            {"name": "total_cost", "meaning": "总成本", "direction": "lower_better", "unit": "元"},
        ],
        "data_requirements": [
            {"field": "demand", "meaning": "历史需求数据", "needed_for": "预测",
             "given_or_missing": "given", "handling_strategy": ""},
        ],
        "assumptions": [
            {"statement": "需求服从泊松分布", "rationale": "短时间高频独立到达"},
        ],
        "problem_domains": ["optimization", "queueing"],
        "modeling_candidates": [
            {"name": "MILP", "route": "混合整数规划", "suitable_for": ["q2"],
             "data_required": ["demand"], "pros": ["精确"], "cons": ["计算量大"],
             "risk": "NP-hard", "recommendation_score": 8},
        ],
        "recommended_route": {"route": "MILP+预测", "reason": "精确且可解释",
                              "baseline": "贪心", "fallback": "启发式"},
        "validation_plan": [
            {"target": "预测精度", "method": "交叉验证", "pass_criteria": "RMSE < 0.1"},
        ],
        "risks": ["数据缺失", "计算超时"],
    }
    bp = ProblemBlueprint.model_validate(raw)
    assert bp.core_task == "共享单车调度优化"
    assert len(bp.subquestions) == 2
    assert bp.subquestions[0].task_type == "prediction"
    assert bp.subquestions[1].depends_on == ["q1"]
    assert bp.decision_variables[0].name == "x_i"
    assert bp.objectives[0].direction == "minimize"
    assert bp.constraints[0].source == "given"
    assert bp.metrics[0].direction == "lower_better"
    assert bp.data_requirements[0].given_or_missing == "given"
    assert len(bp.assumptions) == 1
    assert bp.problem_domains == ["optimization", "queueing"]
    assert bp.modeling_candidates[0].recommendation_score == 8
    assert bp.recommended_route.route == "MILP+预测"
    assert bp.validation_plan[0].target == "预测精度"
    assert bp.risks == ["数据缺失", "计算超时"]


def test_blueprint_defaults_for_optional_fields():
    bp = ProblemBlueprint(core_task="minimal")
    assert bp.subquestions == []
    assert bp.decision_variables == []
    assert bp.objectives == []
    assert bp.constraints == []
    assert bp.metrics == []
    assert bp.data_requirements == []
    assert bp.assumptions == []
    assert bp.problem_domains == []
    assert bp.modeling_candidates == []
    assert bp.recommended_route is None
    assert bp.validation_plan == []
    assert bp.risks == []


def test_blueprint_round_trip_json():
    bp = ProblemBlueprint(
        core_task="test",
        subquestions=[SubQuestionBlueprint(id="q1", original_text="x", task_type="optimization")],
    )
    j = bp.model_dump_json()
    bp2 = ProblemBlueprint.model_validate_json(j)
    assert bp2.core_task == "test"
    assert bp2.subquestions[0].id == "q1"


def test_state_problem_blueprint_overwrite_semantics():
    """problem_blueprint 是覆盖语义，不是追加。"""
    from math_agent.state import MathModelingState
    s = MathModelingState(problem="p")
    s.problem_blueprint = ProblemBlueprint(core_task="v1")
    s.problem_blueprint = ProblemBlueprint(core_task="v2")
    assert s.problem_blueprint.core_task == "v2"


def test_state_blueprint_iteration_default():
    from math_agent.state import MathModelingState
    s = MathModelingState(problem="p")
    assert s.blueprint_iteration == 0
    assert s.code_verify_iteration == 0


def test_model_version_coverage_fields_default():
    m = ModelVersion(stage="final", description="d")
    assert m.question_coverage == []
    assert m.objective_mapping == []
    assert m.constraint_mapping == []
    assert m.validation_mapping == []


def test_code_artifact_batch_default():
    a = CodeArtifact(purpose="p", code="c")
    assert a.batch == 0


def test_model_code_consistency_report_defaults():
    r = ModelCodeConsistencyReport(score=0, approved=False)
    assert r.implemented_variables == []
    assert r.missing_variables == []
    assert r.implemented_objectives == []
    assert r.missing_objectives == []
    assert r.implemented_constraints == []
    assert r.missing_constraints == []
    assert r.output_metric_alignment == []
    assert r.issues == []
    assert r.suggestions == []
