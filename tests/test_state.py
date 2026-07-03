from math_agent.state import (
    MathModelingState,
    Assumption,
    ModelVersion,
    DerivationStep,
    CriticReport,
    CriticIssue,
    PaperSections,
    SensitivityRun,
    FigureArtifact,
)


def test_initial_state_minimal():
    s = MathModelingState(problem="某共享单车调度优化问题")
    assert s.problem
    assert s.assumptions == []
    assert s.model_versions == []
    assert s.iteration == 0


def test_state_can_append_model_version():
    s = MathModelingState(problem="x")
    s.model_versions.append(
        ModelVersion(stage="basic", description="排队论 M/M/1", equations=["lambda < mu"])
    )
    assert s.latest_model().stage == "basic"


def test_state_can_record_critic():
    s = MathModelingState(problem="x")
    s.critic_reports.append(
        CriticReport(
            target="modeler",
            score=7,
            issues=[CriticIssue(section="general", problem="假设过强")],
            suggestions=["放宽到时变需求"],
            stage="basic",
        )
    )
    assert s.critic_reports[-1].score == 7
    assert s.critic_reports[-1].stage == "basic"


def test_latest_critic_for_stage_filters_by_stage():
    s = MathModelingState(problem="x")
    s.critic_reports.append(CriticReport(target="modeler", score=4, stage="basic"))
    s.critic_reports.append(CriticReport(target="modeler", score=9, approved=True, stage="basic"))
    s.critic_reports.append(CriticReport(target="modeler", score=5, stage="improved"))
    # basic 阶段的最新 critic 是 score=9 那条，不是 improved 的 5 分
    assert s.latest_critic_for_stage("modeler", "basic").score == 9
    assert s.latest_critic_for_stage("modeler", "improved").score == 5
    assert s.latest_critic_for_stage("modeler", "final") is None


def test_paper_sections_defaults_empty():
    p = PaperSections()
    assert p.abstract == ""
    assert p.conclusion == ""
    # Plan B 引入 sensitivity 章节
    assert p.sensitivity == ""


def test_state_has_sensitivity_runs():
    s = MathModelingState(problem="p")
    s.sensitivity_runs.append(
        SensitivityRun(
            parameter="lambda", values=[0.5, 1.0, 1.5],
            metric="avg_wait", results=[2.1, 3.5, 8.0],
            interpretation="敏感度高",
        )
    )
    assert s.sensitivity_runs[-1].parameter == "lambda"


def test_state_has_figures():
    s = MathModelingState(problem="p")
    s.figures.append(
        FigureArtifact(
            path="runs/x/fig1.png", purpose="对比", caption="见正文",
            quality_score=8, analysis="单调上升",
        )
    )
    assert s.figures[-1].quality_score == 8


def test_state_has_evaluation_default_none():
    s = MathModelingState(problem="p")
    assert s.evaluation is None


def test_state_has_human_decision_default_none():
    s = MathModelingState(problem="x")
    assert s.human_decision is None


def test_writer_iteration_defaults_to_zero():
    s = MathModelingState(problem="x")
    assert s.writer_iteration == 0


def test_model_version_has_figure_purposes_default_empty():
    from math_agent.state import ModelVersion
    m = ModelVersion(stage="final", description="d")
    assert m.figure_purposes == []
    m.figure_purposes.append("需求时序图")
    assert m.figure_purposes == ["需求时序图"]


def test_derivation_step_carries_step_metadata():
    from math_agent.state import DerivationStep
    d = DerivationStep(
        title="参数估计",
        motivation="为何用 MLE",
        statement="对数似然 \\ell(\\theta)=...",
        result="\\hat\\theta = ...",
    )
    assert d.title == "参数估计"
    assert d.motivation.startswith("为何")
    assert d.statement == "对数似然 \\ell(\\theta)=..."
    assert d.result == "\\hat\\theta = ..."
    assert d.result.startswith("\\hat")
    assert d.statement == "对数似然 \\ell(\\theta)=..."
    assert d.result == "\\hat\\theta = ..."


def test_critic_issue_defaults_section_to_general():
    ci = CriticIssue(problem="数字编造")
    assert ci.section == "general"
    assert ci.problem == "数字编造"


def test_critic_report_accepts_structured_issues():
    r = CriticReport(
        target="paper", score=4, approved=False,
        issues=[CriticIssue(section="solution", problem="46秒数字未在stdout出现")],
    )
    assert r.issues[0].section == "solution"
    assert r.issues[0].problem.startswith("46秒")


def test_state_has_table_warnings_field():
    """table_assembler writes cleaning warnings here; evaluation/latex can read them."""
    s = MathModelingState(problem="x")
    assert hasattr(s, "table_warnings")
    assert s.table_warnings == []


def test_code_artifact_has_category_field():
    """category 区分主方案(figure) vs 对照方案(baseline)。默认 'figure' 保持向后兼容。"""
    from math_agent.state import CodeArtifact
    a = CodeArtifact(purpose="test", code="print(1)")
    assert a.category == "figure"  # 默认值
    b = CodeArtifact(purpose="baseline", code="print(2)", category="baseline:no_schedule")
    assert b.category == "baseline:no_schedule"
