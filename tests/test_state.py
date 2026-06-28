from math_agent.state import (
    MathModelingState,
    Assumption,
    ModelVersion,
    CriticReport,
    PaperSections,
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
            issues=["假设过强"],
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
    basic = next((r for r in reversed(s.critic_reports) if r.target == "modeler" and r.stage == "basic"), None)
    improved = next((r for r in reversed(s.critic_reports) if r.target == "modeler" and r.stage == "improved"), None)
    final = next((r for r in reversed(s.critic_reports) if r.target == "modeler" and r.stage == "final"), None)
    assert basic.score == 9
    assert improved.score == 5
    assert final is None


def test_paper_sections_defaults_empty():
    p = PaperSections()
    assert p.abstract == ""
    assert p.conclusion == ""
    # sensitivity 字段在 MVP 已移除，避免 Plan B 引入后字段名漂移
    assert not hasattr(p, "sensitivity")
