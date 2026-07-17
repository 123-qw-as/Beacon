import json

import fitz
import pytest
from pypdf import PdfWriter

from math_agent.errors import FinalizationError
from math_agent.nodes.finalizer import finalizer_node, load_verified_completion
from math_agent.state import (
    CodeArtifact,
    CriticReport,
    EvaluationReport,
    HumanDecision,
    MathModelingState,
    ModelCodeConsistencyReport,
    ModelVersion,
    PaperSections,
    SensitivityRun,
)


def _ready_state(workdir) -> MathModelingState:
    state = MathModelingState(
        problem="测试题",
        output_dir=str(workdir),
        human_decision=HumanDecision(approved=True),
    )
    state.model_versions.append(ModelVersion(stage="final", description="最终模型"))
    state.paper = PaperSections(
        abstract="摘要" * 50,
        problem_restatement="问题重述" * 30,
        assumptions="假设" * 30,
        notation="符号" * 30,
        model_section="模型" * 100,
        solution="求解" * 100,
        sensitivity="敏感性" * 30,
        conclusion="结论" * 30,
        references="[1] reference",
    )
    state.evaluation = EvaluationReport(
        assumption_reasonableness=8,
        modeling_creativity=8,
        result_correctness=8,
        writing_clarity=8,
        extra_depth=8,
        overall=8,
    )
    state.code_artifacts = [
        CodeArtifact(
            purpose="主方案", code="", success=True,
            stdout="RESULT: baseline=ours total_cost=80 service_rate=0.95 vehicles=7",
        ),
        CodeArtifact(
            purpose="基线1", code="", success=True, category="baseline:no_schedule",
            evidence_role="baseline",
            stdout="RESULT: baseline=no_schedule total_cost=120 service_rate=0.80 vehicles=8",
        ),
        CodeArtifact(
            purpose="基线2", code="", success=True, category="baseline:greedy",
            evidence_role="baseline",
            stdout="RESULT: baseline=greedy total_cost=100 service_rate=0.90 vehicles=7",
        ),
    ]
    state.sensitivity_runs = [SensitivityRun(
        parameter="beta", values=[0.8, 1.0, 1.2], metric="total_cost",
        results=[75, 80, 86],
    )]
    state.model_code_reports = [ModelCodeConsistencyReport(score=8, approved=True)]
    state.critic_reports = [CriticReport(
        target="paper", score=8, approved=True, issues=[], suggestions=[],
    )]
    (workdir / "paper.md").write_text("# paper\n", encoding="utf-8")
    (workdir / "paper.tex").write_text("\\documentclass{article}", encoding="utf-8")
    pdf = fitz.open()
    for page_number in range(20):
        page = pdf.new_page(width=595, height=842)
        for line_number in range(40):
            page.insert_text(
                (50, 50 + line_number * 18),
                f"competition paper body evidence page {page_number + 1} line {line_number + 1} "
                "model data result validation",
                fontsize=8,
            )
    pdf.save(workdir / "paper.pdf")
    pdf.close()
    (workdir / "compile.log").write_text("Output written on paper.pdf", encoding="utf-8")
    return state


def test_finalizer_atomically_commits_valid_run(workdir):
    state = _ready_state(workdir)

    delta = finalizer_node(state)

    report = delta["finalization"]
    assert report.status == "completed"
    assert report.issues == []
    assert {"paper.md", "paper.tex", "paper.pdf"} <= set(report.artifacts)
    completion = json.loads((workdir / "completion.json").read_text(encoding="utf-8"))
    final_state = json.loads((workdir / "final_state.json").read_text(encoding="utf-8"))
    assert completion["status"] == "completed"
    assert final_state["finalization"]["status"] == "completed"
    assert not list(workdir.glob("*.tmp-*"))
    assert load_verified_completion(workdir).status == "completed"


def test_verified_completion_rejects_tampered_artifact(workdir):
    state = _ready_state(workdir)
    finalizer_node(state)
    (workdir / "paper.md").write_text("tampered", encoding="utf-8")
    assert load_verified_completion(workdir) is None


def test_finalizer_marks_existing_nonfatal_errors_as_degraded(workdir):
    state = _ready_state(workdir)
    state.errors.append("rag: embedding unavailable")

    delta = finalizer_node(state)

    assert delta["finalization"].status == "degraded"
    assert "rag: embedding unavailable" in delta["finalization"].warnings


def test_finalizer_ignores_recovered_historical_coder_error(workdir):
    state = _ready_state(workdir)
    state.errors.append("coder: all code tasks failed")

    delta = finalizer_node(state)

    assert delta["finalization"].status == "completed"
    assert "coder: all code tasks failed" not in delta["finalization"].warnings


def test_finalizer_ignores_historical_latex_error_after_verified_recompile(workdir):
    state = _ready_state(workdir)
    state.errors.append("latex compile failed: old xelatex failure")
    (workdir / "compile.log").write_text(
        "[pass 1] exit=0\n[pass 2] exit=0\nOutput written on paper.pdf",
        encoding="utf-8",
    )

    report = finalizer_node(state)["finalization"]

    assert report.status == "completed"
    assert not any("latex compile failed" in item for item in report.warnings)


def test_finalizer_rejects_unfinished_pending_state_and_keeps_recovery_point(workdir):
    state = _ready_state(workdir)
    state.sensitivity_phase = "interpret"
    state.sensitivity_pending_code = "print(1)"

    with pytest.raises(FinalizationError, match="sensitivity_phase"):
        finalizer_node(state)

    completion = json.loads((workdir / "completion.json").read_text(encoding="utf-8"))
    assert completion["status"] == "failed"
    assert any("sensitivity_phase" in item for item in completion["issues"])
    assert not (workdir / "final_state.json").exists()


def test_finalizer_requires_current_core_artifacts(workdir):
    state = _ready_state(workdir)
    (workdir / "paper.tex").unlink()

    with pytest.raises(FinalizationError, match="paper.tex"):
        finalizer_node(state)


def test_finalizer_surfaces_all_quality_gate_failures(workdir):
    state = _ready_state(workdir)
    state.code_artifacts = [CodeArtifact(
        purpose="伪成功", code="", success=True,
        stdout="Error during execution: '纬度'\nRESULT: baseline=ours total_cost=0 service_rate=0",
    )]
    state.sensitivity_runs = []
    state.model_code_reports = [ModelCodeConsistencyReport(score=5, approved=False)]
    state.critic_reports = [CriticReport(target="paper", score=5, approved=False)]
    state.evaluation = EvaluationReport(
        assumption_reasonableness=5, modeling_creativity=4,
        result_correctness=2, writing_clarity=5, extra_depth=2, overall=3.6,
    )

    report = finalizer_node(state)["finalization"]
    joined = "\n".join(report.warnings)

    assert report.status == "degraded"
    assert "主方案 RESULT" in joined
    assert "有效对照方案" in joined
    assert "敏感性分析" in joined
    assert "模型—代码一致性未通过" in joined
    assert "论文评审未通过" in joined
    assert "综合评分" in joined
    assert "结果正确性=2" in joined


def test_finalizer_marks_severe_latex_overflow_degraded(workdir):
    state = _ready_state(workdir)
    (workdir / "compile.log").write_text(
        "Overfull \\hbox (279.5pt too wide) in paragraph at lines 1--2",
        encoding="utf-8",
    )

    report = finalizer_node(state)["finalization"]

    assert report.status == "degraded"
    assert any("279.50pt" in warning for warning in report.warnings)


def test_finalizer_marks_short_competition_paper_body_degraded(workdir):
    state = _ready_state(workdir)
    writer = PdfWriter()
    for _ in range(8):
        writer.add_blank_page(width=595, height=842)
    with (workdir / "paper.pdf").open("wb") as handle:
        writer.write(handle)

    report = finalizer_node(state)["finalization"]

    assert report.status == "degraded"
    assert any("正文页数 8" in warning and "至少需要 20 页" in warning
               for warning in report.warnings)


def test_finalizer_rejects_stale_sensitivity_history_in_paper(workdir):
    state = _ready_state(workdir)
    stale_figure = workdir / "stale.png"
    current_figure = workdir / "current.png"
    state.sensitivity_runs = [
        SensitivityRun(
            parameter="speed", values=[0.8, 1.0, 1.2], metric="cost",
            results=[238754.47, 240000.0, 251132.99],
            interpretation="旧轮次声称速度越高目标函数越大，这是失效数值解释。",
            figure_path=str(stale_figure),
        ),
        SensitivityRun(
            parameter="speed", values=[0.8, 1.0, 1.2], metric="cost",
            results=[146017.04, 144586.99, 145204.85],
            interpretation="当前正式解释。", figure_path=str(current_figure),
        ),
    ]
    (workdir / "paper.md").write_text(
        "旧轮次声称速度越高目标函数越大，这是失效数值解释。\n"
        + str(stale_figure),
        encoding="utf-8",
    )

    report = finalizer_node(state)["finalization"]

    assert report.status == "degraded"
    assert any("历史敏感性解释" in warning for warning in report.warnings)
    assert any("历史敏感性图" in warning for warning in report.warnings)
