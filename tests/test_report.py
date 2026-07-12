"""P2 §8.3: CLI report 测试。

重点断言：
- report 能展示 blueprint 和一致性摘要。
- trace.json 缺失时报错退出。
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from math_agent.cli import app, _print_blueprint_summary, _read_state_summary_data
from math_agent.state import (
    MathModelingState, ProblemBlueprint, SubQuestionBlueprint,
    ModelVersion, ModelQuestionCoverage, CodeArtifact,
    CriticReport, CriticIssue, ModelCodeConsistencyReport, EvaluationReport,
)


runner = CliRunner()


def test_report_exits_when_no_trace(tmp_path):
    result = runner.invoke(app, ["report", "--out", str(tmp_path)])
    assert result.exit_code == 1
    assert "no trace" in result.output


def test_report_exits_cleanly_for_invalid_trace(tmp_path):
    (tmp_path / "trace.json").write_text("[]", encoding="utf-8")
    result = runner.invoke(app, ["report", "--out", str(tmp_path)])
    assert result.exit_code == 1
    assert "invalid trace" in result.output


def test_report_prints_trace_tables(tmp_path):
    trace = {
        "thread_id": "t1",
        "llm_calls": 5,
        "tokens": {"prompt": 100, "completion": 200},
        "per_model": {"gpt-4o": {"calls": 5, "prompt_tokens": 100, "completion_tokens": 200}},
        "nodes": [{"name": "analyst", "duration_ms": 500}],
    }
    (tmp_path / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
    result = runner.invoke(app, ["report", "--out", str(tmp_path)])
    assert result.exit_code == 0
    assert "Run report" in result.output
    assert "LLM calls" in result.output
    assert "analyst" in result.output


def test_blueprint_summary_extracts_scores(tmp_path):
    """_print_blueprint_summary 从 checkpoint state 提取 blueprint/一致性摘要。"""
    from rich.console import Console

    state = MathModelingState(problem="test")
    state.problem_blueprint = ProblemBlueprint(
        core_task="test",
        subquestions=[
            SubQuestionBlueprint(id="q1", original_text="x", task_type="optimization"),
            SubQuestionBlueprint(id="q2", original_text="y", task_type="prediction"),
        ],
    )
    state.model_versions.append(ModelVersion(
        stage="final", description="d",
        question_coverage=[ModelQuestionCoverage(question_id="q1", how_answered="eq1")],
    ))
    state.critic_reports.append(CriticReport(
        target="analyst", score=8, approved=True, critic_type="blueprint"))
    state.model_code_reports.append(ModelCodeConsistencyReport(score=7, approved=True))

    # mock: build_graph + get_state 返回构造的 state
    fake_snap = MagicMock()
    fake_snap.values = state
    fake_graph = MagicMock()
    fake_graph.get_state.return_value = fake_snap

    # checkpoints.sqlite 需存在，否则 _print_blueprint_summary 会跳过
    (tmp_path / "checkpoints.sqlite").touch()

    with patch("math_agent.cli._saver_cm") as mock_saver, \
         patch("math_agent.cli.build_graph", return_value=fake_graph):
        mock_saver.return_value.__enter__ = MagicMock()
        mock_saver.return_value.__exit__ = MagicMock(return_value=False)

        # 用 string capture 的 Console
        import io
        buf = io.StringIO()
        c = Console(file=buf, width=120)
        _print_blueprint_summary(c, tmp_path)

    output = buf.getvalue()
    assert "Blueprint & Consistency" in output
    assert "Blueprint Score" in output
    assert "8/10" in output
    assert "Model-Code Score" in output
    assert "7/10" in output
    assert "Question Coverage" in output
    assert "1/2" in output


def test_blueprint_summary_counts_unresolved_issues(tmp_path):
    """未通过的 critic/consistency 报告的 issue 数应计入 Unresolved Issues。"""
    from rich.console import Console
    import io

    state = MathModelingState(problem="test")
    state.critic_reports.append(CriticReport(
        target="analyst", score=4, approved=False, critic_type="blueprint",
        issues=[CriticIssue(problem="遗漏小问"), CriticIssue(problem="变量空泛")]))
    state.model_code_reports.append(ModelCodeConsistencyReport(
        score=3, approved=False, issues=["missing variable x"]))

    fake_snap = MagicMock()
    fake_snap.values = state
    fake_graph = MagicMock()
    fake_graph.get_state.return_value = fake_snap

    (tmp_path / "checkpoints.sqlite").touch()

    with patch("math_agent.cli._saver_cm") as mock_saver, \
         patch("math_agent.cli.build_graph", return_value=fake_graph):
        mock_saver.return_value.__enter__ = MagicMock()
        mock_saver.return_value.__exit__ = MagicMock(return_value=False)

        buf = io.StringIO()
        c = Console(file=buf, width=120)
        _print_blueprint_summary(c, tmp_path)

    output = buf.getvalue()
    assert "Unresolved Issues" in output
    # 2 critic issues + 1 consistency issue = 3
    assert "3" in output


def test_blueprint_summary_skips_when_no_checkpoint(tmp_path):
    """没有 checkpoints.sqlite 时静默跳过，不报错。"""
    from rich.console import Console
    import io

    buf = io.StringIO()
    c = Console(file=buf, width=120)
    _print_blueprint_summary(c, tmp_path)
    assert buf.getvalue() == ""


def test_state_summary_supports_fully_serialized_checkpoint(tmp_path):
    state = MathModelingState(problem="test")
    state.problem_blueprint = ProblemBlueprint(
        core_task="task",
        subquestions=[SubQuestionBlueprint(
            id="q1", original_text="x", task_type="optimization",
        )],
    )
    state.model_versions.append(ModelVersion(
        stage="final", description="d",
        question_coverage=[ModelQuestionCoverage(question_id="q1", how_answered="eq")],
    ))
    state.critic_reports.append(CriticReport(
        target="analyst", score=8, approved=True, critic_type="blueprint",
    ))
    state.model_code_reports.append(ModelCodeConsistencyReport(score=7, approved=True))
    state.evaluation = EvaluationReport(
        assumption_reasonableness=8, modeling_creativity=8,
        result_correctness=8, writing_clarity=8, extra_depth=8, overall=8.0,
    )

    fake_snap = MagicMock(values=state.model_dump())
    fake_graph = MagicMock()
    fake_graph.get_state.return_value = fake_snap
    (tmp_path / "checkpoints.sqlite").touch()
    with patch("math_agent.cli._saver_cm") as mock_saver, \
         patch("math_agent.cli.build_graph", return_value=fake_graph):
        mock_saver.return_value.__enter__ = MagicMock()
        mock_saver.return_value.__exit__ = MagicMock(return_value=False)
        data = _read_state_summary_data(tmp_path)

    assert data["total_sq"] == 1
    assert data["covered"] == 1
    assert data["bp_critic"]["score"] == 8
    assert data["evaluation_overall"] == 8.0
