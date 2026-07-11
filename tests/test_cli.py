import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from math_agent.cli import app


runner = CliRunner()


def _problem(tmp_path):
    path = tmp_path / "problem.json"
    path.write_text(json.dumps({"title": "t", "questions": ["q"]}), encoding="utf-8")
    return path


def test_run_help_exposes_only_meaningful_no_interrupt_flag():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--no-interrupt" in result.output
    assert "--no-no-interrupt" not in result.output


def test_run_force_removes_existing_checkpoint_files(tmp_path):
    problem = _problem(tmp_path)
    out = tmp_path / "run"
    out.mkdir()
    checkpoint = out / "checkpoints.sqlite"
    checkpoint.write_bytes(b"old")
    (out / "checkpoints.sqlite-wal").write_bytes(b"old")
    (out / "checkpoints.sqlite-shm").write_bytes(b"old")

    fake_graph = MagicMock()
    fake_graph.get_state.return_value = MagicMock(values={"problem": "p"})
    saver_cm = MagicMock()
    saver_cm.__enter__.return_value = object()
    saver_cm.__exit__.return_value = False
    with patch("math_agent.cli._saver_cm", return_value=saver_cm), \
         patch("math_agent.cli.build_graph", return_value=fake_graph), \
         patch("math_agent.cli._dump_state_summary"):
        result = runner.invoke(app, [
            "run", "--problem", str(problem), "--out", str(out),
            "--force", "--no-interrupt",
        ])

    assert result.exit_code == 0, result.output
    assert not checkpoint.exists()
    assert not (out / "checkpoints.sqlite-wal").exists()
    assert not (out / "checkpoints.sqlite-shm").exists()
    initial = fake_graph.invoke.call_args.args[0]
    assert initial["human_decision"].approved is True


def test_run_without_force_preserves_existing_checkpoint(tmp_path):
    problem = _problem(tmp_path)
    out = tmp_path / "run"
    out.mkdir()
    checkpoint = out / "checkpoints.sqlite"
    checkpoint.write_bytes(b"old")

    result = runner.invoke(app, ["run", "--problem", str(problem), "--out", str(out)])

    assert result.exit_code == 1
    assert checkpoint.read_bytes() == b"old"
    assert "already has a checkpoint" in result.output


def test_resume_and_recover_require_checkpoint(tmp_path):
    resume_result = runner.invoke(app, ["resume", "--out", str(tmp_path), "--approve"])
    recover_result = runner.invoke(app, ["recover", "--out", str(tmp_path)])
    assert resume_result.exit_code == 1
    assert recover_result.exit_code == 1
    assert "no checkpoint" in resume_result.output
    assert "no checkpoint" in recover_result.output
    assert not (tmp_path / "trace.json").exists()


def test_resume_requires_explicit_human_decision(tmp_path):
    result = runner.invoke(app, ["resume", "--out", str(tmp_path)])
    assert result.exit_code != 0
    assert "--approve" in result.output


def test_run_force_does_not_destroy_checkpoint_for_invalid_problem(tmp_path):
    problem = tmp_path / "invalid.json"
    problem.write_text("{not json", encoding="utf-8")
    out = tmp_path / "run"
    out.mkdir()
    checkpoint = out / "checkpoints.sqlite"
    checkpoint.write_bytes(b"old")

    result = runner.invoke(app, [
        "run", "--problem", str(problem), "--out", str(out), "--force",
    ])

    assert result.exit_code != 0
    assert checkpoint.read_bytes() == b"old"


def test_run_rejects_invalid_problem_schema_and_template(tmp_path):
    wrong_questions = tmp_path / "wrong.json"
    wrong_questions.write_text(
        json.dumps({"title": "t", "questions": "not-a-list"}), encoding="utf-8",
    )
    bad_schema = runner.invoke(app, ["run", "--problem", str(wrong_questions)])
    assert bad_schema.exit_code != 0
    assert "questions" in bad_schema.output

    valid = _problem(tmp_path)
    bad_template = runner.invoke(app, [
        "run", "--problem", str(valid), "--template", "gmcn",
    ])
    assert bad_template.exit_code != 0
    assert "default" in bad_template.output


def test_resume_wrong_thread_preserves_existing_trace(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "checkpoints.sqlite").write_bytes(b"checkpoint")
    trace = {"thread_id": "t1", "llm_calls": 3}
    trace_path = out / "trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")

    result = runner.invoke(app, [
        "resume", "--out", str(out), "--thread", "t2", "--approve",
    ])

    assert result.exit_code == 1
    assert "belongs to thread=t1" in result.output
    assert json.loads(trace_path.read_text(encoding="utf-8")) == trace


def test_run_passes_data_files_to_initial_state(tmp_path):
    problem = tmp_path / "problem.json"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "orders.xlsx").write_bytes(b"fake")
    problem.write_text(json.dumps({
        "title": "t",
        "questions": ["q"],
        "data_dir": str(data_dir),
        "data_files": [
            {"filename": "orders.xlsx", "file_type": "xlsx", "path": "orders.xlsx",
             "summary": {"sheets": [{"name": "Sheet1", "rows": 10, "cols": 3}]}}
        ],
    }), encoding="utf-8")
    out = tmp_path / "run"
    out.mkdir()

    fake_graph = MagicMock()
    fake_graph.get_state.return_value = MagicMock(values={"problem": "p"})
    saver_cm = MagicMock()
    saver_cm.__enter__.return_value = object()
    saver_cm.__exit__.return_value = False
    with patch("math_agent.cli._saver_cm", return_value=saver_cm), \
         patch("math_agent.cli.build_graph", return_value=fake_graph), \
         patch("math_agent.cli._dump_state_summary"):
        result = runner.invoke(app, [
            "run", "--problem", str(problem), "--out", str(out), "--no-interrupt",
        ])

    assert result.exit_code == 0, result.output
    initial = fake_graph.invoke.call_args.args[0]
    assert initial["data_dir"] == str(data_dir)
    assert len(initial["data_files"]) == 1
    assert initial["data_files"][0].filename == "orders.xlsx"


def test_run_rejects_nonexistent_data_dir(tmp_path):
    problem = tmp_path / "problem.json"
    problem.write_text(json.dumps({
        "title": "t", "questions": ["q"],
        "data_dir": str(tmp_path / "nonexistent"),
    }), encoding="utf-8")
    out = tmp_path / "run"
    out.mkdir()
    result = runner.invoke(app, ["run", "--problem", str(problem), "--out", str(out), "--no-interrupt"])
    assert result.exit_code != 0
    assert "data_dir" in result.output
