import json
import os
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from math_agent.cli import app
from math_agent.run_lock import RunLock
from math_agent.supervisor import SupervisorResult


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
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["thread"] == "default"
    assert manifest["no_interrupt"] is True


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


def test_run_force_cannot_delete_checkpoint_held_by_active_worker(tmp_path):
    problem = _problem(tmp_path)
    out = tmp_path / "run"
    out.mkdir()
    checkpoint = out / "checkpoints.sqlite"
    checkpoint.write_bytes(b"active")

    with RunLock(out):
        result = runner.invoke(app, [
            "run", "--problem", str(problem), "--out", str(out), "--force",
        ])

    assert result.exit_code == 75
    assert checkpoint.read_bytes() == b"active"


def test_supervise_delegates_to_process_supervisor(tmp_path):
    problem = _problem(tmp_path)
    out = tmp_path / "run"
    with patch(
        "math_agent.cli.run_process_supervisor",
        return_value=SupervisorResult(status="completed", attempts=2, recoveries=1),
    ) as supervised:
        result = runner.invoke(app, [
            "supervise", "--problem", str(problem), "--out", str(out),
            "--thread", "t1", "--no-interrupt",
        ])

    assert result.exit_code == 0, result.output
    kwargs = supervised.call_args.kwargs
    assert kwargs["thread"] == "t1"
    assert "--no-interrupt" in kwargs["run_args"]


def test_supervise_returns_nonzero_for_degraded_completion(tmp_path):
    problem = _problem(tmp_path)
    with patch(
        "math_agent.cli.run_process_supervisor",
        return_value=SupervisorResult(status="degraded"),
    ):
        result = runner.invoke(app, ["supervise", "--problem", str(problem)])
    assert result.exit_code == 2
    assert "DEGRADED" in result.output


def test_supervise_refuses_checkpoint_for_different_problem(tmp_path):
    first = _problem(tmp_path)
    second = tmp_path / "other.json"
    second.write_text(json.dumps({"title": "other", "questions": ["different"]}), encoding="utf-8")
    out = tmp_path / "run"
    out.mkdir()
    (out / "checkpoints.sqlite").write_bytes(b"checkpoint")
    from math_agent.cli import _problem_fingerprint, _read_problem_spec
    (out / "run_manifest.json").write_text(json.dumps({
        "thread": "default",
        "problem_sha256": _problem_fingerprint(_read_problem_spec(first)),
    }), encoding="utf-8")

    with patch("math_agent.cli.run_process_supervisor") as supervised:
        result = runner.invoke(app, [
            "supervise", "--problem", str(second), "--out", str(out),
        ])

    assert result.exit_code != 0
    assert "另一道题" in result.output
    supervised.assert_not_called()


def test_start_launches_detached_supervisor(tmp_path):
    problem = _problem(tmp_path)
    out = tmp_path / "run"
    with patch("math_agent.cli.start_detached_supervisor", return_value=4321) as detached:
        result = runner.invoke(app, [
            "start", "--problem", str(problem), "--out", str(out), "--no-interrupt",
        ])
    assert result.exit_code == 0
    assert "4321" in result.output
    assert "--no-interrupt" in detached.call_args.kwargs["supervise_args"]


def test_status_reads_persisted_supervisor_and_completion(tmp_path):
    (tmp_path / "supervisor.json").write_text(
        json.dumps({"status": "running", "heartbeat_at": "now", "worker_pid": 7}),
        encoding="utf-8",
    )
    (tmp_path / "completion.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8",
    )
    with patch("math_agent.cli.inspect_checkpoint") as inspect:
        inspect.return_value = MagicMock(
            checkpoint_exists=True, next_node="writer_section", final_status="",
        )
        result = runner.invoke(app, ["status", "--out", str(tmp_path)])
    assert result.exit_code == 0
    assert "writer_section" in result.output
    assert "running" in result.output
    assert "completed" in result.output


def test_status_marks_running_supervisor_with_dead_pid_as_stale(tmp_path):
    (tmp_path / "supervisor.json").write_text(
        json.dumps({
            "status": "running",
            "supervisor_pid": 99999999,
            "started_at": "2026-07-15T05:31:50+00:00",
            "heartbeat_at": "2026-07-15T05:35:56+00:00",
            "worker_pid": 99999998,
        }),
        encoding="utf-8",
    )
    with patch("math_agent.cli.inspect_checkpoint") as inspect:
        inspect.return_value = MagicMock(
            checkpoint_exists=True, next_node="coder_generate", final_status="pending",
        )
        result = runner.invoke(app, ["status", "--out", str(tmp_path)])

    assert result.exit_code == 0
    assert "supervisor_status: stale" in result.output
    assert "supervisor_pid_not_alive" in result.output


def test_status_marks_supervisor_superseded_by_verified_completion_as_stale(tmp_path):
    (tmp_path / "supervisor.json").write_text(
        json.dumps({"status": "degraded", "worker_pid": 99999998}),
        encoding="utf-8",
    )
    (tmp_path / "completion.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8",
    )
    with (
        patch("math_agent.cli.inspect_checkpoint") as inspect,
        patch("math_agent.cli.load_verified_completion") as verified,
    ):
        inspect.return_value = MagicMock(
            checkpoint_exists=True, next_node="", final_status="completed",
        )
        verified.return_value = MagicMock(status="completed")
        result = runner.invoke(app, ["status", "--out", str(tmp_path)])

    assert result.exit_code == 0
    assert "supervisor_status: stale" in result.output
    assert "superseded_by_verified_completion:completed" in result.output
    assert "effective_status: completed" in result.output


def test_resume_and_recover_require_checkpoint(tmp_path):
    resume_result = runner.invoke(app, ["resume", "--out", str(tmp_path), "--approve"])
    recover_result = runner.invoke(app, ["recover", "--out", str(tmp_path)])
    assert resume_result.exit_code == 1
    assert recover_result.exit_code == 1
    assert "no checkpoint" in resume_result.output
    assert "no checkpoint" in recover_result.output
    assert not (tmp_path / "trace.json").exists()


def test_no_interrupt_recover_does_not_reroute_checkpoint_to_inject_decision(tmp_path):
    (tmp_path / "checkpoints.sqlite").write_bytes(b"checkpoint")
    fake_graph = MagicMock()
    fake_graph.get_state.return_value = MagicMock(
        values={"human_decision": None},
        next=("sensitivity_code_generate",),
    )

    def invoke_without_checkpoint_mutation(*_args, **_kwargs):
        assert os.environ["MATH_AGENT_AUTO_APPROVE_HUMAN_REVIEW"] == "1"

    fake_graph.invoke.side_effect = invoke_without_checkpoint_mutation
    saver_cm = MagicMock()
    saver_cm.__enter__.return_value = object()
    saver_cm.__exit__.return_value = False
    with patch("math_agent.cli._require_trace_thread"), \
         patch("math_agent.cli._saver_cm", return_value=saver_cm), \
         patch("math_agent.cli.build_graph", return_value=fake_graph), \
         patch("math_agent.cli._dump_state_summary"):
        result = runner.invoke(app, [
            "recover", "--out", str(tmp_path), "--no-interrupt",
        ])

    assert result.exit_code == 0, result.output
    fake_graph.update_state.assert_not_called()
    assert "MATH_AGENT_AUTO_APPROVE_HUMAN_REVIEW" not in os.environ


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
