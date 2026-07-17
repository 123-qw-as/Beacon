from math_agent.errors import LLMValidationError
import sys
import time

from math_agent.supervisor import (
    FailureRecord,
    RunInspection,
    SupervisorPolicy,
    WorkerResult,
    supervise_loop,
    start_detached_supervisor,
    _process_worker,
    failure_record_for_exception,
)


def test_supervisor_recovers_retriable_502_then_completes():
    calls = []
    worker_results = iter([
        WorkerResult(1, FailureRecord(node="writer_section", kind="server", retriable=True,
                                      message="502")),
        WorkerResult(0),
    ])
    inspections = iter([
        RunInspection(checkpoint_exists=False),
        RunInspection(checkpoint_exists=True, next_node="writer_section"),
        RunInspection(checkpoint_exists=True, final_status="completed"),
    ])

    result = supervise_loop(
        worker=lambda mode: calls.append(mode) or next(worker_results),
        inspect=lambda: next(inspections),
        policy=SupervisorPolicy(base_delay=0),
        sleep=lambda _: None,
    )

    assert result.status == "completed"
    assert calls == ["run", "recover"]
    assert result.attempts == 2


def test_supervisor_stops_after_same_node_fails_three_times():
    calls = []
    inspections = iter([
        RunInspection(checkpoint_exists=False),
        RunInspection(checkpoint_exists=True, next_node="sensitivity_interpret"),
        RunInspection(checkpoint_exists=True, next_node="sensitivity_interpret"),
        RunInspection(checkpoint_exists=True, next_node="sensitivity_interpret"),
    ])

    result = supervise_loop(
        worker=lambda mode: calls.append(mode) or WorkerResult(
            1, FailureRecord(node="sensitivity_interpret", kind="timeout", retriable=True,
                             message="timeout")
        ),
        inspect=lambda: next(inspections),
        policy=SupervisorPolicy(same_node_limit=3, max_recoveries=20, base_delay=0),
        sleep=lambda _: None,
    )

    assert result.status == "blocked"
    assert result.same_node_failures == 3
    assert calls == ["run", "recover", "recover"]


def test_supervisor_does_not_retry_non_retriable_configuration_error():
    calls = []
    inspections = iter([
        RunInspection(checkpoint_exists=False),
        RunInspection(checkpoint_exists=True, next_node="analyst"),
    ])

    result = supervise_loop(
        worker=lambda mode: calls.append(mode) or WorkerResult(
            1, FailureRecord(node="analyst", kind="authentication", retriable=False,
                             message="401")
        ),
        inspect=lambda: next(inspections),
        policy=SupervisorPolicy(base_delay=0),
        sleep=lambda _: None,
    )

    assert result.status == "blocked"
    assert calls == ["run"]


def test_supervisor_treats_abrupt_worker_death_as_recoverable_interruption():
    calls = []
    worker_results = iter([WorkerResult(-9), WorkerResult(0)])
    inspections = iter([
        RunInspection(checkpoint_exists=False),
        RunInspection(checkpoint_exists=True, next_node="coder_execute"),
        RunInspection(checkpoint_exists=True, final_status="completed"),
    ])

    result = supervise_loop(
        worker=lambda mode: calls.append(mode) or next(worker_results),
        inspect=lambda: next(inspections),
        policy=SupervisorPolicy(base_delay=0),
        sleep=lambda _: None,
    )

    assert result.status == "completed"
    assert calls == ["run", "recover"]


def test_supervisor_returns_paused_without_spawning_worker():
    calls = []
    result = supervise_loop(
        worker=lambda mode: calls.append(mode) or WorkerResult(0),
        inspect=lambda: RunInspection(checkpoint_exists=True, next_node="human_review"),
        policy=SupervisorPolicy(base_delay=0),
        sleep=lambda _: None,
    )
    assert result.status == "paused"
    assert calls == []


def test_supervisor_auto_approve_continues_from_human_review():
    calls = []
    inspections = iter([
        RunInspection(checkpoint_exists=True, next_node="human_review"),
        RunInspection(checkpoint_exists=True, final_status="completed"),
    ])
    result = supervise_loop(
        worker=lambda mode: calls.append(mode) or WorkerResult(0),
        inspect=lambda: next(inspections),
        policy=SupervisorPolicy(base_delay=0, auto_approve=True),
        sleep=lambda _: None,
    )
    assert result.status == "completed"
    assert calls == ["recover"]


def test_supervisor_reports_degraded_terminal_state():
    result = supervise_loop(
        worker=lambda mode: WorkerResult(0),
        inspect=lambda: RunInspection(checkpoint_exists=True, final_status="degraded"),
        policy=SupervisorPolicy(base_delay=0),
        sleep=lambda _: None,
    )
    assert result.status == "degraded"


def test_detached_supervisor_redirects_logs_and_records_pid(mocker, tmp_path):
    proc = mocker.MagicMock(pid=4321)
    popen = mocker.patch("math_agent.supervisor.subprocess.Popen", return_value=proc)

    pid = start_detached_supervisor(
        out=tmp_path, supervise_args=["--problem", "problem.json", "--no-interrupt"],
        cwd=tmp_path,
    )

    assert pid == 4321
    command = popen.call_args.args[0]
    assert command[-3:] == ["--problem", "problem.json", "--no-interrupt"]
    assert popen.call_args.kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
    assert (tmp_path / "launcher.json").exists()


def test_process_worker_returns_immediately_when_child_finishes(tmp_path):
    state = {"attempt": 1}
    started = time.monotonic()
    result = _process_worker(
        mode="run",
        command=[sys.executable, "-c", "raise SystemExit(0)"],
        out=tmp_path,
        state=state,
        heartbeat_seconds=2.0,
    )
    elapsed = time.monotonic() - started
    assert result.returncode == 0
    assert elapsed < 1.5


def test_process_worker_marks_child_as_supervised(tmp_path):
    result = _process_worker(
        mode="recover",
        command=[
            sys.executable,
            "-c",
            "import os; raise SystemExit(0 if os.getenv('MATH_AGENT_SUPERVISED') == '1' else 9)",
        ],
        out=tmp_path,
        state={"attempt": 1},
        heartbeat_seconds=0.1,
    )

    assert result.returncode == 0



def test_failure_record_marks_validation_error_retriable():
    record = failure_record_for_exception("analyst", LLMValidationError("bad json"))
    assert record.retriable is True
    assert record.node == "analyst"
