"""doc"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal

import psutil
from math_agent.checkpointing import sqlite_saver
from math_agent.errors import (
    FinalizationError,
    LLMAuthenticationError,
    LLMBudgetExceededError,
    LLMConnectionError,
    LLMError,
    LLMInvalidRequestError,
    LLMValidationError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    LLMTransportError,
)
from math_agent.run_lock import RunLock


WorkerMode = Literal["run", "resume", "recover"]


@dataclass(frozen=True)
class FailureRecord:
    node: str
    kind: str
    retriable: bool
    message: str = ""


@dataclass(frozen=True)
class WorkerResult:
    returncode: int
    failure: FailureRecord | None = None


@dataclass(frozen=True)
class RunInspection:
    checkpoint_exists: bool
    next_node: str = ""
    final_status: str = ""


@dataclass(frozen=True)
class SupervisorPolicy:
    same_node_limit: int = 3
    max_recoveries: int = 20
    base_delay: float = 2.0
    max_delay: float = 30.0
    auto_approve: bool = False


@dataclass(frozen=True)
class SupervisorResult:
    status: Literal["completed", "degraded", "paused", "rejected", "blocked"]
    attempts: int = 0
    recoveries: int = 0
    last_node: str = ""
    same_node_failures: int = 0
    message: str = ""


def _terminal(inspection: RunInspection, *, auto_approve: bool = False) -> str:
    if inspection.next_node == "human_review" and not auto_approve:
        return "paused"
    if inspection.checkpoint_exists and not inspection.next_node:
        if inspection.final_status in {"completed", "degraded", "rejected"}:
            return inspection.final_status
    return ""


def supervise_loop(
    *,
    worker: Callable[[WorkerMode], WorkerResult],
    inspect: Callable[[], RunInspection],
    policy: SupervisorPolicy | None = None,
    sleep: Callable[[float], None],
    initial_mode: WorkerMode | None = None,
) -> SupervisorResult:
    """doc"""
    policy = policy or SupervisorPolicy()
    inspection = inspect()
    terminal = _terminal(inspection, auto_approve=policy.auto_approve)
    if terminal and initial_mode is None:
        return SupervisorResult(status=terminal)  # type: ignore[arg-type]

    mode: WorkerMode = initial_mode or ("recover" if inspection.checkpoint_exists else "run")
    attempts = 0
    recoveries = 0
    last_node = ""
    same_node_failures = 0

    while True:
        result = worker(mode)
        attempts += 1
        inspection = inspect()
        terminal = _terminal(inspection, auto_approve=policy.auto_approve)
        if terminal:
            return SupervisorResult(
                status=terminal, attempts=attempts, recoveries=recoveries,
                last_node=last_node, same_node_failures=same_node_failures,
            )  # type: ignore[arg-type]

        if result.returncode == 0:
            failure = FailureRecord(
                node=inspection.next_node or "(unknown)", kind="incomplete",
                retriable=True, message="worker exited before reaching a terminal state",
            )
        elif result.failure is None:
            failure = FailureRecord(
                node=inspection.next_node or "(interrupted)", kind="interrupted",
                retriable=True, message=f"worker interrupted, exit={result.returncode}",
            )
        else:
            failure = result.failure

        if failure.node == last_node:
            same_node_failures += 1
        else:
            last_node = failure.node
            same_node_failures = 1

        if not failure.retriable:
            return SupervisorResult(
                status="blocked", attempts=attempts, recoveries=recoveries,
                last_node=last_node, same_node_failures=same_node_failures,
                message=failure.message,
            )
        if same_node_failures >= policy.same_node_limit:
            return SupervisorResult(
                status="blocked", attempts=attempts, recoveries=recoveries,
                last_node=last_node, same_node_failures=same_node_failures,
                message=f"same node failed {same_node_failures} times",
            )
        if recoveries >= policy.max_recoveries:
            return SupervisorResult(
                status="blocked", attempts=attempts, recoveries=recoveries,
                last_node=last_node, same_node_failures=same_node_failures,
                message=f"recovery budget exhausted ({policy.max_recoveries})",
            )

        delay = min(policy.max_delay, policy.base_delay * (2 ** max(0, same_node_failures - 1)))
        if delay > 0:
            sleep(delay)
        recoveries += 1
        mode = "recover"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _pid_identity_is_alive(pid: object, started_at: object = None) -> bool:
    """检查 PID 存活且未被后来创建的无关进程复用。"""
    try:
        process = psutil.Process(int(pid))
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return False
        expected_start = _parse_utc(started_at)
        if expected_start is not None:
            actual_start = datetime.fromtimestamp(process.create_time(), timezone.utc)
            if actual_start > expected_start.replace(microsecond=0) + timedelta(seconds=2):
                return False
        return True
    except (TypeError, ValueError, psutil.Error, OSError):
        return False


def reconcile_supervisor_state(
    payload: dict,
    *,
    now: datetime | None = None,
    stale_after_seconds: float = 30.0,
) -> dict:
    """返回监督状态的只读有效视图，识别死亡 PID、PID 复用和陈旧心跳。"""
    state = dict(payload)
    if state.get("status") not in {"starting", "running"}:
        return state

    supervisor_pid = state.get("supervisor_pid")
    # 兼容早期没有 supervisor_pid 的状态文件；不能仅凭 worker_pid 猜测监督器身份。
    if supervisor_pid is None:
        return state
    if not _pid_identity_is_alive(supervisor_pid, state.get("started_at")):
        state["status"] = "stale"
        state["stale_reason"] = "supervisor_pid_not_alive"
        return state

    heartbeat = _parse_utc(state.get("heartbeat_at"))
    if heartbeat is not None:
        age = ((now or datetime.now(timezone.utc)) - heartbeat).total_seconds()
        if age > max(1.0, stale_after_seconds):
            state["status"] = "stale"
            state["stale_reason"] = f"heartbeat_expired:{int(age)}s"
    return state

def failure_record_for_exception(node: str, exc: BaseException) -> FailureRecord:
    """doc"""
    retriable = isinstance(exc, (
        LLMRateLimitError,
        LLMConnectionError,
        LLMServerError,
        LLMTimeoutError,
        LLMBudgetExceededError,
        LLMTransportError,
        LLMValidationError,
    ))
    if isinstance(exc, (LLMAuthenticationError, LLMInvalidRequestError, FinalizationError)):
        retriable = False
    kind = type(exc).__name__
    message = str(exc).replace("\r", " ").replace("\n", " ")[:1000]
    return FailureRecord(node=node or "(unknown)", kind=kind, retriable=retriable,
                         message=message)


def write_failure_report(out: str | Path, record: FailureRecord) -> Path:
    path = Path(out) / "failure.json"
    _atomic_json(path, {
        "node": record.node,
        "kind": record.kind,
        "retriable": record.retriable,
        "message": record.message,
        "recorded_at": _now(),
    })
    return path


def clear_failure_report(out: str | Path) -> None:
    (Path(out) / "failure.json").unlink(missing_ok=True)


def load_failure_report(out: str | Path) -> FailureRecord | None:
    path = Path(out) / "failure.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return FailureRecord(
            node=str(payload.get("node", "(unknown)")),
            kind=str(payload.get("kind", "unknown")),
            retriable=bool(payload.get("retriable", False)),
            message=str(payload.get("message", "")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def inspect_checkpoint(out: str | Path, thread: str) -> RunInspection:
    out = Path(out)
    checkpoint = out / "checkpoints.sqlite"
    if not checkpoint.is_file():
        return RunInspection(checkpoint_exists=False)
    from math_agent.graph import build_graph

    config = {"configurable": {"thread_id": thread}}
    try:
        with sqlite_saver(checkpoint) as saver:
            snapshot = build_graph(checkpointer=saver, interrupt_before=["human_review"]).get_state(config)
    except Exception:
        return RunInspection(checkpoint_exists=True)
    if snapshot is None or not snapshot.values:
        return RunInspection(checkpoint_exists=True)
    next_node = snapshot.next[0] if snapshot.next else ""
    values = snapshot.values
    finalization = values.get("finalization") if isinstance(values, dict) else getattr(values, "finalization", None)
    if isinstance(finalization, dict):
        final_status = str(finalization.get("status", ""))
    else:
        final_status = str(getattr(finalization, "status", ""))
    if not next_node and final_status in {"", "pending"}:
        human = values.get("human_decision") if isinstance(values, dict) else getattr(values, "human_decision", None)
        approved = human.get("approved") if isinstance(human, dict) else getattr(human, "approved", None)
        if approved is False:
            final_status = "rejected"
        elif approved is True:
            from math_agent.nodes.finalizer import load_verified_completion
            committed = load_verified_completion(out)
            if committed is not None:
                final_status = committed.status
    return RunInspection(True, next_node=next_node, final_status=final_status)


def _process_worker(
    *,
    mode: WorkerMode,
    command: list[str],
    out: Path,
    state: dict,
    heartbeat_seconds: float = 5.0,
) -> WorkerResult:
    clear_failure_report(out)
    state.update({"status": "running", "mode": mode, "command": command, "heartbeat_at": _now()})
    _atomic_json(out / "supervisor.json", state)
    print(f"[supervisor] starting worker: {mode} (attempt {state.get('attempt', 1)})", flush=True)
    proc = subprocess.Popen(command, env={
        **os.environ,
        # 外层 supervisor 已统一管理同节点失败预算；避免 recover 子命令的本地
        # .recover_failed_node 计数器提前截断不同的 supervisor policy。
        "MATH_AGENT_SUPERVISED": "1",
        "PYTHONPATH": os.pathsep.join(filter(None, [
            str(Path(__file__).resolve().parents[1]),
            os.environ.get("PYTHONPATH", ""),
        ])),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    })
    try:
        while proc.poll() is None:
            state["worker_pid"] = proc.pid
            state["heartbeat_at"] = _now()
            _atomic_json(out / "supervisor.json", state)
            try:
                proc.wait(timeout=heartbeat_seconds)
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        raise
    return WorkerResult(proc.returncode or 0, load_failure_report(out))


def run_process_supervisor(
    *,
    out: str | Path,
    thread: str,
    run_args: list[str] | None = None,
    resume_args: list[str] | None = None,
    initial_mode: WorkerMode | None = None,
    policy: SupervisorPolicy | None = None,
) -> SupervisorResult:
    """doc"""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    policy = policy or SupervisorPolicy()
    manifest_auto_approve = False
    try:
        manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        manifest_auto_approve = bool(manifest.get("no_interrupt", False))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        pass
    auto_approve = policy.auto_approve or manifest_auto_approve or bool(
        run_args and "--no-interrupt" in run_args
    )
    if auto_approve != policy.auto_approve:
        policy = replace(policy, auto_approve=auto_approve)
    base = [sys.executable, "-m", "math_agent.cli"]
    state = {
        "thread": thread,
        "supervisor_pid": os.getpid(),
        "started_at": _now(),
        "attempt": 0,
        "status": "starting",
    }

    def worker(mode: WorkerMode) -> WorkerResult:
        state["attempt"] = int(state.get("attempt", 0)) + 1
        if mode == "run":
            if run_args is None:
                return WorkerResult(2, FailureRecord("supervisor", "configuration", False,
                                                     "缂哄皯 run 鍙傛暟"))
            command = [*base, "run", *run_args]
        elif mode == "resume":
            if resume_args is None:
                return WorkerResult(2, FailureRecord("supervisor", "configuration", False,
                                                     "缂哄皯 resume 鍙傛暟"))
            command = [*base, "resume", *resume_args]
        else:
            command = [*base, "recover", "--out", str(out), "--thread", thread]
            if auto_approve:
                command.append("--no-interrupt")
        return _process_worker(mode=mode, command=command, out=out, state=state)

    with RunLock(out, filename=".beacon-supervisor.lock"):
        result = supervise_loop(
            worker=worker,
            inspect=lambda: inspect_checkpoint(out, thread),
            policy=policy,
            sleep=time.sleep,
            initial_mode=initial_mode,
        )
        state.update({
            "status": result.status,
            "ended_at": _now(),
            "recoveries": result.recoveries,
            "last_node": result.last_node,
            "message": result.message,
        })
        _atomic_json(out / "supervisor.json", state)
        return result


def start_detached_supervisor(
    *, out: str | Path, supervise_args: list[str], cwd: str | Path | None = None,
) -> int:
    """doc"""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "math_agent.cli", "supervise", *supervise_args]
    log_path = out / "supervisor.log"
    log = log_path.open("ab")
    kwargs = {
        "cwd": str(Path(cwd).resolve()) if cwd is not None else None,
        "stdin": subprocess.DEVNULL,
        "stdout": log,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
        "env": {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, [
                str(Path(__file__).resolve().parents[1]),
                os.environ.get("PYTHONPATH", ""),
            ])),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        },
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(command, **kwargs)
    finally:
        log.close()
    _atomic_json(out / "launcher.json", {
        "pid": proc.pid,
        "started_at": _now(),
        "log": str(log_path),
    })
    return proc.pid


