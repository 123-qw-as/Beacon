"""math-agent CLI（Plan C 版）。

run     : 启动一次任务（默认在 human_review 处中断）
resume  : 提供 human decision 并续跑
report  : 打印一次运行的 trace 报告
ingest  : 把语料目录嵌入到向量库（RAG 索引）
bench   : 真跑历年题回归基准（live 模式）
"""
from __future__ import annotations
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from math_agent.graph import build_graph
from math_agent.checkpointing import sqlite_saver
from math_agent.state import HumanDecision, DataFileInfo
from math_agent.errors import LLMError, LLMRateLimitError, LLMTransportError
from math_agent.nodes.finalizer import load_verified_completion
from math_agent.run_lock import RunLock, RunLockedError
from math_agent.supervisor import (
    SupervisorPolicy,
    clear_failure_report,
    failure_record_for_exception,
    inspect_checkpoint,
    reconcile_supervisor_state,
    run_process_supervisor,
    start_detached_supervisor,
    write_failure_report,
)
from math_agent.tracing import (
    Tracer, get_last_node, set_current, reset_current, clear_failed_node,
)


app = typer.Typer(help="Math modeling multi-agent system.")


def _field(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _dump_model(obj):
    if isinstance(obj, dict):
        return obj
    return obj.model_dump()


def _read_state_summary_data(out: Path, thread: str = "default") -> dict | None:
    """从 checkpoint 读 final state，提取 blueprint/critic/consistency 摘要数据。

    _dump_state_summary 和 _print_blueprint_summary 共用此函数，避免重复代码。
    无 checkpoint 或读取失败时返回 None。
    """
    chk = out / "checkpoints.sqlite"
    if not chk.exists():
        return None
    try:
        with _saver_cm(out) as saver:
            g = build_graph(checkpointer=saver)
            snap = g.get_state(_config(thread))
    except Exception:
        return None
    if snap is None or snap.values is None:
        return None

    state = snap.values

    def _get(key, default=None):
        return _field(state, key, default)

    bp = _get("problem_blueprint")
    critics = _get("critic_reports") or []
    mc_reports = _get("model_code_reports") or []
    models = _get("model_versions") or []
    evaluation = _get("evaluation")

    bp_critic = next(
        (r for r in reversed(critics)
         if _field(r, "target", "") == "analyst" and _field(r, "critic_type", "") == "blueprint"), None)
    model_critic = next(
        (r for r in reversed(critics) if _field(r, "target", "") == "modeler"), None)
    paper_critic = next(
        (r for r in reversed(critics) if _field(r, "target", "") == "paper"), None)

    total_sq = len(_field(bp, "subquestions", []) or []) if bp else 0
    covered = len(_field(models[-1], "question_coverage", []) or []) if models else 0

    unresolved = 0
    for r in critics:
        if not _field(r, "approved", True):
            unresolved += len(_field(r, "issues", []) or [])
    for r in mc_reports:
        if not _field(r, "approved", True):
            unresolved += len(_field(r, "issues", []) or [])

    return {
        "bp": bp,
        "bp_critic": bp_critic,
        "model_critic": model_critic,
        "paper_critic": paper_critic,
        "mc_reports": mc_reports,
        "models": models,
        "total_sq": total_sq,
        "covered": covered,
        "unresolved": unresolved,
        "stage_target": _get("stage_target"),
        "iteration": _get("iteration"),
        "evaluation_overall": _field(evaluation, "overall", None) if evaluation else None,
    }


def _dump_state_summary(out: Path, thread: str = "default") -> None:
    """从 checkpoint 读 final state，写 state_summary.json 供 Web UI 消费。

    ponytail: 直接从 checkpoint 读，不侵入 graph 节点。无 checkpoint 时静默跳过。
    """
    data = _read_state_summary_data(out, thread)
    if data is None:
        return

    bp = data["bp"]
    bp_critic = data["bp_critic"]
    model_critic = data["model_critic"]
    paper_critic = data["paper_critic"]
    mc_reports = data["mc_reports"]

    summary = {
        "problem_blueprint": _dump_model(bp) if bp else None,
        "blueprint_critic": {
            "score": _field(bp_critic, "score", None),
            "approved": _field(bp_critic, "approved", None),
            "issues": [_field(i, "problem", str(i)) for i in _field(bp_critic, "issues", []) or []]
                       if bp_critic else [],
            "suggestions": _field(bp_critic, "suggestions", []) or [] if bp_critic else [],
        } if bp_critic else None,
        "model_critic": {
            "score": _field(model_critic, "score", None),
            "approved": _field(model_critic, "approved", None),
        } if model_critic else None,
        "paper_critic": {
            "score": _field(paper_critic, "score", None),
            "approved": _field(paper_critic, "approved", None),
        } if paper_critic else None,
        "model_code_consistency": {
            "score": _field(mc_reports[-1], "score", None) if mc_reports else None,
            "approved": _field(mc_reports[-1], "approved", None) if mc_reports else None,
            "missing_variables": _field(mc_reports[-1], "missing_variables", []) or [] if mc_reports else [],
            "missing_objectives": _field(mc_reports[-1], "missing_objectives", []) or [] if mc_reports else [],
            "missing_constraints": _field(mc_reports[-1], "missing_constraints", []) or [] if mc_reports else [],
            "issues": _field(mc_reports[-1], "issues", []) or [] if mc_reports else [],
        } if mc_reports else None,
        "question_coverage": f"{data['covered']}/{data['total_sq']}",
        "unresolved_issues": data["unresolved"],
        "stage_target": data["stage_target"],
        "iteration": data["iteration"],
        "evaluation_overall": data["evaluation_overall"],
    }
    (out / "state_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _saver_cm(out: Path):
    """返回 SqliteSaver 的 contextmanager；调用方需用 with 包起来。"""
    out.mkdir(parents=True, exist_ok=True)
    return sqlite_saver(out / "checkpoints.sqlite")


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _failure_node(exc: BaseException) -> str:
    return str(getattr(exc, "_math_agent_failed_node", "") or get_last_node())


def _record_failure(out: Path, exc: BaseException):
    record = failure_record_for_exception(_failure_node(exc), exc)
    write_failure_report(out, record)
    return record


def _require_checkpoint(out: Path) -> None:
    checkpoint = out / "checkpoints.sqlite"
    if not checkpoint.is_file():
        typer.echo(f"no checkpoint at {checkpoint}; run a pipeline first", err=True)
        raise typer.Exit(1)


def _require_trace_thread(out: Path, thread: str) -> None:
    """防止错误 thread 的 resume/recover 覆盖同目录中另一任务的 trace。"""
    trace_path = out / "trace.json"
    if not trace_path.is_file():
        return
    try:
        blob = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return  # 损坏 trace 不阻止 checkpoint 恢复；Tracer 会从空统计开始
    existing_thread = blob.get("thread_id") if isinstance(blob, dict) else None
    if existing_thread and existing_thread != thread:
        typer.echo(
            f"trace at {trace_path} belongs to thread={existing_thread}, not {thread}",
            err=True,
        )
        raise typer.Exit(1)


def _read_problem_spec(problem: Path) -> dict:
    """读取并校验题目 JSON；任何破坏性 ``--force`` 操作都必须晚于此步骤。"""
    try:
        spec = json.loads(problem.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"题目文件不是有效的 UTF-8 JSON：{exc}", param_hint="--problem") from exc
    if not isinstance(spec, dict):
        raise typer.BadParameter("题目 JSON 顶层必须是对象", param_hint="--problem")

    title = spec.get("title", "")
    background = spec.get("background", "")
    questions = spec.get("questions", [])
    if not isinstance(title, str) or not isinstance(background, str):
        raise typer.BadParameter("title 和 background 必须是字符串", param_hint="--problem")
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        raise typer.BadParameter("questions 必须是字符串数组", param_hint="--problem")
    if not title.strip() and not any(q.strip() for q in questions):
        raise typer.BadParameter("title 与 questions 不能同时为空", param_hint="--problem")

    data_files = spec.get("data_files", [])
    data_dir = spec.get("data_dir", "")
    if not isinstance(data_files, list):
        raise typer.BadParameter("data_files 必须是数组", param_hint="--problem")
    if not isinstance(data_dir, str):
        raise typer.BadParameter("data_dir 必须是字符串", param_hint="--problem")
    if data_dir:
        data_dir_path = Path(data_dir)
        if not data_dir_path.is_absolute():
            data_dir_path = problem.parent / data_dir_path
        if not data_dir_path.is_dir():
            raise typer.BadParameter(
                f"data_dir 不存在: {data_dir_path}", param_hint="--problem"
            )
        data_dir = str(data_dir_path.resolve())

    return {"title": title, "background": background, "questions": questions,
            "data_files": data_files, "data_dir": data_dir}


def _problem_fingerprint(spec: dict) -> str:
    canonical = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_run_manifest(
    out: Path, thread: str, spec: dict, *, no_interrupt: bool = False
) -> None:
    path = out / "run_manifest.json"
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps({
            "thread": thread,
            "problem_sha256": _problem_fingerprint(spec),
            "no_interrupt": no_interrupt,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _validate_existing_run_manifest(out: Path, thread: str, spec: dict, force: bool) -> None:
    if force or not (out / "checkpoints.sqlite").is_file():
        return
    path = out / "run_manifest.json"
    if not path.is_file():
        typer.echo("[supervisor] 旧 checkpoint 没有 run_manifest，按 thread 兼容恢复。", err=True)
        return
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"run_manifest.json 损坏：{exc}", param_hint="--out") from exc
    if manifest.get("thread") != thread:
        raise typer.BadParameter(
            f"输出目录属于 thread={manifest.get('thread')}，不是 {thread}", param_hint="--thread",
        )
    if manifest.get("problem_sha256") != _problem_fingerprint(spec):
        raise typer.BadParameter(
            "输出目录中的 checkpoint 属于另一道题；请换 --out，或明确使用 --force",
            param_hint="--problem",
        )


def _prepare_run_output(out: Path, thread: str, force: bool) -> None:
    """在 worker 锁内检查/清理输出目录，防止 --force 与活跃任务竞态。"""
    checkpoint = out / "checkpoints.sqlite"
    if checkpoint.exists() and not force:
        typer.echo(
            f"Output dir {out} already has a checkpoint (thread={thread}).\n"
            f"  - Use a different --out to start a fresh run.\n"
            f"  - Or append --force to overwrite the existing run.\n"
            f"  - Or use `recover` to continue the existing run.",
            err=True,
        )
        raise typer.Exit(1)
    if not force:
        return
    for checkpoint_file in (
        checkpoint, Path(str(checkpoint) + "-wal"), Path(str(checkpoint) + "-shm"),
    ):
        checkpoint_file.unlink(missing_ok=True)
    for stale_name in (
        "trace.json", "state_summary.json", "paper.md", "paper.tex", "paper.pdf",
        "completion.json", "final_state.json", "failure.json", "supervisor.json",
        "run_manifest.json",
    ):
        (out / stale_name).unlink(missing_ok=True)


@app.command()
def run(
    problem: Path = typer.Option(..., exists=True, readable=True),
    out: Path = typer.Option(Path("runs/latest")),
    thread: str = typer.Option("default"),
    no_interrupt: bool = typer.Option(False, "--no-interrupt", help="跳过 HITL，直接跑到底"),
    template: str = typer.Option("default", help="LaTeX 模板：default | gmcm（国赛 gmcmthesis）"),
    school: str = typer.Option("", help="学校名称（gmcm 模板用）"),
    team_id: str = typer.Option("", help="参赛报名号（gmcm 模板用）"),
    members: str = typer.Option("", help="队员名字，逗号分隔：'张三,李四,王五'（gmcm 模板用）"),
    force: bool = typer.Option(False, "--force", help="即使已有 checkpoint 也覆盖（慎用）"),
):
    spec = _read_problem_spec(problem)
    if template not in {"default", "gmcm"}:
        raise typer.BadParameter("template 只能是 default 或 gmcm", param_hint="--template")

    # 防止以同一 --thread 重复输出到同一目录，掩盖上次 runs
    out.mkdir(parents=True, exist_ok=True)
    interrupt = [] if no_interrupt else ["human_review"]

    initial = {
        "problem": spec.get("title", "") + "\n" + "\n".join(spec.get("questions", [])),
        "background": spec.get("background", ""),
        "questions": spec.get("questions", []),
        "stage_target": "basic",
        "iteration": 0,
        "output_dir": str(out),
        "data_dir": spec.get("data_dir") or None,
        "data_files": [DataFileInfo(**f) for f in spec.get("data_files", [])],
        "latex_template": template,
        "school": school or None,
        "team_id": team_id or None,
        "members": members or None,
        # --no-interrupt 表示显式跳过人审，等价于自动批准；否则拒绝路由
        # 无法区分“自动模式”与“恢复时遗漏决定”。
        "human_decision": HumanDecision(approved=True, notes="--no-interrupt") if no_interrupt else None,
    }
    clear_failed_node()
    tracer = None
    tok = None
    try:
        with RunLock(out):
            _prepare_run_output(out, thread, force)
            _write_run_manifest(out, thread, spec, no_interrupt=no_interrupt)
            clear_failure_report(out)
            tracer = Tracer(thread_id=thread, out_dir=out)
            tok = set_current(tracer)
            with _saver_cm(out) as saver:
                g = build_graph(checkpointer=saver, interrupt_before=interrupt)
                g.invoke(initial, config=_config(thread))
    except RunLockedError as e:
        typer.echo(f"[BUSY] {e}", err=True)
        raise typer.Exit(75)
    except LLMTransportError as e:
        failure = _record_failure(out, e)
        typer.echo(f"\n[FAILED] LLM transport error at node '{failure.node}': {e}", err=True)
        typer.echo(f"  Checkpoint saved (thread={thread}). Router may be temporarily unavailable.", err=True)
        typer.echo(f"  Recover: uv run math-agent recover --out {out} --thread {thread}", err=True)
        raise typer.Exit(1)
    except LLMRateLimitError as e:
        failure = _record_failure(out, e)
        typer.echo(f"\n[FAILED] Rate limit exhausted at node '{failure.node}': {e}", err=True)
        typer.echo(f"  Retry budget used up. Wait and resume:", err=True)
        typer.echo(f"  uv run math-agent recover --out {out} --thread {thread}")
        raise typer.Exit(1)
    except LLMError as e:
        failure = _record_failure(out, e)
        typer.echo(f"\n[FAILED] LLM error at node '{failure.node}': {e}", err=True)
        typer.echo(f"  Checkpoint saved (thread={thread}). This error may not be retriable.", err=True)
        raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        failure = _record_failure(out, e)
        typer.echo(f"\n[FAILED] Unexpected error: {type(e).__name__}: {e}", err=True)
        typer.echo(f"  Checkpoint saved (thread={thread}). Debug trace at {out / 'trace.json'}")
        raise typer.Exit(1)
    finally:
        if tracer is not None:
            tracer.flush()
        if tok is not None:
            reset_current(tok)
    clear_failure_report(out)
    _dump_state_summary(out, thread)
    if interrupt:
        typer.echo(f"pipeline paused before human_review (thread={thread}); trace at {out / 'trace.json'}")
        typer.echo(f"use `math-agent resume --out {out} --thread {thread} --approve` to continue.")
    else:
        typer.echo(f"done. paper at {out / 'paper.md'}; trace at {out / 'trace.json'}")


@app.command()
def resume(
    out: Path = typer.Option(Path("runs/latest")),
    thread: str = typer.Option("default"),
    approve: bool | None = typer.Option(
        None, "--approve/--no-approve",
        help="必须明确批准或拒绝，避免无参数恢复时意外最终定稿",
    ),
    notes: str = typer.Option(""),
):
    if approve is None:
        raise typer.BadParameter(
            "必须显式传入 --approve 或 --no-approve", param_hint="--approve/--no-approve",
        )
    _require_checkpoint(out)
    _require_trace_thread(out, thread)
    clear_failed_node()
    clear_failure_report(out)
    tracer = Tracer(thread_id=thread, out_dir=out, append_existing=True)
    tok = set_current(tracer)
    try:
        with RunLock(out):
            with _saver_cm(out) as saver:
                g = build_graph(checkpointer=saver, interrupt_before=["human_review"])
                snapshot = g.get_state(_config(thread))
                if snapshot is None or not snapshot.values:
                    raise ValueError(f"checkpoint has no state for thread={thread}")
                g.update_state(_config(thread),
                               {"human_decision": HumanDecision(approved=approve, notes=notes)})
                g.invoke(None, config=_config(thread))
    except RunLockedError as e:
        typer.echo(f"[BUSY] {e}", err=True)
        raise typer.Exit(75)
    except LLMError as e:
        failure = _record_failure(out, e)
        typer.echo(f"[FAILED] LLM error at node '{failure.node}': {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        failure = _record_failure(out, e)
        typer.echo(f"[FAILED] resume error at node '{failure.node}': {type(e).__name__}: {e}", err=True)
        raise typer.Exit(1)
    finally:
        tracer.flush()
        reset_current(tok)
    clear_failure_report(out)
    _dump_state_summary(out, thread)
    if approve:
        typer.echo(f"done. tex/md written to {out}")
    else:
        typer.echo(f"pipeline rejected at human_review; no finalization was performed for {out}")


@app.command()
def recover(
    out: Path = typer.Option(Path("runs/latest")),
    thread: str = typer.Option("default"),
    no_interrupt: bool = typer.Option(
        False, "--no-interrupt", help="沿用自动批准策略，不在 human_review 暂停"
    ),
):
    """从最近 checkpoint 续跑，不注入 human_decision。

    用于 writer/coder/figure 等节点崩溃后的恢复。与 resume 的区别：
    resume 注入 human_decision 服务 human_review；recover 纯续跑。
    writer 子流程拆成 prep + section 循环后，section 崩溃不丢已完成节。

    附录 A.4 死循环防护：recover 检测同一节点连续失败次数，超过阈值后停止。
    """
    _require_checkpoint(out)
    _require_trace_thread(out, thread)
    try:
        manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        no_interrupt = no_interrupt or bool(manifest.get("no_interrupt", False))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        pass
    clear_failed_node()
    clear_failure_report(out)
    tracer = Tracer(thread_id=thread, out_dir=out, append_existing=True)
    tok = set_current(tracer)
    # 连续失败计数只防止真正的死循环；间歇性 502 至少允许三次人工 recover。
    fail_marker = out / ".recover_failed_node"
    supervised = os.getenv("MATH_AGENT_SUPERVISED", "") == "1"
    failure_state = {"node": "", "count": 0}
    if fail_marker.exists() and not supervised:
        try:
            loaded = json.loads(fail_marker.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                failure_state = {"node": str(loaded.get("node", "")),
                                 "count": int(loaded.get("count", 0))}
        except (OSError, ValueError, json.JSONDecodeError):
            # 兼容旧版本只写节点名的 marker：视为失败一次，而不是立即阻断。
            failure_state = {"node": fail_marker.read_text(encoding="utf-8").strip(), "count": 1}
    _RECOVER_FAIL_LIMIT = 3

    def _record_recover_failure(node: str) -> None:
        count = failure_state["count"] + 1 if failure_state["node"] == node else 1
        fail_marker.write_text(json.dumps({"node": node, "count": count}, ensure_ascii=False),
                               encoding="utf-8")
    try:
        with RunLock(out), _saver_cm(out) as saver:
            g = build_graph(
                checkpointer=saver,
                interrupt_before=[] if no_interrupt else ["human_review"],
            )
            snapshot = g.get_state(_config(thread))
            if snapshot is None or not snapshot.values:
                raise ValueError(f"checkpoint has no state for thread={thread}")
            # 同一微节点已经连续失败三次才停止自动重跑。
            current_node = ""
            try:
                current_node = snapshot.next[0] if snapshot.next else ""
            except (IndexError, TypeError):
                pass
            if (not supervised and current_node and failure_state["node"] == current_node
                    and failure_state["count"] >= _RECOVER_FAIL_LIMIT):
                typer.echo(
                    f"[BLOCKED] 节点 '{current_node}' 已连续 recover 失败 "
                    f"{failure_state['count']} 次。\n"
                    f"  可能是 prompt 过长或输出过大导致持续 timeout。\n"
                    f"  建议：1) 检查该节点的 prompt/schema  2) 手动调整 .env 中的 timeout\n"
                    f"  3) 确认问题后删除 {fail_marker} 再重试 recover。",
                    err=True,
                )
                raise typer.Exit(1)
            auto_approve_key = "MATH_AGENT_AUTO_APPROVE_HUMAN_REVIEW"
            previous_auto_approve = os.environ.get(auto_approve_key)
            if no_interrupt:
                # 不在任意中间 checkpoint 上调用 update_state。LangGraph 无法从一条
                # 人工恢复分支可靠推断“最后执行节点”，这会把 next_node 错路由到旧节点。
                # 自动批准只在 human_review 节点真正执行时生效，因此不改变恢复点。
                os.environ[auto_approve_key] = "1"
            try:
                g.invoke(None, config=_config(thread))
            finally:
                if previous_auto_approve is None:
                    os.environ.pop(auto_approve_key, None)
                else:
                    os.environ[auto_approve_key] = previous_auto_approve
        # 成功则清除失败标记
        fail_marker.unlink(missing_ok=True)
    except RunLockedError as e:
        typer.echo(f"[BUSY] {e}", err=True)
        raise typer.Exit(75)
    except LLMError as e:
        failure = _record_failure(out, e)
        failed = failure.node
        if not supervised:
            _record_recover_failure(failed)
        typer.echo(f"[FAILED] LLM error at node '{failed}': {e}", err=True)
        typer.echo(f"  若同一节点反复失败，检查 prompt/timeout 或手动干预。", err=True)
        raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        failure = _record_failure(out, e)
        failed = failure.node
        if not supervised:
            _record_recover_failure(failed)
        typer.echo(f"[FAILED] recover error at node '{failed}': {type(e).__name__}: {e}", err=True)
        raise typer.Exit(1)
    finally:
        tracer.flush()
        reset_current(tok)
    clear_failure_report(out)
    _dump_state_summary(out, thread)
    typer.echo(f"recovered. trace at {out / 'trace.json'}")


def _supervisor_exit(result, out: Path, thread: str) -> None:
    if result.status == "completed":
        typer.echo(f"supervised run completed (thread={thread}); paper at {out / 'paper.md'}")
        return
    if result.status == "paused":
        typer.echo(f"pipeline paused before human_review (thread={thread}); trace at {out / 'trace.json'}")
        return
    if result.status == "rejected":
        typer.echo(f"pipeline rejected at human_review; no finalization was performed for {out}")
        return
    if result.status == "degraded":
        typer.echo(
            f"[DEGRADED] 流程已收口但存在警告；查看 {out / 'completion.json'}",
            err=True,
        )
        raise typer.Exit(2)
    typer.echo(
        f"[BLOCKED] 自动恢复停止于节点 '{result.last_node}'：{result.message}\n"
        f"  状态：{out / 'supervisor.json'}\n"
        f"  失败：{out / 'failure.json'}",
        err=True,
    )
    raise typer.Exit(1)


@app.command()
def supervise(
    problem: Path = typer.Option(..., exists=True, readable=True),
    out: Path = typer.Option(Path("runs/latest")),
    thread: str = typer.Option("default"),
    no_interrupt: bool = typer.Option(False, "--no-interrupt", help="跳过 HITL，直接跑到底"),
    template: str = typer.Option("default", help="LaTeX 模板：default | gmcm"),
    school: str = typer.Option(""),
    team_id: str = typer.Option(""),
    members: str = typer.Option(""),
    force: bool = typer.Option(False, "--force", help="清除旧 checkpoint 后启动新任务"),
    same_node_limit: int = typer.Option(3, min=1, help="同一微节点连续失败上限"),
    max_recoveries: int = typer.Option(20, min=1, help="整次任务自动恢复总上限"),
    retry_delay: float = typer.Option(2.0, min=0.0, help="恢复退避基准秒数"),
):
    """以独立 worker 运行完整流程；崩溃或可恢复故障后自动从 checkpoint 续跑。"""
    spec = _read_problem_spec(problem)
    if template not in {"default", "gmcm"}:
        raise typer.BadParameter("template 只能是 default 或 gmcm", param_hint="--template")
    out = out.resolve()
    _validate_existing_run_manifest(out, thread, spec, force)
    args = ["--problem", str(problem.resolve()), "--out", str(out), "--thread", thread,
            "--template", template]
    if no_interrupt:
        args.append("--no-interrupt")
    if school:
        args.extend(["--school", school])
    if team_id:
        args.extend(["--team-id", team_id])
    if members:
        args.extend(["--members", members])
    if force:
        args.append("--force")
    try:
        result = run_process_supervisor(
            out=out,
            thread=thread,
            run_args=args,
            initial_mode="run" if force else None,
            policy=SupervisorPolicy(
                same_node_limit=same_node_limit,
                max_recoveries=max_recoveries,
                base_delay=retry_delay,
                auto_approve=no_interrupt,
            ),
        )
    except RunLockedError as e:
        typer.echo(f"[BUSY] {e}", err=True)
        raise typer.Exit(75)
    _supervisor_exit(result, out, thread)


@app.command()
def start(
    problem: Path = typer.Option(..., exists=True, readable=True),
    out: Path = typer.Option(Path("runs/latest")),
    thread: str = typer.Option("default"),
    no_interrupt: bool = typer.Option(False, "--no-interrupt"),
    template: str = typer.Option("default"),
    force: bool = typer.Option(False, "--force"),
    same_node_limit: int = typer.Option(3, min=1),
    max_recoveries: int = typer.Option(20, min=1),
    retry_delay: float = typer.Option(2.0, min=0.0),
):
    """在后台启动受监管任务，适合 Codex CLI、Claude CLI 和短生命周期终端。"""
    spec = _read_problem_spec(problem)
    if template not in {"default", "gmcm"}:
        raise typer.BadParameter("template 只能是 default 或 gmcm", param_hint="--template")
    out = out.resolve()
    _validate_existing_run_manifest(out, thread, spec, force)
    args = [
        "--problem", str(problem.resolve()), "--out", str(out), "--thread", thread,
        "--template", template, "--same-node-limit", str(same_node_limit),
        "--max-recoveries", str(max_recoveries), "--retry-delay", str(retry_delay),
    ]
    if no_interrupt:
        args.append("--no-interrupt")
    if force:
        args.append("--force")
    pid = start_detached_supervisor(out=out, supervise_args=args, cwd=Path.cwd())
    typer.echo(f"Beacon supervisor 已在后台启动，PID={pid}")
    typer.echo(f"状态：uv run math-agent status --out {out} --thread {thread}")
    typer.echo(f"日志：{out / 'supervisor.log'}")


@app.command()
def status(
    out: Path = typer.Option(Path("runs/latest")),
    thread: str = typer.Option("default"),
):
    """读取 checkpoint、supervisor 和最终提交标记，不修改运行状态。"""
    inspection = inspect_checkpoint(out, thread)
    supervisor_state = None
    completion = None
    for path, target in ((out / "supervisor.json", "supervisor"),
                         (out / "completion.json", "completion")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            payload = None
        if target == "supervisor":
            supervisor_state = (
                reconcile_supervisor_state(payload) if isinstance(payload, dict) else payload
            )
        else:
            completion = payload
    verified_completion = load_verified_completion(out)
    if (
        supervisor_state
        and verified_completion is not None
        and supervisor_state.get("status") != verified_completion.status
    ):
        supervisor_state = dict(supervisor_state)
        supervisor_state["status"] = "stale"
        supervisor_state["stale_reason"] = (
            f"superseded_by_verified_completion:{verified_completion.status}"
        )
        supervisor_state["effective_status"] = verified_completion.status
    typer.echo(f"checkpoint: {'yes' if inspection.checkpoint_exists else 'no'}")
    typer.echo(f"next_node: {inspection.next_node or '-'}")
    typer.echo(f"final_status: {inspection.final_status or '-'}")
    if supervisor_state:
        typer.echo(f"supervisor_status: {supervisor_state.get('status', '-')}")
        typer.echo(f"heartbeat_at: {supervisor_state.get('heartbeat_at', '-')}")
        typer.echo(f"worker_pid: {supervisor_state.get('worker_pid', '-')}")
        if supervisor_state.get("stale_reason"):
            typer.echo(f"stale_reason: {supervisor_state['stale_reason']}")
        if supervisor_state.get("effective_status"):
            typer.echo(f"effective_status: {supervisor_state['effective_status']}")
    if completion:
        typer.echo(f"completion: {completion.get('status', '-')}")


@app.command("supervise-resume")
def supervise_resume(
    out: Path = typer.Option(Path("runs/latest")),
    thread: str = typer.Option("default"),
    approve: bool | None = typer.Option(None, "--approve/--no-approve"),
    notes: str = typer.Option(""),
    same_node_limit: int = typer.Option(3, min=1),
    max_recoveries: int = typer.Option(20, min=1),
    retry_delay: float = typer.Option(2.0, min=0.0),
):
    """提交人审决定，并监管 LaTeX/finalizer 直到形成明确终态。"""
    if approve is None:
        raise typer.BadParameter(
            "必须显式传入 --approve 或 --no-approve", param_hint="--approve/--no-approve",
        )
    _require_checkpoint(out)
    _require_trace_thread(out, thread)
    out = out.resolve()
    args = ["--out", str(out), "--thread", thread,
            "--approve" if approve else "--no-approve"]
    if notes:
        args.extend(["--notes", notes])
    try:
        result = run_process_supervisor(
            out=out,
            thread=thread,
            resume_args=args,
            initial_mode="resume",
            policy=SupervisorPolicy(
                same_node_limit=same_node_limit,
                max_recoveries=max_recoveries,
                base_delay=retry_delay,
                auto_approve=bool(approve),
            ),
        )
    except RunLockedError as e:
        typer.echo(f"[BUSY] {e}", err=True)
        raise typer.Exit(75)
    _supervisor_exit(result, out, thread)


@app.command()
def report(
    out: Path = typer.Option(Path("runs/latest")),
    thread: str = typer.Option("default"),
):
    """打印一次运行的 trace 报告 + per-model / per-node 摘要 + blueprint/一致性摘要。"""
    trace_path = out / "trace.json"
    if not trace_path.exists():
        typer.echo(f"no trace at {trace_path}")
        raise typer.Exit(1)
    try:
        blob = json.loads(trace_path.read_text(encoding="utf-8"))
        if not isinstance(blob, dict) or not isinstance(blob.get("tokens"), dict):
            raise ValueError("trace 顶层或 tokens 结构无效")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        typer.echo(f"invalid trace at {trace_path}: {exc}", err=True)
        raise typer.Exit(1)

    c = Console()
    t = Table(title=f"Run report ({blob.get('thread_id', thread)})")
    t.add_column("metric"); t.add_column("value")
    t.add_row("LLM calls", str(blob.get("llm_calls", 0)))
    t.add_row("Prompt tokens", str(blob["tokens"].get("prompt", 0)))
    t.add_row("Completion tokens", str(blob["tokens"].get("completion", 0)))
    c.print(t)

    tm = Table(title="Per model")
    tm.add_column("model"); tm.add_column("calls"); tm.add_column("prompt"); tm.add_column("completion")
    per_model = blob.get("per_model", {})
    if not isinstance(per_model, dict):
        per_model = {}
    for m, d in per_model.items():
        if isinstance(d, dict):
            tm.add_row(
                m, str(d.get("calls", 0)), str(d.get("prompt_tokens", 0)),
                str(d.get("completion_tokens", 0)),
            )
    c.print(tm)

    tn = Table(title="Nodes")
    tn.add_column("node"); tn.add_column("duration_ms")
    for n in blob.get("nodes", []):
        if isinstance(n, dict):
            tn.add_row(str(n.get("name", "(unknown)")), str(n.get("duration_ms", 0)))
    c.print(tn)

    # P2 §6.3: blueprint + 一致性摘要（从 checkpoint 读 final state）
    _print_blueprint_summary(c, out, thread)


def _print_blueprint_summary(c: Console, out: Path, thread: str = "default") -> None:
    """从 checkpoint 读取 final state，打印 blueprint/一致性摘要。"""
    data = _read_state_summary_data(out, thread)
    if data is None:
        return

    tq = Table(title="Blueprint & Consistency")
    tq.add_column("metric"); tq.add_column("value")

    # Blueprint critic score
    bp_critic = data["bp_critic"]
    if bp_critic is not None:
        tq.add_row("Blueprint Score", f"{_field(bp_critic, 'score', '?')}/10")
    else:
        tq.add_row("Blueprint Score", "N/A")

    # Model-code consistency score
    mc_reports = data["mc_reports"]
    if mc_reports:
        last = mc_reports[-1]
        tq.add_row("Model-Code Score", f"{_field(last, 'score', '?')}/10")
    else:
        tq.add_row("Model-Code Score", "N/A")

    # Question coverage
    tq.add_row("Question Coverage", f"{data['covered']}/{data['total_sq']}")

    # Unresolved issues
    tq.add_row("Unresolved Issues", str(data["unresolved"]))

    c.print(tq)


@app.command()
def ingest(
    src: Path = typer.Option(..., exists=True, readable=True),
    db: Path = typer.Option(Path("runs/rag.sqlite")),
    embedding_model: str = typer.Option("text-embedding-3-small"),
    dim: int = typer.Option(1536),
):
    """扫描语料目录 → 切块 → 嵌入 → 入 sqlite-vec 库。"""
    from math_agent.rag.ingest import ingest_directory
    rep = ingest_directory(src_dir=src, db_path=db,
                           embedding_model=embedding_model, dim=dim)
    typer.echo(f"files={rep.files_processed} chunks={rep.chunks_added} skipped={len(rep.skipped)}")


@app.command()
def bench(out: Path = typer.Option(Path("runs/bench"))):
    """真跑历年题回归基准（live 模式，需要真 LLM API key）。

    要跑 mock 模式（结构性校验，不消耗 API），用：pytest tests/bench/
    """
    from math_agent.bench.runner import run_bench
    rep = run_bench(out_dir=out)
    for c in rep.cases:
        flag = "PASS" if c.passed else "FAIL"
        typer.echo(f"[{flag}] {c.problem_id} overall={c.overall} failures={c.failures}")


if __name__ == "__main__":
    app()
