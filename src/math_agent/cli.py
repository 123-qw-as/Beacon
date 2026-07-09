"""math-agent CLI（Plan C 版）。

run     : 启动一次任务（默认在 human_review 处中断）
resume  : 提供 human decision 并续跑
report  : 打印一次运行的 trace 报告
ingest  : 把语料目录嵌入到向量库（RAG 索引）
bench   : 真跑历年题回归基准（live 模式）
"""
from __future__ import annotations
import json
from pathlib import Path

import typer
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console
from rich.table import Table

from math_agent.graph import build_graph
from math_agent.state import HumanDecision
from math_agent.errors import LLMError, LLMRateLimitError, LLMTransportError
from math_agent.tracing import Tracer, get_last_node, set_current, reset_current


app = typer.Typer(help="Math modeling multi-agent system.")


def _saver_cm(out: Path):
    """返回 SqliteSaver 的 contextmanager；调用方需用 with 包起来。"""
    out.mkdir(parents=True, exist_ok=True)
    return SqliteSaver.from_conn_string(str(out / "checkpoints.sqlite"))


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@app.command()
def run(
    problem: Path = typer.Option(..., exists=True, readable=True),
    out: Path = typer.Option(Path("runs/latest")),
    thread: str = typer.Option("default"),
    no_interrupt: bool = typer.Option(False, help="跳过 HITL，直接跑到底"),
    template: str = typer.Option("default", help="LaTeX 模板：default | gmcm（国赛 gmcmthesis）"),
    school: str = typer.Option("", help="学校名称（gmcm 模板用）"),
    team_id: str = typer.Option("", help="参赛报名号（gmcm 模板用）"),
    members: str = typer.Option("", help="队员名字，逗号分隔：'张三,李四,王五'（gmcm 模板用）"),
    force: bool = typer.Option(False, "--force", help="即使已有 checkpoint 也覆盖（慎用）"),
):
    # 防止以同一 --thread 重复输出到同一目录，掩盖上次 runs
    out.mkdir(parents=True, exist_ok=True)
    chk = out / "checkpoints.sqlite"
    if chk.exists() and not force:
        typer.echo(
            f"Output dir {out} already has a checkpoint (thread={thread}).\n"
            f"  - Use a different --out or a different --thread to start a fresh run.\n"
            f"  - Or append --force to overwrite the existing run.\n"
            f"  - Or use `resume` to continue the existing run.",
            err=True,
        )
        raise typer.Exit(1)
    spec = json.loads(problem.read_text(encoding="utf-8"))
    interrupt = [] if no_interrupt else ["human_review"]

    initial = {
        "problem": spec.get("title", "") + "\n" + "\n".join(spec.get("questions", [])),
        "background": spec.get("background", ""),
        "questions": spec.get("questions", []),
        "stage_target": "basic",
        "iteration": 0,
        "output_dir": str(out),
        "latex_template": template,
        "school": school or None,
        "team_id": team_id or None,
        "members": members or None,
    }
    tracer = Tracer(thread_id=thread, out_dir=out)
    tok = set_current(tracer)
    try:
        with _saver_cm(out) as saver:
            g = build_graph(checkpointer=saver, interrupt_before=interrupt)
            g.invoke(initial, config=_config(thread))
    except LLMTransportError as e:
        typer.echo(f"\n[FAILED] LLM transport error at node '{get_last_node()}': {e}", err=True)
        typer.echo(f"  Checkpoint saved (thread={thread}). Router may be temporarily unavailable.", err=True)
        typer.echo(f"  Resume: python -m math_agent.cli resume --out {out} --thread {thread} --approve", err=True)
        typer.echo(f"  Or: python -m math_agent.cli run ... --out {out} --thread {thread}")
        raise typer.Exit(1)
    except LLMRateLimitError as e:
        typer.echo(f"\n[FAILED] Rate limit exhausted at node '{get_last_node()}': {e}", err=True)
        typer.echo(f"  Retry budget used up. Wait and resume:", err=True)
        typer.echo(f"  python -m math_agent.cli resume --out {out} --thread {thread} --approve")
        raise typer.Exit(1)
    except LLMError as e:
        typer.echo(f"\n[FAILED] LLM error at node '{get_last_node()}': {e}", err=True)
        typer.echo(f"  Checkpoint saved (thread={thread}). This error may not be retriable.", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"\n[FAILED] Unexpected error: {type(e).__name__}: {e}", err=True)
        typer.echo(f"  Checkpoint saved (thread={thread}). Debug trace at {out / 'trace.json'}")
        raise typer.Exit(1)
    finally:
        tracer.flush()
        reset_current(tok)
    if interrupt:
        typer.echo(f"pipeline paused before human_review (thread={thread}); trace at {out / 'trace.json'}")
        typer.echo(f"use `math-agent resume --out {out} --thread {thread} --approve` to continue.")
    else:
        typer.echo(f"done. paper at {out / 'paper.md'}; trace at {out / 'trace.json'}")


@app.command()
def resume(
    out: Path = typer.Option(Path("runs/latest")),
    thread: str = typer.Option("default"),
    approve: bool = typer.Option(True),
    notes: str = typer.Option(""),
):
    tracer = Tracer(thread_id=thread, out_dir=out)
    tok = set_current(tracer)
    try:
        with _saver_cm(out) as saver:
            g = build_graph(checkpointer=saver, interrupt_before=["human_review"])
            g.update_state(_config(thread),
                           {"human_decision": HumanDecision(approved=approve, notes=notes)})
            g.invoke(None, config=_config(thread))
    finally:
        tracer.flush()
        reset_current(tok)
    typer.echo(f"done. tex/md written to {out}")


@app.command()
def recover(
    out: Path = typer.Option(Path("runs/latest")),
    thread: str = typer.Option("default"),
):
    """从最近 checkpoint 续跑，不注入 human_decision。

    用于 writer/coder/figure 等节点崩溃后的恢复。与 resume 的区别：
    resume 注入 human_decision 服务 human_review；recover 纯续跑。
    writer 子流程拆成 prep + section 循环后，section 崩溃不丢已完成节。
    """
    tracer = Tracer(thread_id=thread, out_dir=out)
    tok = set_current(tracer)
    try:
        with _saver_cm(out) as saver:
            g = build_graph(checkpointer=saver, interrupt_before=["human_review"])
            g.invoke(None, config=_config(thread))
    finally:
        tracer.flush()
        reset_current(tok)
    typer.echo(f"recovered. trace at {out / 'trace.json'}")


@app.command()
def report(out: Path = typer.Option(Path("runs/latest"))):
    """打印一次运行的 trace 报告 + per-model / per-node 摘要 + blueprint/一致性摘要。"""
    trace_path = out / "trace.json"
    if not trace_path.exists():
        typer.echo(f"no trace at {trace_path}")
        raise typer.Exit(1)
    blob = json.loads(trace_path.read_text(encoding="utf-8"))

    c = Console()
    t = Table(title=f"Run report ({blob['thread_id']})")
    t.add_column("metric"); t.add_column("value")
    t.add_row("LLM calls", str(blob["llm_calls"]))
    t.add_row("Prompt tokens", str(blob["tokens"]["prompt"]))
    t.add_row("Completion tokens", str(blob["tokens"]["completion"]))
    c.print(t)

    tm = Table(title="Per model")
    tm.add_column("model"); tm.add_column("calls"); tm.add_column("prompt"); tm.add_column("completion")
    for m, d in blob.get("per_model", {}).items():
        tm.add_row(m, str(d["calls"]), str(d["prompt_tokens"]), str(d["completion_tokens"]))
    c.print(tm)

    tn = Table(title="Nodes")
    tn.add_column("node"); tn.add_column("duration_ms")
    for n in blob.get("nodes", []):
        tn.add_row(n["name"], str(n["duration_ms"]))
    c.print(tn)

    # P2 §6.3: blueprint + 一致性摘要（从 checkpoint 读 final state）
    _print_blueprint_summary(c, out)


def _print_blueprint_summary(c: Console, out: Path) -> None:
    """从 checkpoint 读取 final state，打印 blueprint/一致性摘要。"""
    chk = out / "checkpoints.sqlite"
    if not chk.exists():
        return
    try:
        with _saver_cm(out) as saver:
            g = build_graph(checkpointer=saver)
            snap = g.get_state(_config("default"))
    except Exception:
        return  # checkpoint 损坏等 -> 静默跳过
    if snap is None or snap.values is None:
        return

    state = snap.values

    def _get(key, default=None):
        if isinstance(state, dict):
            return state.get(key, default)
        return getattr(state, key, default)

    tq = Table(title="Blueprint & Consistency")
    tq.add_column("metric"); tq.add_column("value")

    # Blueprint critic score
    critics = _get("critic_reports") or []
    bp_critic = next(
        (r for r in reversed(critics) if getattr(r, "target", "") == "analyst"), None)
    if bp_critic is not None:
        tq.add_row("Blueprint Score", f"{getattr(bp_critic, 'score', '?')}/10")
    else:
        tq.add_row("Blueprint Score", "N/A")

    # Model-code consistency score
    mc_reports = _get("model_code_reports") or []
    if mc_reports:
        last = mc_reports[-1]
        tq.add_row("Model-Code Score", f"{getattr(last, 'score', '?')}/10")
    else:
        tq.add_row("Model-Code Score", "N/A")

    # Question coverage
    bp = _get("problem_blueprint")
    models = _get("model_versions") or []
    total_sq = len(getattr(bp, "subquestions", []) or []) if bp else 0
    covered = len(getattr(models[-1], "question_coverage", []) or []) if models else 0
    tq.add_row("Question Coverage", f"{covered}/{total_sq}")

    # Unresolved issues: 统计所有 critic/consistency 报告中未通过的 issue 数
    unresolved = 0
    for r in critics:
        if not getattr(r, "approved", True):
            unresolved += len(getattr(r, "issues", []) or [])
    for r in mc_reports:
        if not getattr(r, "approved", True):
            unresolved += len(getattr(r, "issues", []) or [])
    tq.add_row("Unresolved Issues", str(unresolved))

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
