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
from math_agent.tracing import Tracer, set_current, reset_current


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
):
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
def report(out: Path = typer.Option(Path("runs/latest"))):
    """打印一次运行的 trace 报告 + per-model / per-node 摘要。"""
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
