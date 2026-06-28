"""math-agent CLI（Plan B 版）。

run    : 启动一次任务（默认在 human_review 处中断）
resume : 提供 human decision 并续跑
"""
from __future__ import annotations
import json
from pathlib import Path

import typer
from langgraph.checkpoint.sqlite import SqliteSaver

from math_agent.graph import build_graph
from math_agent.state import HumanDecision


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
    }
    with _saver_cm(out) as saver:
        g = build_graph(checkpointer=saver, interrupt_before=interrupt)
        g.invoke(initial, config=_config(thread))
    if interrupt:
        typer.echo(f"pipeline paused before human_review (thread={thread}).")
        typer.echo(f"use `math-agent resume --out {out} --thread {thread} --approve` to continue.")
    else:
        typer.echo(f"done. paper at {out / 'paper.md'}")


@app.command()
def resume(
    out: Path = typer.Option(Path("runs/latest")),
    thread: str = typer.Option("default"),
    approve: bool = typer.Option(True),
    notes: str = typer.Option(""),
):
    with _saver_cm(out) as saver:
        g = build_graph(checkpointer=saver, interrupt_before=["human_review"])
        g.update_state(_config(thread),
                       {"human_decision": HumanDecision(approved=approve, notes=notes)})
        g.invoke(None, config=_config(thread))
    typer.echo(f"done. tex/md written to {out}")


if __name__ == "__main__":
    app()
