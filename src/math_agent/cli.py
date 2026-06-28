"""math-agent CLI。

用法:
  math-agent run --problem path/to/problem.json --out runs/2026-06-28
"""
import json
from pathlib import Path
import typer

from math_agent.graph import build_graph

app = typer.Typer(help="Math modeling multi-agent system (MVP).")


@app.command()
def run(
    problem: Path = typer.Option(..., exists=True, readable=True),
    out: Path = typer.Option(Path("runs/latest")),
):
    spec = json.loads(problem.read_text(encoding="utf-8"))
    initial = {
        "problem": spec.get("title", "") + "\n" + "\n".join(spec.get("questions", [])),
        "background": spec.get("background", ""),
        "questions": spec.get("questions", []),
        "stage_target": "basic",
        "iteration": 0,
        "output_dir": str(out),
    }
    g = build_graph()
    final = g.invoke(initial)
    typer.echo(f"done. paper at {out / 'paper.md'}")
    typer.echo(f"models: {[m.stage for m in final.get('model_versions', [])]}")
    typer.echo(f"iterations: {final.get('iteration')}")


if __name__ == "__main__":
    app()
