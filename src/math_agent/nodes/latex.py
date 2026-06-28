"""latex 节点：渲染 .tex → 编译 .pdf → 失败时回退到 Markdown。"""
from __future__ import annotations

from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

from math_agent.nodes.writer import render_markdown
from math_agent.state import MathModelingState
from math_agent.tools.latex_compile import compile_latex


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=select_autoescape([]))


def latex_node(state: MathModelingState) -> dict:
    workdir = Path(state.output_dir or ".")
    workdir.mkdir(parents=True, exist_ok=True)

    tmpl = _env.get_template("paper.tex.j2")
    tex = tmpl.render(
        problem=state.problem, paper=state.paper,
        figures=state.figures, sensitivity_runs=state.sensitivity_runs,
    )
    tex_path = workdir / "paper.tex"
    tex_path.write_text(tex, encoding="utf-8")

    # 始终也写一份 Markdown，作为降级 / 备查
    (workdir / "paper.md").write_text(render_markdown(state), encoding="utf-8")

    res = compile_latex(tex_path)
    if not res.success:
        return {"errors": [f"latex compile failed: {res.log[:500]}"]}
    return {}
