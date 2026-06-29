"""latex 节点：渲染 .tex → 编译 .pdf → 失败时回退到 Markdown。"""
from __future__ import annotations

from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

from math_agent.nodes.writer import render_markdown
from math_agent.state import FigureArtifact, MathModelingState
from math_agent.tools.latex_compile import compile_latex


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=select_autoescape([]))


def _latex_escape(s: str) -> str:
    """转义 LaTeX 文本类高危字符。仅作用于 caption / 图说 / problem 标题等纯文本。

    保留 _/\\/{/} 让 writer 写的 inline 数学（`s_i^0`、`\\sigma`）能正常渲染。
    只处理 % # & $——其他高危符号 writer 极少在纯文本里裸用。
    """
    return s.replace("%", r"\%") \
        .replace("#", r"\#") \
        .replace("&", r"\&") \
        .replace("$", r"\$")


def _latex_path(p: str) -> str:
    """把 Windows 路径包成 LaTeX 可读形式：正斜杠 + \\detokenize 阻止解释 _ 等。"""
    return r"\detokenize{" + p.replace("\\", "/") + "}"


def latex_node(state: MathModelingState) -> dict:
    workdir = Path(state.output_dir or ".")
    workdir.mkdir(parents=True, exist_ok=True)

    # ponytail: 为图重打一份 LaTeX 安全视图，避免 caption/path 里的 _/&/% 炸编译
    safe_figures = [
        FigureArtifact(
            path=_latex_path(f.path),
            purpose=_latex_escape(f.purpose),
            caption=_latex_escape(f.caption or f.purpose),
            quality_score=f.quality_score,
            quality_issues=list(f.quality_issues),
            analysis=_latex_escape(f.analysis),
        )
        for f in state.figures
    ]

    tmpl = _env.get_template("paper.tex.j2")
    tex = tmpl.render(
        problem=_latex_escape(state.problem), paper=state.paper,
        figures=safe_figures, sensitivity_runs=state.sensitivity_runs,
    )
    tex_path = workdir / "paper.tex"
    tex_path.write_text(tex, encoding="utf-8")

    # 始终也写一份 Markdown，作为降级 / 备查
    (workdir / "paper.md").write_text(render_markdown(state), encoding="utf-8")

    res = compile_latex(tex_path)
    if not res.success:
        return {"errors": [f"latex compile failed: {res.log[:500]}"]}
    return {}
