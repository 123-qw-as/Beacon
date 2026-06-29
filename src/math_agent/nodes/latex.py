"""latex 节点：渲染 .tex → 编译 .pdf → 失败时回退到 Markdown。"""
from __future__ import annotations

import re
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

from math_agent.nodes.writer import render_markdown
from math_agent.state import FigureArtifact, MathModelingState, PaperSections, SensitivityRun
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


# === Plan 2: writer 输出的 markdown/unicode → LaTeX 兼容 ===

# 孤立 unicode 数学字符 → LaTeX 命令。覆盖国赛论文最常出现的 ~50 个符号。
_UNICODE_MATH_MAP = {
    # 希腊字母（小写）
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    # 希腊字母（大写常用）
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
    # 关系/算术
    "≥": r"\geq", "≤": r"\leq", "≠": r"\neq", "≈": r"\approx",
    "±": r"\pm", "∓": r"\mp", "×": r"\times", "÷": r"\div", "·": r"\cdot",
    # 集合/逻辑
    "∈": r"\in", "∉": r"\notin", "⊂": r"\subset", "⊆": r"\subseteq",
    "∪": r"\cup", "∩": r"\cap", "∅": r"\emptyset",
    "∀": r"\forall", "∃": r"\exists",
    # 求和/极限
    "∑": r"\sum", "∏": r"\prod", "∫": r"\int",
    "∞": r"\infty", "∂": r"\partial", "∇": r"\nabla",
    "→": r"\to", "←": r"\leftarrow", "↔": r"\leftrightarrow",
}


def _wrap_unicode_math(s: str) -> str:
    """把字符串中孤立的 unicode 数学字符替换为 `$\\cmd$`。

    跳过已在 `$...$` 内的部分，避免双重包裹。**只处理 _UNICODE_MATH_MAP 中的字符**；
    不识别 `s_i^0` 这种裸 LaTeX 下标——那是 writer prompt 的责任。
    """
    if not s:
        return s
    # 按 $ 切分：偶数下标在 $...$ 外，奇数下标在内
    parts = s.split("$")
    for i in range(0, len(parts), 2):
        for ch, cmd in _UNICODE_MATH_MAP.items():
            if ch in parts[i]:
                parts[i] = parts[i].replace(ch, f"${cmd}$")
    return "$".join(parts)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_HEADING_LEVELS = {
    1: r"\section*",     2: r"\subsection*",  3: r"\subsubsection*",
    4: r"\paragraph",    5: r"\subparagraph", 6: r"\subparagraph",
}


def _md_headings_to_latex(s: str) -> str:
    """把行首 `### xxx` 形式的 markdown 标题转成 `\\subsubsection*{xxx}` 等。

    用星号版（不进目录）避免重复编号。仅匹配 BOL + 1-6 个 # + 空格 + 内容；
    行内 `#1`、注释 `# foo` 不动。
    """
    if not s:
        return s

    def _sub(m: re.Match) -> str:
        level = len(m.group(1))
        cmd = _HEADING_LEVELS[level]
        return f"{cmd}{{{m.group(2)}}}"

    return _HEADING_RE.sub(_sub, s)


def _prepare_section(s: str) -> str:
    """paper 段渲染前预处理：markdown 标题 → section + unicode 数学 → inline math。

    顺序敏感：先转标题（再 wrap unicode 不会破坏 \\subsubsection*{}），
    再 wrap unicode（最后才把 $ 引入文本）。**不调 _latex_escape**——writer
    在 paper 段会故意写 LaTeX inline math 和 markdown 表格，escape 会破坏。
    """
    return _wrap_unicode_math(_md_headings_to_latex(s))


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

    # paper 各段做 markdown 标题 + unicode 数学的确定性预处理
    safe_paper = PaperSections(**{
        k: _prepare_section(v) if isinstance(v, str) else v
        for k, v in state.paper.model_dump().items()
    })
    safe_sens = [
        SensitivityRun(
            parameter=r.parameter, values=r.values, metric=r.metric,
            results=r.results, interpretation=_prepare_section(r.interpretation),
            figure_path=r.figure_path,
        )
        for r in state.sensitivity_runs
    ]

    tmpl = _env.get_template("paper.tex.j2")
    tex = tmpl.render(
        problem=_latex_escape(_wrap_unicode_math(state.problem)),
        paper=safe_paper, figures=safe_figures, sensitivity_runs=safe_sens,
    )
    tex_path = workdir / "paper.tex"
    tex_path.write_text(tex, encoding="utf-8")

    # 始终也写一份 Markdown，作为降级 / 备查
    (workdir / "paper.md").write_text(render_markdown(state), encoding="utf-8")

    res = compile_latex(tex_path)
    if not res.success:
        return {"errors": [f"latex compile failed: {res.log[:500]}"]}
    return {}
