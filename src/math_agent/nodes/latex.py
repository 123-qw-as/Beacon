"""latex 节点：渲染 .tex → 编译 .pdf → 失败时回退到 Markdown。"""
from __future__ import annotations

import re
import shutil
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

    跳过已在 `$...$` 内的部分。如果 unicode 后面紧跟 `_xxx` 或 `^xxx`
    （下标/上标 token），把整体作为一个 inline math 包起来——避免
    `α_i` 被切成 `$\\alpha$_i` 导致 `_i` 又进 text mode 触发错误。
    """
    if not s:
        return s
    parts = s.split("$")
    sub_re = re.compile(r"([_^](?:\{[^}]+\}|[A-Za-z0-9]+))+")
    for i in range(0, len(parts), 2):
        out = []
        j = 0
        seg = parts[i]
        while j < len(seg):
            ch = seg[j]
            cmd = _UNICODE_MATH_MAP.get(ch)
            if cmd is None:
                out.append(ch)
                j += 1
                continue
            # 看后面是否紧跟下标/上标（re.match(pos) 仍从字符串绝对开头匹配 ^，所以去掉 ^ 并 anchor 在 pos）
            m = sub_re.match(seg, j + 1)
            if m and m.start() == j + 1:
                out.append(f"${cmd}{m.group(0)}$")
                j = m.end()
            else:
                out.append(f"${cmd}$")
                j += 1
        parts[i] = "".join(out)
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


_BACKTICK_RE = re.compile(r"`([^`\n]+?)`")
# 裸 LaTeX 下标/上标：[字母][后续字母数字]*([_^](identifier|{...}))+
# 例：D_i / c_{ij} / S_i^{(1)} / x_t^k；前面不接 \（命令名）或 $（已在 math 内）
_NAKED_SUB_RE = re.compile(
    r"(?<![\\$])"
    r"([A-Za-z][A-Za-z0-9]*"
    r"(?:[_^](?:\{[^}]+\}|[A-Za-z0-9]+))+)"
)


def _md_inline_code_to_math(s: str) -> str:
    """把 markdown 反引号 inline `code` 转成 `$code$`。

    writer 在符号说明表里习惯写 `` `S_i` `` 这种 markdown 内联代码（v6.2 实测）。
    LaTeX 不认反引号；转为 inline math 让 `S_i` 渲染为 S_i 的数学下标。
    转换时把内容里的 unicode 数学字符也展开为 LaTeX 命令（α → \\alpha），
    避免外层 _wrap_unicode_math 跳过 $...$ 内导致 α 仍裸用。
    """
    if not s:
        return s

    def _sub(m: re.Match) -> str:
        content = m.group(1)
        for ch, cmd in _UNICODE_MATH_MAP.items():
            if ch in content:
                content = content.replace(ch, cmd)
        return f"${content}$"

    return _BACKTICK_RE.sub(_sub, s)


def _wrap_naked_subscripts(s: str) -> str:
    """把行内裸的 LaTeX 下标/上标自动包成 $...$。

    覆盖 writer RULE 4 没治住的 `D_i`、`c_{ij}`、`S_i^{(1)}`。
    跳过已在 $...$ 内、跳过 \\command 前缀（避免动 \\paragraph{w\\_RF}）。

    已知 over-wrap 风险：`a_b_c` 这种合法变量名也会被当数学。但 paper 段里
    几乎不会出现，且即便错伤，渲出来仍是合法 LaTeX（math italic 显示）。
    """
    if not s:
        return s
    parts = s.split("$")
    for i in range(0, len(parts), 2):
        parts[i] = _NAKED_SUB_RE.sub(r"$\1$", parts[i])
    return "$".join(parts)


# ===== Markdown 排版 → LaTeX 命令 =====

# 加粗内容不能以空白开头/结尾（避免 "普通 ** 段; **" 被当作一对）
_BOLD_RE = re.compile(r"\*\*(\S(?:[^\*\n]*?\S)?)\*\*")


def _md_bold_to_latex(s: str) -> str:
    """**xxx** → \\textbf{xxx}。LaTeX 不认 markdown 加粗。"""
    if not s:
        return s
    return _BOLD_RE.sub(r"\\textbf{\1}", s)


_BULLET_RE = re.compile(r"^[ \t]*[-*+]\s+(.+)$", re.MULTILINE)


def _md_bullets_to_latex(s: str) -> str:
    """连续的 `- xxx` 行 → \\begin{itemize} ... \\end{itemize}。

    简单做法：扫每一行，识别 bullet 行；连续 bullet 段开头加 begin、结束加 end。
    """
    if not s or "\n" not in s and not s.startswith(("-", "*", "+")):
        return s
    lines = s.split("\n")
    out = []
    in_list = False
    for line in lines:
        m = _BULLET_RE.match(line)
        if m:
            if not in_list:
                out.append(r"\begin{itemize}")
                in_list = True
            out.append(r"\item " + m.group(1))
        else:
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append(line)
    if in_list:
        out.append(r"\end{itemize}")
    return "\n".join(out)


def _md_table_to_latex(s: str) -> str:
    """markdown pipe-table → LaTeX tabular。

    识别连续行：
        | h1 | h2 |
        |----|----|
        | a  | b  |

    转成：
        \\begin{tabular}{|l|l|}\\hline h1 & h2 \\\\\\hline a & b \\\\\\hline \\end{tabular}

    分隔行 (`---`) 用来确定列数；之前一行作 header。
    不在表内的 `|` 不动。
    """
    if not s or "|" not in s:
        return s
    lines = s.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # 检测表头模式：当前行 | 开头 + 至少一个 | + 下一行是分隔行
        if line.lstrip().startswith("|") and i + 1 < len(lines):
            sep = lines[i + 1].strip()
            # 分隔行只含 |、-、:、空格
            if sep.startswith("|") and set(sep) <= set("|-: "):
                header_cells = [c.strip() for c in line.strip().strip("|").split("|")]
                ncols = len(header_cells)
                col_spec = "|" + "l|" * ncols
                tbl = [r"\begin{tabular}{" + col_spec + r"}",
                       r"\hline",
                       " & ".join(header_cells) + r" \\",
                       r"\hline"]
                j = i + 2
                while j < len(lines) and lines[j].lstrip().startswith("|"):
                    cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                    # 对齐列数（缺则补空，多则截）
                    cells = (cells + [""] * ncols)[:ncols]
                    tbl.append(" & ".join(cells) + r" \\")
                    tbl.append(r"\hline")
                    j += 1
                tbl.append(r"\end{tabular}")
                # 表格前后留空行让 LaTeX 不把表格挤进段落
                out.append("")
                out.extend(tbl)
                out.append("")
                i = j
                continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _prepare_section(s: str) -> str:
    """paper 段渲染前预处理：链式确定性转换，全部跳过已有 $...$。

    顺序敏感：
    1. markdown 表格 → tabular（在 inline code/heading 之前；表格里可能有 backtick 数学）
    2. markdown 标题 `### xxx` → `\\subsubsection*{xxx}`
    3. markdown 加粗 `**x**` → `\\textbf{x}`
    4. markdown 列表 `- x` → itemize
    5. markdown 反引号 `` `S_i` `` → `$S_i$`（内容里 unicode 同步展开）
    6. 裸 LaTeX 下标 `D_i` → `$D_i$`（writer 漏的兜底）
    7. unicode 数学符号 α → `$\\alpha$`（在 $...$ 之外的）

    **不调 _latex_escape**——writer 在 paper 段会故意写 LaTeX inline math。
    """
    s = _md_table_to_latex(s)
    s = _md_headings_to_latex(s)
    s = _md_bold_to_latex(s)
    s = _md_bullets_to_latex(s)
    s = _md_inline_code_to_math(s)
    s = _wrap_naked_subscripts(s)
    s = _wrap_unicode_math(s)
    return s


def latex_node(state: MathModelingState) -> dict:
    workdir = Path(state.output_dir or ".")
    workdir.mkdir(parents=True, exist_ok=True)

    # ponytail: 为图重打一份 LaTeX 安全视图，避免 caption/path 里的 _/&/% 炸编译
    # caption 限到约 55 字（包含完整结尾标点），analysis 给完整版用作正文段
    safe_figures = [
        FigureArtifact(
            path=_latex_path(f.path),
            purpose=_latex_escape(f.purpose),
            caption=_latex_escape((f.caption or f.purpose)[:55]),
            quality_score=f.quality_score,
            quality_issues=list(f.quality_issues),
            analysis=_latex_escape(f.analysis),
        )
        for f in state.figures
    ]

    # paper 各段做 markdown → LaTeX 的确定性预处理（粗体/表格/列表/标题/数学）
    safe_paper = PaperSections(**{
        k: _prepare_section(v) if isinstance(v, str) else v
        for k, v in state.paper.model_dump().items()
    })
    safe_sens = [
        SensitivityRun(
            parameter=r.parameter.replace("_", r"\_"),  # paragraph{} 是 text mode，_ 会切到 math
            values=r.values, metric=r.metric,
            results=r.results, interpretation=_prepare_section(r.interpretation),
            figure_path=r.figure_path,
        )
        for r in state.sensitivity_runs
    ]

    # title 取 problem 第一行（避免把整段问题描述塞进 \title{}）
    title_line = state.problem.split("\n", 1)[0].strip()

    # 选模板：default 用 article 简版；gmcm 用 gmcmthesis 国赛规范
    use_gmcm = state.latex_template == "gmcm"
    tmpl_name = "gmcm.tex.j2" if use_gmcm else "paper.tex.j2"

    render_kwargs = dict(
        problem=_latex_escape(_wrap_unicode_math(title_line)),
        paper=safe_paper, figures=safe_figures, sensitivity_runs=safe_sens,
    )
    if use_gmcm:
        # 队员逗号拆分到 a/b/c，空位补占位
        mem = (state.members or "").split(",")
        mem = [m.strip() for m in mem] + ["", "", ""]
        render_kwargs.update(
            school=state.school,
            team_id=state.team_id,
            member_a=mem[0] or None,
            member_b=mem[1] or None,
            member_c=mem[2] or None,
            keywords="数学建模, 多智能体, 优化",  # writer 未生成关键词；后续 plan 可加
            code_artifacts=[a for a in state.code_artifacts if a.success],
        )
        # cls 必须和 .tex 在同一目录才能被 xelatex 找到；封面图也要带上
        cls_src = _TEMPLATE_DIR / "gmcmthesis.cls"
        cls_dst = workdir / "gmcmthesis.cls"
        shutil.copyfile(cls_src, cls_dst)
        fig_src = _TEMPLATE_DIR / "gmcm_figures"
        fig_dst = workdir / "figures"
        if fig_src.is_dir():
            fig_dst.mkdir(parents=True, exist_ok=True)
            for f in fig_src.iterdir():
                shutil.copyfile(f, fig_dst / f.name)

    tmpl = _env.get_template(tmpl_name)
    tex = tmpl.render(**render_kwargs)
    tex_path = workdir / "paper.tex"
    tex_path.write_text(tex, encoding="utf-8")

    # 始终也写一份 Markdown，作为降级 / 备查
    (workdir / "paper.md").write_text(render_markdown(state), encoding="utf-8")

    res = compile_latex(tex_path)
    if not res.success:
        return {"errors": [f"latex compile failed: {res.log[:500]}"]}
    return {}
