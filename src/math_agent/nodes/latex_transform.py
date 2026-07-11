"""Markdown → LaTeX 纯文本转换链。

所有函数都是确定性的字符串到字符串变换，不依赖 Jinja2 模板或 LangGraph 状态。
由 latex_node.py 的 latex_node() 调用 _prepare_section 编排链来驱动。

核心工具：
  split_text_math(text)        — 按 $ 切分为 (text, math) 交替段（A 类保护）
  split_with_display_math(text) — 先保护 display math 块，再在文本段内按 $ 切（B 类保护）
"""
from __future__ import annotations

import re

from math_agent.nodes.rendering import _latex_plain_text


# ============================================================
# 切分工具函数
# ============================================================

def split_text_math(text: str) -> list[tuple[str, str | None]]:
    """按 ``$`` 切分为 (text_outside, math_inside_or_None) 交替段。

    返回列表：
      [(text₀, math₁), (text₂, math₃), ..., (textₙ, None)]

    - 偶数段（text）在 ``$...$`` 之外
    - 奇数段（math）在 ``$...$`` 之内，含两侧 ``$`` 分隔符
    - 末尾段 math 部分为 None

    用法::

        segs = split_text_math(s)
        for i, (text, math) in enumerate(segs):
            segs[i] = (transform(text), math)
        return "".join(t + (m or "") for t, m in segs)
    """
    parts = text.split("$")
    result: list[tuple[str, str | None]] = []
    for i in range(0, len(parts), 2):
        text_seg = parts[i]
        if i + 1 < len(parts):
            math_seg = "$" + parts[i + 1] + "$"
        else:
            math_seg = None
        result.append((text_seg, math_seg))
    return result


_DISPLAY_MATH_RE = re.compile(
    r"(\\\[.*?\\\]|\\\(.*?\\\)|\\begin\{equation\*?\}.*?\\end\{equation\*?\})",
    re.DOTALL,
)


def split_with_display_math(text: str) -> list[tuple[str, str | None]]:
    """两层切分：先保护 display math 块，再在文本段内按 ``$`` 切分。

    返回格式同 :func:`split_text_math`。display math 块
    （``\\[...\\]``、``\\(...\\)``、``\\begin{equation}...\\end{equation}``）
    被视为 math 段，内部不再按 ``$`` 二次切分。

    适用于 _pad_math_commands、_wrap_naked_subscripts 等需要同时
    保护 display math 和 inline math 的场景。
    """
    display_parts = _DISPLAY_MATH_RE.split(text)
    result: list[tuple[str, str | None]] = []
    for k, part in enumerate(display_parts):
        is_display = k % 2 == 1  # split 奇数索引 = display math 块
        if is_display:
            result.append(("", part))
        else:
            result.extend(split_text_math(part))
    return result


# ============================================================
# Unicode 数学映射
# ============================================================

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
    # 上标 unicode
    "²": r"^2", "³": r"^3", "¹": r"^1", "⁰": r"^0",
}


def _wrap_unicode_math(s: str) -> str:
    """把字符串中孤立的 unicode 数学字符替换为 `$\\cmd$`。

    跳过已在 `$...$` 内的部分。连续相邻的 unicode 数学字符（如 `σ²`）会
    合并到同一个 inline math span。
    """
    if not s:
        return s
    sub_re = re.compile(r"([_^](?:\{[^}]+\}|[A-Za-z0-9]+))+")

    def _process(seg: str) -> str:
        out: list[str] = []
        j = 0
        while j < len(seg):
            ch = seg[j]
            if ch not in _UNICODE_MATH_MAP:
                out.append(ch)
                j += 1
                continue
            # 收集连续 unicode 数学符 run
            run_end = j
            while run_end < len(seg) and seg[run_end] in _UNICODE_MATH_MAP:
                run_end += 1
                while run_end < len(seg) and seg[run_end] in ("_", "^"):
                    sm = sub_re.match(seg, run_end)
                    if sm and sm.start() == run_end:
                        run_end = sm.end()
                    else:
                        run_end += 1
            token = seg[j:run_end]
            converted: list[str] = []
            pos = 0
            while pos < len(token):
                c = token[pos]
                cmd = _UNICODE_MATH_MAP.get(c)
                if cmd is not None:
                    converted.append(cmd)
                    pos += 1
                    sm = sub_re.match(token, pos)
                    if sm and sm.start() == pos:
                        converted.append(sm.group(0))
                        pos = sm.end()
                else:
                    converted.append(c)
                    pos += 1
            out.append("$" + "".join(converted) + "$")
            j = run_end
        return "".join(out)

    segs = split_text_math(s)
    for i, (text, math) in enumerate(segs):
        segs[i] = (_process(text), math)
    return "".join(t + (m or "") for t, m in segs)


# ============================================================
# Math 命令间距修复
# ============================================================

_KNOWN_MATH_CMDS = frozenset({
    # 二元关系
    "leq", "geq", "neq", "leqslant", "geqslant", "approx", "sim", "propto",
    "equiv", "in", "notin", "subset", "supset", "cup", "cap",
    "subseteq", "supseteq", "subsetneq", "supsetneq", "nsubseteq", "nsupseteq",
    "subsetneqq", "supsetneqq",
    # 二元/一元算子
    "cdot", "cdotp", "times", "div", "pm", "mp", "ast", "star", "circ",
    "ll", "gg", "lll", "ggg",
    # 大运算符
    "sum", "prod", "int", "iint", "iiint", "oint", "bigcup", "bigcap",
    "lim", "sup", "inf", "max", "min", "arg",
    # 希腊字母（小写）
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta",
    "eta", "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu",
    "xi", "omicron", "pi", "varpi", "rho", "varrho", "sigma", "varsigma",
    "tau", "upsilon", "phi", "varphi", "chi", "psi", "omega",
    # 希腊字母（大写）
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon",
    "Phi", "Psi", "Omega",
    # 箭头
    "to", "leftarrow", "rightarrow", "leftrightarrow",
    "Leftarrow", "Rightarrow", "Leftrightarrow",
    "longrightarrow", "Longrightarrow", "longleftarrow", "Longleftarrow",
    "mapsto", "hookrightarrow", "hookleftarrow",
    # 常用符号
    "infty", "partial", "nabla", "forall", "exists", "emptyset", "hbar",
    "ell", "Re", "Im", "aleph", "cdots", "ldots", "vdots", "ddots",
    "square", "blacksquare", "qed",
    # 字体/装饰
    "hat", "bar", "tilde", "vec", "dot", "ddot", "overline", "underline",
    "widehat", "widetilde", "mathbb", "mathbf", "mathcal", "mathrm",
    "boldsymbol", "text", "textrm", "textbf", "textit",
    # 分式/根号
    "frac", "sqrt", "binom", "dfrac", "tfrac",
    # 左右括号
    "left", "right", "bigl", "bigr", "Bigl", "Bigr", "big", "Big",
    # 常用函数名
    "sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp",
    "sinh", "cosh", "tanh", "det", "gcd", "deg", "dim", "ker",
})

_MATH_CMD_WORD_RE = re.compile(r"\\([A-Za-z]+)")


def _split_known_prefix(word: str) -> str | None:
    """返回 word 的最长已知命令前缀（不含尾部残余），若不存在返回 None。"""
    for n in range(len(word) - 1, 0, -1):
        if word[:n] in _KNOWN_MATH_CMDS:
            return word[:n]
    return None


def _pad_math_commands(s: str) -> str:
    """在 math span 和 equation 块内，把 `\\<known-cmd><letters>`
    拆为 `\\<known-cmd>\\,<letters>`。"""
    if not s:
        return s

    def _sub(m: re.Match) -> str:
        word = m.group(1)
        if word in _KNOWN_MATH_CMDS:
            return m.group(0)
        prefix = _split_known_prefix(word)
        if prefix is None:
            return m.group(0)
        rest = word[len(prefix):]
        return f"\\{prefix}\\,{rest}"

    segs = split_with_display_math(s)
    for i, (text, math) in enumerate(segs):
        if math is not None:
            segs[i] = (text, _MATH_CMD_WORD_RE.sub(_sub, math))
    return "".join(t + (m or "") for t, m in segs)


# ============================================================
# Markdown 排版 → LaTeX
# ============================================================

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_HEADING_LEVELS = {
    1: r"\section", 2: r"\subsection", 3: r"\subsubsection",
    4: r"\paragraph", 5: r"\subparagraph", 6: r"\subparagraph",
}


def _md_headings_to_latex(s: str) -> str:
    """把行首 `### xxx` 形式的 markdown 标题转成 `\\subsubsection*{xxx}` 等。"""
    if not s:
        return s

    def _sub(m: re.Match) -> str:
        level = len(m.group(1))
        cmd = _HEADING_LEVELS[level]
        return f"{cmd}{{{m.group(2)}}}"

    return _HEADING_RE.sub(_sub, s)


_BACKTICK_RE = re.compile(r"`([^`\n]+?)`")
_UNICODE_MATH_CHARS = "".join(_UNICODE_MATH_MAP.keys())
_NAKED_SUB_RE = re.compile(
    r"(?<![\\$\w{])"
    r"([A-Za-z" + re.escape(_UNICODE_MATH_CHARS) + r"][A-Za-z0-9]*"
    r"(?:_(?:\{[^}]+\}|[A-Za-z0-9]+))"
    r"(?:\^(?:\{[^}]+\}|[A-Za-z0-9]+))?"
    r")"
    r"(?![\w./])"
)


def _md_inline_code_to_math(s: str) -> str:
    """把 markdown 反引号 inline `code` 转成 `$code$`。"""
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
    """把行内裸的 LaTeX 下标/上标自动包成 $...$。"""
    if not s:
        return s

    def _sub(m: re.Match) -> str:
        content = m.group(1)
        for ch, cmd in _UNICODE_MATH_MAP.items():
            if ch in content:
                content = content.replace(ch, cmd)
        return f"${content}$"

    def _process_text(t: str) -> str:
        segs = split_text_math(t)
        for i, (text, math) in enumerate(segs):
            segs[i] = (_NAKED_SUB_RE.sub(_sub, text), math)
        return "".join(t + (m or "") for t, m in segs)

    segs = split_with_display_math(s)
    for i, (text, math) in enumerate(segs):
        if math is None:
            segs[i] = (_process_text(text), None)
        elif math.startswith("\\[") or math.startswith("\\(") or math.startswith("\\begin{equation"):
            # display math 块：text 部分需要处理（来自 split_with_display_math 的文本段）
            pass  # text 已经是文本段，math 是 display 块，不动
        else:
            segs[i] = (_process_text(text), math)
    # Re-join: split_with_display_math 返回的 display math 在 math 位置，
    # text 部分的 text 需要再按 $ 切处理
    # 实际上 split_with_display_math 已经帮我们做了两层切分，
    # 所以这里的 text 都是纯文本段（不含 display math），math 要么是 inline 要么是 display
    result_segs: list[str] = []
    for text, math in segs:
        if math is None:
            result_segs.append(_process_text(text))
        elif math.startswith("\\[") or math.startswith("\\(") or math.startswith("\\begin{equation"):
            # display math 块：text 需要处理
            result_segs.append(_process_text(text))
            result_segs.append(math)
        else:
            # inline math: text 需要处理，math 不动
            result_segs.append(_process_text(text))
            result_segs.append(math)
    return "".join(result_segs)


_BOLD_RE = re.compile(r"\*\*(\S(?:[^\*\n]*?\S)?)\*\*")


def _md_bold_to_latex(s: str) -> str:
    """**xxx** → \\textbf{xxx}。"""
    if not s:
        return s
    return _BOLD_RE.sub(r"\\textbf{\1}", s)


_BULLET_RE = re.compile(r"^[ \t]*[-*+]\s+(.+)$", re.MULTILINE)


def _md_bullets_to_latex(s: str) -> str:
    """连续的 `- xxx` 行 → \\begin{itemize} ... \\end{itemize}。"""
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
    """markdown pipe-table → LaTeX tabular。"""
    def _escape_cell_amps(cell: str) -> str:
        r"""转义 cell 里的裸 &，但跳过 $...$ math 模内的和已转义的 \&。"""
        segs = split_text_math(cell)
        for i, (text, math) in enumerate(segs):
            segs[i] = (re.sub(r"(?<!\\)&", r"\&", text), math)
        return "".join(t + (m or "") for t, m in segs)

    if not s or "|" not in s:
        return s
    lines = s.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("|") and i + 1 < len(lines):
            sep = lines[i + 1].strip()
            if sep.startswith("|") and set(sep) <= set("|-: "):
                header_cells = [c.strip() for c in line.strip().strip("|").split("|")]
                header_cells = [_escape_cell_amps(c) for c in header_cells]
                ncols = len(header_cells)
                sep_cells = [c.strip() for c in sep.strip().strip("|").split("|")]
                has_align_colons = any(":" in sc for sc in sep_cells)
                if has_align_colons:
                    col_spec = "".join(
                        "c" if sc.startswith(":") and sc.endswith(":") else
                        "r" if sc.endswith(":") else
                        "l" if sc.startswith(":") else "l"
                        for sc in sep_cells
                    )
                else:
                    col_spec = "X" * ncols
                col_spec = (col_spec + "X" * ncols)[:ncols]
                tbl = [r"\begin{tabularx}{\linewidth}{" + col_spec + r"}",
                       r"\toprule",
                       " & ".join(header_cells) + r" \\",
                       r"\midrule"]
                j = i + 2
                while j < len(lines) and lines[j].lstrip().startswith("|"):
                    cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                    cells = [_escape_cell_amps(c) for c in cells]
                    cells = (cells + [""] * ncols)[:ncols]
                    tbl.append(" & ".join(cells) + r" \\")
                    j += 1
                tbl.append(r"\bottomrule")
                tbl.append(r"\end{tabularx}")
                out.append("")
                out.extend(tbl)
                out.append("")
                i = j
                continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _escape_text_chars_skip_math(content: str, chars: str = "_%&#") -> str:
    r"""转义 text-mode 特殊字符，但跳过 $...$ math 段。"""
    segs = split_text_math(content)
    for i, (text, math) in enumerate(segs):
        for ch in chars:
            text = re.sub(rf"(?<!\\){re.escape(ch)}", rf"\{ch}", text)
        segs[i] = (text, math)
    return "".join(t + (m or "") for t, m in segs)


def _escape_remaining_underscores(s: str) -> str:
    r"""对 $...$ 之外、math/tabularx 环境之外的 text-mode 特殊字符做转义。"""
    if not s:
        return s

    protected: list[tuple[int, int, str]] = []

    _env_re = re.compile(
        r"\\begin\{(equation|align|gather|multline|split|aligned|gathered|cases|"
        r"matrix|pmatrix|bmatrix|vmatrix|smallmatrix|array|subarray|tabularx)"
        r"\*?\}.*?\\end\{\1\*?\}",
        re.DOTALL,
    )
    for m in _env_re.finditer(s):
        content = m.group(0)
        env_name = m.group(1)
        if env_name == "tabularx":
            content = _escape_text_chars_skip_math(content, chars="_%#")
        protected.append((m.start(), m.end(), content))

    _display_re = re.compile(r"\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)
    for m in _display_re.finditer(s):
        if not any(p[0] <= m.start() < p[1] for p in protected):
            protected.append((m.start(), m.end(), m.group(0)))

    protected.sort()
    merged: list[tuple[int, int, str]] = []
    for start, end, content in protected:
        if merged and start < merged[-1][1]:
            prev_start, prev_end, prev_content = merged.pop()
            merged.append((prev_start, max(prev_end, end), prev_content + content[prev_end - start:]))
        else:
            merged.append((start, end, content))

    def _escape_text_segment(text: str) -> str:
        segs = split_text_math(text)
        for i, (t, m) in enumerate(segs):
            t = re.sub(r"(?<!\\)_", r"\_", t)
            t = re.sub(r"(?<!\\)%", r"\%", t)
            t = re.sub(r"(?<!\\)&", r"\&", t)
            t = re.sub(r"(?<!\\)#", r"\#", t)
            segs[i] = (t, m)
        return "".join(t + (m or "") for t, m in segs)

    result: list[str] = []
    pos = 0
    for start, end, content in merged:
        if pos < start:
            result.append(_escape_text_segment(s[pos:start]))
        result.append(content)
        pos = end

    if pos < len(s):
        result.append(_escape_text_segment(s[pos:]))

    return "".join(result)


# ============================================================
# Equation 提升
# ============================================================

_LONG_MATH_INDICATORS = ("=", r"\leq", r"\geq", r"\sum", r"\prod", r"\int", r"\le ", r"\ge ", "\\le}", "\\ge}")

_TABLE_SPLIT_RE = re.compile(
    r"(\\begin\{tabularx\}.*?\\end\{tabularx\})",
    re.DOTALL,
)


def _is_long_equation(content: str) -> bool:
    if len(content) < 10:
        return False
    return any(ind in content for ind in _LONG_MATH_INDICATORS)


def _promote_inline_equations(s: str) -> str:
    r"""把段内"独立式"长公式从 `$...$` 提升为 `\begin{equation}...\end{equation}`。"""
    if not s or "$" not in s:
        return s

    segments = _TABLE_SPLIT_RE.split(s)
    result = []
    for seg_idx, seg in enumerate(segments):
        if seg_idx % 2 == 1:
            result.append(seg)
            continue
        result.append(_promote_inline_in_text(seg))
    return "".join(result)


def _promote_inline_in_text(s: str) -> str:
    """对不含 tabularx 表格的文本做 inline math → equation 提升。"""
    out = []
    i = 0
    while i < len(s):
        if s[i] == "$":
            end = s.find("$", i + 1)
            if end == -1:
                out.append(s[i:])
                break
            content = s[i + 1:end]
            after = s[end + 1:end + 2]
            before = s[max(0, i - 1):i]
            sep_chars = set("。，；：、 \n\t")
            before_ok = before == "" or before in sep_chars or before in "。，；：、"
            after_ok = after == "" or after in sep_chars or after in "。，；：、"
            if _is_long_equation(content) and before_ok and after_ok:
                eat = end + 1
                while eat < len(s) and s[eat] in "。，；：、 \t":
                    eat += 1
                out.append("\n\\begin{equation}\n" + content + "\n\\end{equation}\n")
                i = eat
                continue
            out.append(s[i:end + 1])
            i = end + 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


# ============================================================
# GMCM 参考文献
# ============================================================

_REFERENCE_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s*(.+?)\s*$")


def _gmcm_bibliography(s: str) -> str:
    r"""把 writer 的 ``[n] 文献`` 行转换为 thebibliography 所需的 ``\bibitem``。"""
    items: list[str] = []
    auto_index = 1
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _REFERENCE_LINE_RE.match(line)
        if match:
            key, content = match.groups()
        else:
            key, content = str(auto_index), line
        items.append(f"\\bibitem{{ref{key}}} {content}")
        auto_index += 1
    return "\n".join(items)


# ============================================================
# 转换链编排
# ============================================================

def _prepare_section(s: str) -> str:
    """paper 段渲染前预处理：链式确定性转换，全部跳过已有 $...$。

    顺序敏感：
    1. markdown 表格 → tabular
    2. markdown 标题 → LaTeX 标题命令
    3. markdown 加粗 → \\textbf
    4. markdown 列表 → itemize
    5. markdown 反引号 → inline math
    6. 裸 LaTeX 下标 → $...$
    7. unicode 数学符号 → $\\cmd$
    8. 长 inline equation → display equation
    9. math 命令间距修复
    10. 残留特殊字符转义
    """
    s = _md_table_to_latex(s)
    s = _md_headings_to_latex(s)
    s = _md_bold_to_latex(s)
    s = _md_bullets_to_latex(s)
    s = _md_inline_code_to_math(s)
    s = _wrap_naked_subscripts(s)
    s = _wrap_unicode_math(s)
    s = _promote_inline_equations(s)
    s = _pad_math_commands(s)
    s = _escape_remaining_underscores(s)
    return s


def _prepare_inline_text(s: str) -> str:
    """处理标题/题注等单行参数，不生成 equation、列表或章节命令。"""
    s = _md_inline_code_to_math(s)
    s = _wrap_naked_subscripts(s)
    s = _wrap_unicode_math(s)
    s = _pad_math_commands(s)
    segs = split_text_math(s)
    for i, (text, math) in enumerate(segs):
        segs[i] = (_latex_plain_text(text) or "", math)
    return "".join(t + (m or "") for t, m in segs)


def _prepare_title(s: str) -> str:
    """转义标题文本，同时保留已有及由 Unicode 符号生成的 ``$...$`` 数学段。"""
    s = _wrap_unicode_math(s)
    segs = split_text_math(s)
    for i, (text, math) in enumerate(segs):
        segs[i] = (_latex_plain_text(text) or "", math)
    return "".join(t + (m or "") for t, m in segs)
