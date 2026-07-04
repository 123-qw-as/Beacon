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
    # 上标 unicode
    "²": r"^2", "³": r"^3", "¹": r"^1", "⁰": r"^0",
}


def _wrap_unicode_math(s: str) -> str:
    """把字符串中孤立的 unicode 数学字符替换为 `$\\cmd$`。

    跳过已在 `$...$` 内的部分。连续相邻的 unicode 数学字符（如 `σ²`）会
    合并到同一个 inline math span（`$\\sigma^2$`），避免 `$\\sigma$$^2$`
    两个独立 span 导致排版断开。
    """
    if not s:
        return s
    parts = s.split("$")
    sub_re = re.compile(r"([_^](?:\{[^}]+\}|[A-Za-z0-9]+))+")

    for i in range(0, len(parts), 2):
        seg = parts[i]
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
                # 吃可选的 _xxx / ^xxx 后缀
                while run_end < len(seg) and seg[run_end] in ("_", "^"):
                    sm = sub_re.match(seg, run_end)
                    if sm and sm.start() == run_end:
                        run_end = sm.end()
                    else:
                        run_end += 1
            # 转换 run[j:run_end]
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
        parts[i] = "".join(out)
    return "$".join(parts)


# `\cmd` 后紧跟字母会被 xelatex 认成一个更长的命令：
#   `\cdot dist_ij` → OK （空格分隔）
#   `\cdotdist_ij`  → \cdotdist 未定义控制序列，halt-on-error 直接停编译
# writer 从 markdown 抽 markdown-math 时经常把 `$\cdot$dist_{ij}$` 写成
# 挤在一起。在这些命令后自动补 `\,`（薄空格，不影响排版）。
#
# 只在 $...$ math span 内做，text 段的 `\textbf` 等命令外部由 latex 自身
# 边界规则（大括号/空白）判定。
#
# 关键：字符串 `\cdotp` 有两种合法解读——`\cdotp`（自身是命令）或
# `\cdot`+`p`（两截）。正则本身分不出，必须用命令白名单：
# - 完整词命中已知命令（如 `\cdotp`） → 不动
# - 完整词命中未知命令但**最长已知前缀存在** → 在前缀后插 `\,`
# - 完整词命中未知命令且无已知前缀 → 不动（写手自己的锅）
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

# 命令名词：`\` + 一段字母（LaTeX 词法完全等价）
_MATH_CMD_WORD_RE = re.compile(r"\\([A-Za-z]+)")


def _split_known_prefix(word: str) -> str | None:
    """返回 word 的最长已知命令前缀（不含尾部残余），若不存在返回 None。"""
    for n in range(len(word) - 1, 0, -1):  # 前缀长度从长到短，且必须留至少 1 字符尾部
        if word[:n] in _KNOWN_MATH_CMDS:
            return word[:n]
    return None


def _pad_math_commands(s: str) -> str:
    """在 math span（`$...$`）和 equation 块内，把 `\\<known-cmd><letters>`
    拆为 `\\<known-cmd>\\,<letters>`。

    只处理"最长已知前缀 + 未知尾巴"的情形；完整词命中已知命令的（如 `\\cdotp`）
    不动；完整词非已知命令也无已知前缀的（如 `\\myVar`）也不动，避免误伤写手
    自定义宏。text 段（`$` 之外、equation 块之外）完全跳过。
    """
    if not s:
        return s

    def _sub(m: re.Match) -> str:
        word = m.group(1)
        if word in _KNOWN_MATH_CMDS:
            return m.group(0)  # 完整词就是合法命令
        prefix = _split_known_prefix(word)
        if prefix is None:
            return m.group(0)  # 未知，不动
        rest = word[len(prefix):]
        return f"\\{prefix}\\,{rest}"

    # 先按 display/inline math 块切；块内直接处理，块外再按 $ 切处理 inline math span
    # 需要处理：\[...\]、\(...\)、\begin{equation}...\end{equation}
    math_re = re.compile(
        r"(\\\[.*?\\\]|\\\(.*?\\\)|\\begin\{equation\*?\}.*?\\end\{equation\*?\})",
        re.DOTALL,
    )
    outer = math_re.split(s)
    for k in range(len(outer)):
        # 属于 math 块（奇数索引）
        is_block = outer[k].startswith(r"\[") or outer[k].startswith(r"\(") or outer[k].startswith(r"\begin{equation")
        if is_block:
            outer[k] = _MATH_CMD_WORD_RE.sub(_sub, outer[k])
        else:
            parts = outer[k].split("$")
            for i in range(1, len(parts), 2):  # 奇数段 = math span
                parts[i] = _MATH_CMD_WORD_RE.sub(_sub, parts[i])
            outer[k] = "$".join(parts)
    return "".join(outer)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
# 带编号版本：让小节进目录、有编号（gmcm 模板顶层 \section 已经用；writer 用 ## 表示子节）
_HEADING_LEVELS = {
    1: r"\section",      2: r"\subsection",   3: r"\subsubsection",
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
# unicode 数学字符（_UNICODE_MATH_MAP 的 key），用于 _NAKED_SUB_RE 的 ident 起头集
_UNICODE_MATH_CHARS = "".join(_UNICODE_MATH_MAP.keys())
# 裸 LaTeX 下标/上标：[字母 or unicode 数学符][后续字母数字]*([_^](identifier|{...}))+
# 允许 _^ 交替（如 S_i^{(1)} / λ_i^net），但不匹配连续同向（如 a_b_c → double subscript）
# 例：D_i / c_{ij} / λ_i^net / γ_{ij} / S_i^{(1)}
# 排除：
#   (?<![\\$\w{]) 前面不是 \（命令名）/ $（math 内）/ 单词字符 / {（命令参数内，如 \paragraph{w_RF}）
#   (?![\w.]) 后面不是单词字符或点：防止 v8 实测的 'sensitivity_capacity.png' 误判
_NAKED_SUB_RE = re.compile(
    r"(?<![\\$\w{])"
    r"([A-Za-z" + re.escape(_UNICODE_MATH_CHARS) + r"][A-Za-z0-9]*"
    r"(?:_(?:\{[^}]+\}|[A-Za-z0-9]+))"  # 一个 _ 段
    r"(?:\^(?:\{[^}]+\}|[A-Za-z0-9]+))?"  # 可选一个 ^ 段
    r")"
    r"(?![\w./])"
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

    覆盖 writer RULE 4 治不住的 `D_i`、`c_{ij}`、`S_i^{(1)}`、`λ_i^net`。
    跳过已在 $...$ 内、跳过 \\command 前缀（避免动 \\paragraph{w\\_RF}）。
    若 token 内含 unicode 数学字符（λ/σ 等），同步展开为 LaTeX 命令。

    还要**跳过 display math 块**：`\\[...\\]`、`\\(...\\)`、
    `\\begin{equation}...\\end{equation}`。这些已经是 math mode，再往里嵌 `$...$`
    会触发 'Display math should end with $$' halt。writer 写
    `\\[ \\min \\sum_k d_{ij} x_{ijk} \\]` 时，d_{ij} 不该被再包成 $d_{ij}$。

    已知 over-wrap 风险：`a_b_c` 这种合法变量名也会被当数学。但 paper 段里
    几乎不会出现，且即便错伤，渲出来仍是合法 LaTeX（math italic 显示）。
    """
    if not s:
        return s

    def _sub(m: re.Match) -> str:
        content = m.group(1)
        for ch, cmd in _UNICODE_MATH_MAP.items():
            if ch in content:
                content = content.replace(ch, cmd)
        return f"${content}$"

    def _process_text(t: str) -> str:
        # 已是 display/inline math span 切分后的 text 段：只用 $ 切分跳 inline
        parts = t.split("$")
        for i in range(0, len(parts), 2):
            parts[i] = _NAKED_SUB_RE.sub(_sub, parts[i])
        return "$".join(parts)

    # 先按 display/inline math 块切分；块内 (display) 不动，偶数段 (text) 内继续按 $ 切
    # 兼容：\[ ... \]、\( ... \)、\begin{equation} ... \end{equation}
    display_re = re.compile(
        r"(\\\[.*?\\\]|\\\(.*?\\\)|\\begin\{equation\*?\}.*?\\end\{equation\*?\})",
        re.DOTALL,
    )
    out = []
    last = 0
    for m in display_re.finditer(s):
        # 块前 text 段
        out.append(_process_text(s[last:m.start()]))
        # 块内原样保留
        out.append(m.group(0))
        last = m.end()
    out.append(_process_text(s[last:]))
    return "".join(out)


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
    """markdown pipe-table → LaTeX tabular.

    识别连续行：
        | h1 | h2 |
        |----|----|
        | a  | b  |

    转成：
        \\begin{tabular}{|l|l|}\\hline h1 & h2 \\\\\\hline a & b \\\\\\hline \\end{tabular}

    分隔行 (`---`) 用来确定列数；之前一行作 header。
    不在表内的 `|` 不动。

    cell 内容里的裸 `&` 必须转义为 `\\&`——否则 LaTeX 把它当列分隔符，
    报 'Extra alignment tab' 并 halt。但 `$...$` math 模内的 `&`（如
    align 环境的列对齐符）不动。
    """
    def _escape_cell_amps(cell: str) -> str:
        """转义 cell 里的裸 &，但跳过 $...$ math 模内的和已转义的 \&。"""
        parts = cell.split("$")
        for i in range(0, len(parts), 2):   # 偶数段 = text，奇数段 = math
            parts[i] = re.sub(r"(?<!\\)&", r"\&", parts[i])
        return "$".join(parts)

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
                header_cells = [_escape_cell_amps(c) for c in header_cells]
                ncols = len(header_cells)
                # 从分隔行解析对齐方式：:--- = 左, :---: = 中, ---: = 右, --- = 左（默认）
                sep_cells = [c.strip() for c in sep.strip().strip("|").split("|")]
                # 检查分隔行是否有对齐标记 `:`。没有 `:` 时保持原有 X（等宽自适应列）
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
                # 补齐/截断到 ncols
                col_spec = (col_spec + "X" * ncols)[:ncols]
                tbl = [r"\begin{tabularx}{\linewidth}{" + col_spec + r"}",
                       r"\toprule",
                       " & ".join(header_cells) + r" \\",
                       r"\midrule"]
                j = i + 2
                while j < len(lines) and lines[j].lstrip().startswith("|"):
                    cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                    cells = [_escape_cell_amps(c) for c in cells]
                    # 对齐列数（缺则补空，多则截）
                    cells = (cells + [""] * ncols)[:ncols]
                    tbl.append(" & ".join(cells) + r" \\")
                    j += 1
                tbl.append(r"\bottomrule")
                tbl.append(r"\end{tabularx}")
                # 表格前后留空行让 LaTeX 不把表格挤进段落
                out.append("")
                out.extend(tbl)
                out.append("")
                i = j
                continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _escape_text_chars_skip_math(content: str, chars: str = "_%&#") -> str:
    r"""转义 text-mode 特殊字符，但跳过 $...$ math 段。

    用于 tabularx 等环境内：cell 里混有 text 和 inline math，
    text 段的 _ % # 需转义，math 段的 _ 是下标不能动。
    """
    segs = content.split("$")
    for i in range(0, len(segs), 2):  # 偶数段 = text，奇数段 = math
        for ch in chars:
            segs[i] = re.sub(rf"(?<!\\){re.escape(ch)}", rf"\{ch}", segs[i])
    return "$".join(segs)


def _escape_remaining_underscores(s: str) -> str:
    r"""对 $...$ 之外、math/tabularx 环境之外的 text-mode 特殊字符做转义。

    chain 末尾用：前面几步包了真数学（`D_i` → `$D_i$` 或 \begin{equation}...），
    剩下的特殊字符必然出现在文本段，裸用会让 LaTeX 报错：
    - `_` → Missing $ inserted（text mode 进 math）
    - `&` → Misplaced alignment tab
    - `#` → macro parameter character
    - `%` → 注释掉后续内容——`\textbf{56.7%}` 会吞掉整个段落

    保护以下环境，内部特殊字符按语境处理：
    - math 环境（equation, align, gather, ……，含 \[ \] 和 \( \)）：_ & 都是合法语法，不动
    - tabularx 环境：& 是列分隔符保留；_ % # 仍需转义（text mode）
    """
    if not s:
        return s

    # 收集所有受保护块：(start, end, 处理后的内容)
    protected: list[tuple[int, int, str]] = []

    # 1) \begin{env}...\end{env} 环境
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
            # tabularx: & 是列分隔符保留；_ % # 仍需 text-mode 转义，
            # 但 $...$ math 模内的 _ 是下标语法，不能转义
            content = _escape_text_chars_skip_math(content, chars="_%#")
        # 其他 math 环境：全部不动
        protected.append((m.start(), m.end(), content))

    # 2) \[...\] 和 \(...\) display/inline math
    _display_re = re.compile(r"\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)
    for m in _display_re.finditer(s):
        # 避免嵌套（已经在 env 内的跳过）
        if not any(p[0] <= m.start() < p[1] for p in protected):
            protected.append((m.start(), m.end(), m.group(0)))

    # 3) 合并重叠/相邻区间
    protected.sort()
    merged: list[tuple[int, int, str]] = []
    for start, end, content in protected:
        if merged and start < merged[-1][1]:
            prev_start, prev_end, prev_content = merged.pop()
            merged.append((prev_start, max(prev_end, end), prev_content + content[prev_end - start:]))
        else:
            merged.append((start, end, content))

    # 4) 重建：未保护段做 4 个字符转义
    result: list[str] = []
    pos = 0
    for start, end, content in merged:
        if pos < start:
            text = s[pos:start]
            segs = text.split("$")
            for j in range(0, len(segs), 2):
                segs[j] = re.sub(r"(?<!\\)_", r"\_", segs[j])
                segs[j] = re.sub(r"(?<!\\)%", r"\%", segs[j])
                segs[j] = re.sub(r"(?<!\\)&", r"\&", segs[j])
                segs[j] = re.sub(r"(?<!\\)#", r"\#", segs[j])
            result.append("$".join(segs))
        result.append(content)
        pos = end

    if pos < len(s):
        text = s[pos:]
        segs = text.split("$")
        for j in range(0, len(segs), 2):
            segs[j] = re.sub(r"(?<!\\)_", r"\_", segs[j])
            segs[j] = re.sub(r"(?<!\\)%", r"\%", segs[j])
            segs[j] = re.sub(r"(?<!\\)&", r"\&", segs[j])
            segs[j] = re.sub(r"(?<!\\)#", r"\#", segs[j])
        result.append("$".join(segs))

    return "".join(result)


# 长 inline math token：含 `=` 或 `\leq` / `\geq` / `\sum` / `\prod` 且长度 >= 10
# → 提为独占行 equation（带自动编号）
_LONG_MATH_INDICATORS = ("=", r"\leq", r"\geq", r"\sum", r"\prod", r"\int", r"\le ", r"\ge ", "\\le}", "\\ge}")


def _promote_inline_equations(s: str) -> str:
    r"""把段内"独立式"长公式从 `$...$` 提升为 `\begin{equation}...\end{equation}`（带编号）。

    判定规则：行内 `$...$` 满足以下全部条件 → 提升
      1. 内容包含等号或 \\sum/\\prod/\\int/\\leq/\\geq
      2. 长度 >= 10 字符（避免 $x=1$ 这种简短赋值被打散）
      3. 周围以中文标点（。，；：）或行边界结尾 → 说明 writer 用它当独立式

    不动：行内简单引用（`$x_i$`、`$\\alpha$`）；多个连续 inline math。
    **跳过 tabularx 表格区域**——表格 cell 内的 inline math 提升为 equation
    块会破坏 tabularx（块级环境不能嵌在表格单元里）。
    """
    if not s or "$" not in s:
        return s

    def _is_long_equation(content: str) -> bool:
        if len(content) < 10:
            return False
        return any(ind in content for ind in _LONG_MATH_INDICATORS)

    # 先按 tabularx 块切分，只对表格外文本做提升
    _TABLE_SPLIT_RE = re.compile(
        r"(\\begin\{tabularx\}.*?\\end\{tabularx\})",
        re.DOTALL,
    )
    segments = _TABLE_SPLIT_RE.split(s)
    result = []
    for seg_idx, seg in enumerate(segments):
        # split 的偶数索引 = 表格外文本，奇数索引 = tabularx 块（保留不动）
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
            # 找匹配 $
            end = s.find("$", i + 1)
            if end == -1:
                out.append(s[i:])
                break
            content = s[i + 1:end]
            after = s[end + 1:end + 2]
            before = s[max(0, i - 1):i]
            # before 是中文标点或行首/换行；after 是中文标点或行末/换行
            sep_chars = set("。，；：、 \n\t")
            before_ok = before == "" or before in sep_chars or before in "。，；：、"
            after_ok = after == "" or after in sep_chars or after in "。，；：、"
            if _is_long_equation_inner(content) and before_ok and after_ok:
                # 提升为 equation 块；吃掉紧邻其后的中文标点（公式独占行后这些标点已冗余）
                eat = end + 1
                while eat < len(s) and s[eat] in "。，；：、 \t":
                    eat += 1
                # 同时如果跳过的字符里含 \n，保留一个换行
                if "\n" in s[end + 1:eat] or eat < len(s) and s[eat] == "\n":
                    pass  # 自然换行已在 equation 块两侧加了
                out.append("\n\\begin{equation}\n" + content + "\n\\end{equation}\n")
                i = eat
                continue
            # 保留 inline
            out.append(s[i:end + 1])
            i = end + 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _is_long_equation_inner(content: str) -> bool:
    if len(content) < 10:
        return False
    return any(ind in content for ind in _LONG_MATH_INDICATORS)



def _prepare_section(s: str) -> str:
    """paper 段渲染前预处理：链式确定性转换，全部跳过已有 $...$。

    顺序敏感：
    1. markdown 表格 → tabular（在 inline code/heading 之前；表格里可能有 backtick 数学）
    2. markdown 标题 `### xxx` → `\\subsubsection{xxx}`
    3. markdown 加粗 `**x**` → `\\textbf{x}`
    4. markdown 列表 `- x` → itemize
    5. markdown 反引号 `` `S_i` `` → `$S_i$`（内容里 unicode 同步展开）
    6. 裸 LaTeX 下标 `D_i` → `$D_i$`（writer 漏的兜底）
    7. unicode 数学符号 α → `$\\alpha$`（在 $...$ 之外的）
    8. 残留的 text-mode 特殊字符 `_` `%` `&` `#` → 转义（文件名/百分比/特殊符）

    **不调 _latex_escape**——writer 在 paper 段会故意写 LaTeX inline math。
    """
    s = _md_table_to_latex(s)
    s = _md_headings_to_latex(s)
    s = _md_bold_to_latex(s)
    s = _md_bullets_to_latex(s)
    s = _md_inline_code_to_math(s)
    s = _wrap_naked_subscripts(s)
    s = _wrap_unicode_math(s)
    s = _promote_inline_equations(s)
    s = _pad_math_commands(s)   # `$\cdot$dist_{ij}$` -> `$\cdot\,dist_{ij}$`
    s = _escape_remaining_underscores(s)
    return s


def _truncate_caption(s: str, *, max_chars: int = 55) -> str:
    """把长图注截到 max_chars 以内，但优先切在完整句/短语边界。

    LLM 写的图 caption 常常两三个句子；直接 `s[:55]` 会切在逗号/单字上（"…成本增加，"、"…前"）。
    策略：先看 max_chars 处是否已是终结符；否则在 [max_chars*0.6, max_chars] 内找最靠后的
    句末字符（。！？；.!?）；没有则退到最靠后的逗号（，、,）；再退不到就硬截。
    """
    if not s or len(s) <= max_chars:
        return s
    hard_end = s[max_chars - 1]
    if hard_end in "。！？；.!?":
        return s[:max_chars]
    lo = max(1, int(max_chars * 0.6))
    window = s[lo:max_chars]
    for stops in ("。！？；.!?", "，、,"):
        idx = max((window.rfind(c) for c in stops), default=-1)
        if idx != -1:
            return s[: lo + idx + 1]
    return s[:max_chars]


def _curate_code(code: str, max_lines: int = 80) -> str:
    """截取代码前 max_lines 行（ponytail: 一行 Python，不注册 Jinja2 过滤器）。"""
    lines = code.split("\n")
    if len(lines) <= max_lines:
        return code
    return "\n".join(lines[:max_lines]) + f"\n# ... (共 {len(lines)} 行，截取前 {max_lines} 行)"


def _curate_stdout(stdout: str) -> str:
    """提取 stdout 关键行：RESULT: 行 + 末尾 5 行。"""
    if not stdout:
        return ""
    lines = stdout.splitlines()
    result_lines = [l for l in lines if l.strip().startswith("RESULT:")]
    tail = lines[-5:]
    # 去重保序
    seen = set()
    out = []
    for l in result_lines + tail:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return "\n".join(out)


def latex_node(state: MathModelingState) -> dict:
    workdir = Path(state.output_dir or ".")
    workdir.mkdir(parents=True, exist_ok=True)

    # ponytail: 为图重打一份 LaTeX 安全视图，避免 caption/path 里的 _/&/% 炸编译
    # caption 尽量截到最近的句/短语边界，避免"曲线呈下降趋势，"这种半截逗号结尾
    # analysis 走完整 _prepare_section（v8 实测：figure_pipeline LLM 在图说里
    # 会写 sensitivity_capacity.png 这种文件名，必须 escape）
    safe_figures = [
        FigureArtifact(
            path=_latex_path(f.path),
            purpose=_latex_escape(f.purpose),
            caption=_latex_escape(_truncate_caption(f.caption or f.purpose, max_chars=55)),
            quality_score=f.quality_score,
            quality_issues=list(f.quality_issues),
            analysis=_prepare_section(f.analysis),
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
        code_artifacts=[
            {
                "purpose": a.purpose, "code": a.code, "stdout": a.stdout,
                "success": a.success, "artifact_paths": a.artifact_paths,
                "curated_code": _curate_code(a.code),
                "curated_stdout": _curate_stdout(a.stdout),
            }
            for a in state.code_artifacts if a.success
        ],
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
            keywords=(state.paper.keywords or "数学建模").strip(),
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
