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
    # 二元/一元算子
    "cdot", "cdotp", "times", "div", "pm", "mp", "ast", "star", "circ",
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
    # 常用符号
    "infty", "partial", "nabla", "forall", "exists", "emptyset", "hbar",
    "ell", "Re", "Im", "aleph", "cdots", "ldots", "vdots", "ddots",
    # 字体/装饰
    "hat", "bar", "tilde", "vec", "dot", "ddot", "overline", "underline",
    "widehat", "widetilde", "mathbb", "mathbf", "mathcal", "mathrm",
    "boldsymbol", "text", "textrm", "textbf", "textit",
    # 分式/根号
    "frac", "sqrt", "binom", "dfrac", "tfrac",
    # 左右括号
    "left", "right",
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

    # 先按 equation 块切；块内直接处理，块外再按 $ 切处理 inline math span
    eq_re = re.compile(r"(\\begin\{equation\*?\}.*?\\end\{equation\*?\})", re.DOTALL)
    outer = eq_re.split(s)
    for k in range(len(outer)):
        if outer[k].startswith(r"\begin{equation"):
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
# 例：D_i / c_{ij} / λ_i^net / γ_{ij}
# 排除：
#   (?<![\\$\w]) 前面不是 \（命令名）/ $（math 内）/ 单词字符（避免中段切）
#   (?![\w.]) 后面不是单词字符或点：防止 v8 实测的 'sensitivity_capacity.png' 误判
_NAKED_SUB_RE = re.compile(
    r"(?<![\\$\w])"
    r"([A-Za-z" + re.escape(_UNICODE_MATH_CHARS) + r"][A-Za-z0-9]*"
    r"(?:[_^](?:\{[^}]+\}|[A-Za-z0-9]+))+)"
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
        """转义 cell 里的裸 &，但跳过 $...$ math 模内的。"""
        parts = cell.split("$")
        for i in range(0, len(parts), 2):   # 偶数段 = text，奇数段 = math
            parts[i] = parts[i].replace("&", r"\&")
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
                # tabularx 自适应列宽 + booktabs 三线表（gmcmthesis 已 RequirePackage 二者）
                # 三线表惯例：\toprule（顶）/ \midrule（表头下）/ \bottomrule（底），中间无横线
                col_spec = "X" * ncols  # tabularx X 列等分 \linewidth；三线表不画竖线
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


def _escape_remaining_underscores(s: str) -> str:
    r"""对 $...$ 之外、equation 块之外残留的 `_` 做 escape。

    chain 末尾用：前面几步包了真数学（`D_i` → `$D_i$` 或 \begin{equation}...），
    剩下的 `_` 必然出现在文件名 (`sensitivity_capacity.png`) / 路径 / 中文中间，
    这些都属于 text mode，裸用会让 LaTeX 进 math mode 触发 'Missing $ inserted'。

    跳过 \command 后紧跟的 `_`（保留 `\paragraph{w\_RF}` 已 escape 形式）。
    跳过 \begin{equation}...\end{equation} 块内（公式里的 _ 是合法 math 语法）。
    """
    if not s:
        return s
    # 先按 equation 块切分，块内不动；块外再按 $ 切分
    parts = re.split(r"(\\begin\{equation\*?\}.*?\\end\{equation\*?\})", s, flags=re.DOTALL)
    for i in range(len(parts)):
        if parts[i].startswith(r"\begin{equation"):
            continue  # 公式块内 _ 不动
        # 块外按 $ 再切
        segs = parts[i].split("$")
        for j in range(0, len(segs), 2):
            segs[j] = re.sub(r"(?<!\\)_", r"\_", segs[j])
        parts[i] = "$".join(segs)
    return "".join(parts)


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
    """
    if not s or "$" not in s:
        return s

    def _is_long_equation(content: str) -> bool:
        if len(content) < 10:
            return False
        return any(ind in content for ind in _LONG_MATH_INDICATORS)

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
            if _is_long_equation(content) and before_ok and after_ok:
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
    8. 残留的裸 `_` → `\\_`（文件名/路径里的 `_` 必须 escape，否则 text mode 进 math）

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
