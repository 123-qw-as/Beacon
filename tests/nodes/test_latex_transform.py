"""Tests for math_agent.nodes.latex_transform — pure text transform functions."""
from math_agent.nodes.latex_transform import (
    _wrap_unicode_math, _md_headings_to_latex, _md_inline_code_to_math,
    _wrap_naked_subscripts, _md_bold_to_latex, _md_bullets_to_latex,
    _md_table_to_latex, _escape_remaining_underscores, _promote_inline_equations,
    _pad_math_commands, _prepare_section,
)


# ---- _wrap_unicode_math ----

def test_wrap_unicode_math_handles_greek_and_relations():
    out = _wrap_unicode_math("参数 α 满足 σ ≥ 0，误差 ±5")
    assert r"$\alpha$" in out
    assert r"$\sigma$" in out
    assert r"$\geq$" in out
    assert r"$\pm$" in out
    assert "参数" in out


def test_wrap_unicode_math_skips_inside_existing_math():
    """已在 $...$ 内的不再二次包裹；$...$ 外的孤立 unicode（含下标）正确包。"""
    out = _wrap_unicode_math(r"$\sigma_d$ 与 σ_r")
    assert out.count("$") % 2 == 0
    assert r"$\sigma_d$" in out
    assert r"$\sigma_r$" in out
    assert r"$\sigma$_r" not in out


def test_wrap_unicode_math_no_op_on_pure_text():
    s = "纯中文与 ascii 段落 abc 123."
    assert _wrap_unicode_math(s) == s


def test_wrap_unicode_math_attaches_subscript():
    """α_i / σ_{d,i} 必须整体包，否则 _i 进 text mode 又会炸。"""
    out = _wrap_unicode_math("比例 α_i 与 σ_{d,i} 控制波动")
    assert r"$\alpha_i$" in out
    assert r"$\sigma_{d,i}$" in out
    assert r"$\alpha$_i" not in out


# ---- _md_headings_to_latex ----

def test_md_headings_to_latex():
    """### basic阶段 → \\subsubsection{basic阶段}（带编号版本以入目录）。"""
    s = "### basic阶段\n内容\n\n## 章节"
    out = _md_headings_to_latex(s)
    assert r"\subsubsection{basic阶段}" in out
    assert r"\subsection{章节}" in out
    assert "###" not in out
    assert "## 章节" not in out


def test_md_headings_only_match_line_start():
    """正文里普通 # 不当标题（如 markdown 内联用法、注释）。"""
    s = "成本 #1 是 ## 5\n# title\nbody"
    out = _md_headings_to_latex(s)
    assert "成本 #1 是 ## 5" in out
    assert r"\section{title}" in out


# ---- _md_inline_code_to_math ----

def test_md_inline_code_to_math():
    """`S_i` → $S_i$（v6.2 实测：writer 在符号表里这么写）。"""
    s = "符号 `S_i` 表示站点 i 库存，公式 `S_i^{(1)}` 是阶段 1。"
    out = _md_inline_code_to_math(s)
    assert "$S_i$" in out
    assert "$S_i^{(1)}$" in out
    assert "`" not in out


def test_md_inline_code_unwraps_unicode_inside():
    """`α_i` → $\\alpha_i$（内容里的 unicode 同步展开为 LaTeX 命令）。"""
    s = "比例 `α_i` 和 `γ`"
    out = _md_inline_code_to_math(s)
    assert r"$\alpha_i$" in out
    assert r"$\gamma$" in out
    assert "α" not in out
    assert "γ" not in out


def test_md_inline_code_preserves_normal_text():
    """没有反引号的字符串不动。"""
    s = "纯文本，无 backticks"
    assert _md_inline_code_to_math(s) == s


# ---- _wrap_naked_subscripts ----

def test_wrap_naked_subscripts_handles_writer_omissions():
    """D_i / c_{ij} / S_i^{(1)} 等裸下标自动包成 $...$。"""
    s = "预测各站点需求 D_i 和运输成本 c_{ij}，库存 S_i^{(1)} 初始化"
    out = _wrap_naked_subscripts(s)
    assert "$D_i$" in out
    assert "$c_{ij}$" in out
    assert "$S_i^{(1)}$" in out


def test_wrap_naked_subscripts_skips_inside_math():
    """已在 $...$ 内的不二次包裹。"""
    s = "$D_i$ 与 c_{ij}"
    out = _wrap_naked_subscripts(s)
    assert out.count("$D_i$") == 1
    assert "$c_{ij}$" in out


def test_wrap_naked_subscripts_skips_command_args():
    """\\paragraph{w_RF} 这种 LaTeX 命令参数里的 _ 不动（已被上游 escape 为 \\_）。"""
    s = r"\paragraph{w\_RF}"
    out = _wrap_naked_subscripts(s)
    assert out == s


def test_wrap_naked_subscripts_handles_unicode_ident_head():
    """λ_i^net、γ_{ij} 这种 unicode 起头 + 下标也要整体包，并展开 unicode。"""
    out = _wrap_naked_subscripts("参数 λ_i^net 与 γ_{ij}")
    assert r"$\lambda_i^net$" in out
    assert r"$\gamma_{ij}$" in out
    assert "λ_i^net" not in out
    assert "γ_{ij}" not in out


def test_wrap_naked_subscripts_skips_filenames_and_paths():
    """v8 实测：sensitivity_capacity.png / attempt_0/_run.py 这类文件名/路径不能误包。"""
    s = "图（sensitivity_capacity.png）参考 attempt_0/_run.py 与变量 D_i"
    out = _wrap_naked_subscripts(s)
    assert "sensitivity_capacity.png" in out
    assert "attempt_0/_run.py" in out
    assert "$D_i$" in out


def test_wrap_naked_subscripts_skips_display_math_brackets():
    r"""`\[ ... d_{ij} ... \]` 是 display math，d_{ij} 已在 math mode，
    不应再被包成 `$d_{ij}$` 嵌进 display math，否则 xelatex halt。"""
    src = r"目标函数 \[ \min \sum_k d_{ij} x_{ijk} \] 是最小化。"
    out = _wrap_naked_subscripts(src)
    assert "$d_{ij}$" not in out, out
    assert "$x_{ijk}$" not in out, out
    assert r"\[ \min \sum_k d_{ij} x_{ijk} \]" in out


def test_wrap_naked_subscripts_skips_equation_block():
    src = r"如下：\begin{equation} \sum_k d_{ij} \end{equation} 结束。"
    out = _wrap_naked_subscripts(src)
    assert "$d_{ij}$" not in out
    src2 = r"前面 s_i 块 \begin{equation} \sum_k d_{ij} \end{equation} 后面 c_{ab}"
    out2 = _wrap_naked_subscripts(src2)
    assert "$s_i$" in out2
    assert "$c_{ab}$" in out2
    assert "$d_{ij}$" not in out2


def test_wrap_naked_subscripts_skips_paren_display_math():
    src = r"用 \( \sum_i x_{ij} \) 形式。"
    out = _wrap_naked_subscripts(src)
    assert "$x_{ij}$" not in out


# ---- _escape_remaining_underscores ----

def test_escape_remaining_underscores_in_text():
    """$...$ 外残留的 _ 必须 escape，否则 LaTeX text mode 报 Missing $."""
    out = _escape_remaining_underscores("图（sensitivity_capacity.png）显示")
    assert "sensitivity\\_capacity.png" in out
    out2 = _escape_remaining_underscores("$D_i$ 与 sensitivity_x.png")
    assert "$D_i$" in out2
    assert "sensitivity\\_x.png" in out2
    out3 = _escape_remaining_underscores(r"\paragraph{w\_RF}")
    assert out3 == r"\paragraph{w\_RF}"


def test_escape_remaining_underscores_skips_equation_blocks():
    """equation 块内 _ 是合法 math 语法，不能 escape。"""
    s = "见 sensitivity_v8.png：\n\\begin{equation}\n\\hat{d}_i = \\sum_{k=1}^N d_i^{(k)}\n\\end{equation}\n后文 file_y.png"
    out = _escape_remaining_underscores(s)
    assert r"\hat{d}_i = \sum_{k=1}^N d_i^{(k)}" in out
    assert "sensitivity\\_v8.png" in out
    assert "file\\_y.png" in out


def test_escape_remaining_underscores_skips_math_inside_tabularx():
    """tabularx cell 内的 $...$ math 模下标不能被转义。"""
    s = (
        "\\begin{tabularx}{\\linewidth}{XX}\n"
        "\\toprule\n"
        "符号 & 含义 \\\\\n"
        "\\midrule\n"
        "$D_{i,t}$ & 站点需求 \\\\\n"
        "$\\hat{D}_{i,t}^{(\\alpha)}$ & 预测值 \\\\\n"
        "file_name.png & 文件名 \\\\\n"
        "\\bottomrule\n"
        "\\end{tabularx}\n"
    )
    out = _escape_remaining_underscores(s)
    assert r"$D_{i,t}$" in out, f"math 内下标被转义: {out!r}"
    assert r"$\hat{D}_{i,t}^{(\alpha)}$" in out, f"math 内下标被转义: {out!r}"
    assert r"file\_name.png" in out, f"文件名未转义: {out!r}"


# ---- _md_bold_to_latex ----

def test_md_bold_to_latex():
    s = "**假设1**：内容; 普通 ** 段; **依据**：x"
    out = _md_bold_to_latex(s)
    assert r"\textbf{假设1}" in out
    assert r"\textbf{依据}" in out
    assert "** 段;" in out


def test_md_bold_skips_whitespace_only():
    """**     ** 这种空白粗体不动。"""
    s = "**   **"
    assert _md_bold_to_latex(s) == s


# ---- _md_bullets_to_latex ----

def test_md_bullets_to_latex():
    s = "段落起头\n- a 项\n- b 项\n\n下一段\n* c 项"
    out = _md_bullets_to_latex(s)
    assert r"\begin{itemize}" in out
    assert r"\item a 项" in out
    assert r"\item b 项" in out
    assert r"\item c 项" in out
    assert r"\end{itemize}" in out


# ---- _md_table_to_latex ----

def test_md_table_to_latex():
    s = """符号说明前导。
| 符号 | 含义 | 单位 |
|------|------|------|
| S_i | 库存 | 辆 |
| D_i | 需求 | 辆 |

下文。"""
    out = _md_table_to_latex(s)
    assert r"\begin{tabularx}{\linewidth}{XXX}" in out
    assert r"\toprule" in out
    assert r"\midrule" in out
    assert r"\bottomrule" in out
    assert r"符号 & 含义 & 单位 \\" in out
    assert r"S_i & 库存 & 辆 \\" in out
    assert r"\end{tabularx}" in out
    assert r"\hline" not in out
    assert "|------|" not in out
    assert "| 符号 |" not in out
    assert "下文。" in out


def test_md_table_escapes_ampersand_in_cells():
    r"""cell 内容里的裸 & 必须转义为 \&。"""
    s = """| 符号 | 含义 | 单位 |
|------|------|------|
| F | 固定&变动成本 | 元 |
| $q_i$ | 调度量 | 辆 |"""
    out = _md_table_to_latex(s)
    assert r"固定\&变动成本" in out
    f_line = [l for l in out.split("\n") if "固定" in l][0]
    assert f_line.count(r"\&") == 1
    assert f_line.count(" & ") == 2


def test_md_table_to_latex_no_op_when_no_table():
    s = "无表格段落，但有 | 符号 |。"
    assert _md_table_to_latex(s) == s


def test_md_table_does_not_double_escape_ampersand():
    """I2 回归：已转义的 \\& 不应被二次转义为 \\\\&。"""
    bs = chr(92)
    s = f"| x | 固定{bs}&变动成本 | 元 |\n|---|---|---|"
    out = _md_table_to_latex(s)
    x_line = [l for l in out.split("\n") if "固定" in l][0]
    assert bs + bs + "&" not in x_line, f"double-escaped: {x_line!r}"
    assert bs + "&" in x_line


# ---- _promote_inline_equations ----

def test_promote_inline_equations_basic():
    """段内独立的长公式（含 =）应被提升为 equation 块。"""
    s = "调度后存量满足 $\\eta_i = s_i + x_i - y_i + \\xi_i$。其余约束如下。"
    out = _promote_inline_equations(s)
    assert r"\begin{equation}" in out
    assert r"\end{equation}" in out
    assert r"\eta_i = s_i + x_i - y_i + \xi_i" in out


def test_promote_inline_equations_leaves_short_inline():
    """短的 inline math（$x_i$、$\\alpha$）不提升。"""
    s = "参数 $x_i$ 和 $\\alpha$ 控制需求。"
    out = _promote_inline_equations(s)
    assert r"\begin{equation}" not in out
    assert "$x_i$" in out
    assert r"$\alpha$" in out


def test_promote_inline_equations_skips_when_no_separator():
    """行内夹在中文段中的简短 inline math 不动（即便长一点）。"""
    s = "因此$\\sum x_i$代表所有调度量。"
    out = _promote_inline_equations(s)
    assert r"\begin{equation}" not in out


def test_promote_inline_equations_eats_trailing_punctuation():
    """公式升 equation 块后，吃掉紧邻其后的中文标点。"""
    s = "需求预测：$\\hat{d}_i = \\sum x_i$。净需求：$net_i = d_i - s_i$。"
    out = _promote_inline_equations(s)
    assert r"\begin{equation}" in out
    for line in out.split("\n"):
        stripped = line.strip()
        if stripped:
            assert not stripped.startswith("。"), f"line starts with stray 。: {line!r}"
            assert not stripped.startswith("，"), f"line starts with stray ，: {line!r}"


def test_promote_inline_equations_skips_inside_tabularx():
    """表格 cell 内的长 inline math 不应提升为 equation 块。"""
    s = (
        "\\begin{tabularx}{\\linewidth}{XXXX}\n"
        "\\toprule\n"
        "符号 & 含义 & 单位 & 类别 \\\\\n"
        "\\midrule\n"
        "$\\text{MAE}$ & 预测误差，$\\text{MAE}=\\frac{1}{NT}\\sum_{i,t}|\\hat{D}_{i,t}-D_{i,t}|$ & 辆 & 指标 \\\\\n"
        "\\bottomrule\n"
        "\\end{tabularx}\n"
    )
    out = _promote_inline_equations(s)
    assert r"\begin{equation}" not in out, (
        f"表格内 inline math 被提升为 equation，会破坏 tabularx: {out!r}"
    )
    assert r"$\text{MAE}=\frac{1}{NT}" in out


# ---- _pad_math_commands ----

def test_pad_math_commands_splits_cdot_stuck_to_letters():
    assert _pad_math_commands(r"$\cdotdist_{ij}$") == r"$\cdot\,dist_{ij}$"


def test_pad_math_commands_splits_sum_stuck_to_exp():
    assert _pad_math_commands(r"$\sumexp(x)$") == r"$\sum\,exp(x)$"


def test_pad_math_commands_leaves_legit_cdotp_alone():
    r"""\cdotp 是合法命令名（不是 \cdot+p），不能拆。"""
    assert _pad_math_commands(r"$\cdotp$") == r"$\cdotp$"


def test_pad_math_commands_leaves_trailing_subscript_alone():
    assert _pad_math_commands(r"$\alpha_i$") == r"$\alpha_i$"
    assert _pad_math_commands(r"$\sum_{i=1}^n a_i$") == r"$\sum_{i=1}^n a_i$"


def test_pad_math_commands_leaves_unknown_macro_alone():
    r"""写手自定义宏 \myVar 无法判断，保持原样避免误伤。"""
    assert _pad_math_commands(r"$\myVar$") == r"$\myVar$"


def test_pad_math_commands_skips_text_span():
    assert _pad_math_commands(r"$\cdot$dist") == r"$\cdot$dist"


def test_pad_math_commands_covers_equation_block():
    src = r"\begin{equation}\cdotdist_{ij}\end{equation}"
    assert _pad_math_commands(src) == r"\begin{equation}\cdot\,dist_{ij}\end{equation}"


def test_pad_math_commands_handles_multiple_in_one_span():
    assert _pad_math_commands(r"$a\cdotb\cdotc$") == r"$a\cdot\,b\cdot\,c$"


# ---- _prepare_section (end-to-end pipeline) ----

def test_prepare_section_table_inline_math_not_promoted(workdir):
    """端到端：markdown 表格含长 inline math → 转 tabularx 后不出现 equation。"""
    md = (
        "| 符号 | 含义 |\n"
        "|------|------|\n"
        "| $\\text{MAE}$ | 误差 $\\text{MAE}=\\frac{1}{NT}\\sum_{i,t}|D_i|$ |\n"
    )
    out = _prepare_section(md)
    assert r"\begin{equation}" not in out, f"表格内 inline math 被提升: {out!r}"
    assert r"\begin{tabularx}" in out


def test_prepare_section_pipeline_defuses_cdot_dist():
    r"""端到端：_prepare_section 应把 writer 常见的 `$\cdot$dist_{ij}$`
    最终产出可编译 tex（\cdot 与 dist 之间不粘）。"""
    src = r"目标函数含 $\cdot$dist_{ij}$ 项。"
    out = _prepare_section(src)
    assert "cdotdist" not in out


def test_prepare_section_defuses_nested_inline_in_display():
    r"""端到端：writer 常见 `\[ \min \sum ... d_{ij} x_{ijk} \]`
    经过 _prepare_section 后不能含 `$d_{ij}$` 这种嵌套进 display math 的 $...$。"""
    src = r"目标 \[ \min \sum\_{k} d_{ij} x_{ijk} \] 结束。"
    out = _prepare_section(src)
    assert "$d_{ij}$" not in out
    assert "$x_{ijk}$" not in out


def test_complete_section_escapes_percent_in_textbf():
    """P0-1 回归：**56.7%** → \\textbf{56.7\\%}，% 必须转义。"""
    bs = chr(92)
    out = _prepare_section("成本降低 **56.7%**")
    assert bs + "textbf{56.7" + bs + "%}" in out
    assert "56.7%}" not in out


def test_complete_section_escapes_amp_hash_percent_in_text():
    """P0-2 回归：文本段里的 & # % 都要转义。"""
    bs = chr(92)
    out = _prepare_section("100% 完成 & 排名 #1")
    assert "100" + bs + "%" in out
    assert bs + "&" in out
    assert bs + "#" in out


def test_complete_section_preserves_math_subsets():
    r"""P1-1 回归：\subseteq 不被拆分为 \subset\,eq。"""
    out = _prepare_section(r"集合 $A \subseteq B$")
    assert r"\subseteq" in out
    assert r"\subset\,eq" not in out
