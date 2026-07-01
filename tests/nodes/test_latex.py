from math_agent.state import MathModelingState, PaperSections
from math_agent.nodes.latex import (
    latex_node, _wrap_unicode_math, _md_headings_to_latex, _md_inline_code_to_math,
    _wrap_naked_subscripts, _md_bold_to_latex, _md_bullets_to_latex, _md_table_to_latex,
    _escape_remaining_underscores, _promote_inline_equations,
)


def _state(workdir):
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.paper = PaperSections(
        abstract="a"*100, problem_restatement="b"*100, assumptions="c"*100,
        notation="d"*100, model_section="e"*100, solution="f"*100,
        sensitivity="g"*100, conclusion="h"*100, references="-",
    )
    return s


def test_latex_node_writes_tex_and_markdown_fallback(mocker, workdir):
    mocker.patch(
        "math_agent.nodes.latex.compile_latex",
        return_value=type("R", (), {"success": False, "pdf_path": "", "log": "no xelatex"})(),
    )
    s = _state(workdir)
    delta = latex_node(s)
    assert (workdir / "paper.tex").exists()
    assert (workdir / "paper.md").exists()
    assert delta["errors"]
    assert "no xelatex" in delta["errors"][0]


def test_latex_node_records_pdf_path_on_success(mocker, workdir):
    pdf = workdir / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    mocker.patch(
        "math_agent.nodes.latex.compile_latex",
        return_value=type("R", (), {"success": True, "pdf_path": str(pdf), "log": ""})(),
    )
    s = _state(workdir)
    delta = latex_node(s)
    assert (workdir / "paper.tex").exists()
    assert delta == {} or "errors" not in delta


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
    # 旧 $...$ 段保留
    assert r"$\sigma_d$" in out
    # 新 σ_r 被整体包（不是 $\sigma$_r 这种破裂形式）
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
    assert r"$\alpha$_i" not in out  # 错误拆分


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
    assert "成本 #1 是 ## 5" in out  # 行内 # 不动
    assert r"\section{title}" in out


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
    assert out == s  # 不动


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
    # 文件名/路径保持原样
    assert "sensitivity_capacity.png" in out
    assert "attempt_0/_run.py" in out
    # 真实数学仍包
    assert "$D_i$" in out


def test_escape_remaining_underscores_in_text():
    """$...$ 外残留的 _ 必须 escape，否则 LaTeX text mode 报 Missing $."""
    out = _escape_remaining_underscores("图（sensitivity_capacity.png）显示")
    assert "sensitivity\\_capacity.png" in out
    # $...$ 内不动
    out2 = _escape_remaining_underscores("$D_i$ 与 sensitivity_x.png")
    assert "$D_i$" in out2
    assert "sensitivity\\_x.png" in out2
    # 已 escape 的 \_ 不二次 escape
    out3 = _escape_remaining_underscores(r"\paragraph{w\_RF}")
    assert out3 == r"\paragraph{w\_RF}"


def test_escape_remaining_underscores_skips_equation_blocks():
    """equation 块内 _ 是合法 math 语法，不能 escape。"""
    s = "见 sensitivity_v8.png：\n\\begin{equation}\n\\hat{d}_i = \\sum_{k=1}^N d_i^{(k)}\n\\end{equation}\n后文 file_y.png"
    out = _escape_remaining_underscores(s)
    # equation 块内不动
    assert r"\hat{d}_i = \sum_{k=1}^N d_i^{(k)}" in out
    # 块外文件名 escape
    assert "sensitivity\\_v8.png" in out
    assert "file\\_y.png" in out


def test_md_inline_code_preserves_normal_text():
    """没有反引号的字符串不动。"""
    s = "纯文本，无 backticks"
    assert _md_inline_code_to_math(s) == s


def test_latex_node_processes_markdown_in_paper(mocker, workdir):
    """端到端：paper 段含 ### 三级标题与 unicode 数学，渲到 tex 后两者都被处理。"""
    mocker.patch(
        "math_agent.nodes.latex.compile_latex",
        return_value=type("R", (), {"success": True, "pdf_path": "", "log": ""})(),
    )
    s = _state(workdir)
    s.paper.model_section = "### basic阶段\n参数 σ 是常数。"
    latex_node(s)
    tex = (workdir / "paper.tex").read_text(encoding="utf-8")
    assert r"\subsubsection{basic阶段}" in tex
    assert r"$\sigma$" in tex
    assert "###" not in tex


def test_md_bold_to_latex():
    s = "**假设1**：内容; 普通 ** 段; **依据**：x"
    out = _md_bold_to_latex(s)
    assert r"\textbf{假设1}" in out
    assert r"\textbf{依据}" in out
    # 中间带空白起头的孤立 ** 不被错配（仍保留至少一处）
    assert "** 段;" in out


def test_md_bold_skips_whitespace_only():
    """**     ** 这种空白粗体不动。"""
    s = "**   **"
    assert _md_bold_to_latex(s) == s


def test_md_bullets_to_latex():
    s = "段落起头\n- a 项\n- b 项\n\n下一段\n* c 项"
    out = _md_bullets_to_latex(s)
    assert r"\begin{itemize}" in out
    assert r"\item a 项" in out
    assert r"\item b 项" in out
    assert r"\item c 项" in out
    assert r"\end{itemize}" in out


def test_md_table_to_latex():
    s = """符号说明前导。
| 符号 | 含义 | 单位 |
|------|------|------|
| S_i | 库存 | 辆 |
| D_i | 需求 | 辆 |

下文。"""
    out = _md_table_to_latex(s)
    # 三线表（booktabs）：toprule/midrule/bottomrule，列规范不带竖线
    assert r"\begin{tabularx}{\linewidth}{XXX}" in out
    assert r"\toprule" in out
    assert r"\midrule" in out
    assert r"\bottomrule" in out
    assert r"符号 & 含义 & 单位 \\" in out
    assert r"S_i & 库存 & 辆 \\" in out
    assert r"\end{tabularx}" in out
    # 数据行间不再有 \hline
    assert r"\hline" not in out
    assert "|------|" not in out
    assert "| 符号 |" not in out
    assert "下文。" in out


def test_promote_inline_equations_promotes_long_equation():
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
    # 前后是中文（非空格/标点），不提升
    assert r"\begin{equation}" not in out


def test_promote_inline_equations_eats_trailing_punctuation():
    """公式升 equation 块后，吃掉紧邻其后的中文标点，避免下一段开头出现孤立 '。'。"""
    s = "需求预测：$\\hat{d}_i = \\sum x_i$。净需求：$net_i = d_i - s_i$。"
    out = _promote_inline_equations(s)
    assert r"\begin{equation}" in out
    # 不能出现以 '。' 开头的段落（之前的 bug）
    for line in out.split("\n"):
        stripped = line.strip()
        if stripped:
            assert not stripped.startswith("。"), f"line starts with stray 。: {line!r}"
            assert not stripped.startswith("，"), f"line starts with stray ，: {line!r}"


def test_md_table_to_latex_no_op_when_no_table():
    s = "无表格段落，但有 | 符号 |。"
    assert _md_table_to_latex(s) == s


def test_latex_node_title_only_first_line(mocker, workdir):
    """title 只取 problem 第一行，避免长问题描述塞进标题。"""
    mocker.patch(
        "math_agent.nodes.latex.compile_latex",
        return_value=type("R", (), {"success": True, "pdf_path": "", "log": ""})(),
    )
    s = _state(workdir)
    s.problem = "城市共享单车调度优化\n建立模型预测各站点未来一小时的需求量。\n在不超过 100 辆运力的前提下，设计最优调度方案。"
    latex_node(s)
    tex = (workdir / "paper.tex").read_text(encoding="utf-8")
    assert r"\title{城市共享单车调度优化}" in tex
    # 问题正文不应进入 title 命令
    assert r"建立模型预测各站点未来" not in tex.split(r"\title{")[1].split("}")[0]


def test_latex_node_figure_caption_truncated(mocker, workdir):
    """figure caption 截到 55 字，避免标题占两行。"""
    from math_agent.state import FigureArtifact
    mocker.patch(
        "math_agent.nodes.latex.compile_latex",
        return_value=type("R", (), {"success": True, "pdf_path": "", "log": ""})(),
    )
    s = _state(workdir)
    s.figures.append(FigureArtifact(
        path="a.png", purpose="test",
        caption="这是一段非常长的图说" * 10,  # 90 字
        analysis="完整分析" * 20,
    ))
    latex_node(s)
    tex = (workdir / "paper.tex").read_text(encoding="utf-8")
    # caption 出现在 \caption{} 内，限 55 字
    import re
    m = re.search(r"\\caption\{([^}]+)\}", tex)
    assert m and len(m.group(1)) <= 56  # +1 容忍尾标点


def test_default_paper_tex_template_declares_tabularx_and_booktabs():
    r"""回归：_md_table_to_latex 会吐 \begin{tabularx}{...}{X X X} + \toprule/\midrule/\bottomrule；
    default 模板 preamble 必须引入这两个包，否则含表格的 markdown 编译失败（Plan C 首次 RAG 跑复现）。"""
    from pathlib import Path
    tpl = (Path(__file__).resolve().parent.parent.parent
           / "src" / "math_agent" / "templates" / "paper.tex.j2")
    src = tpl.read_text(encoding="utf-8")
    assert r"\usepackage{tabularx}" in src
    assert r"\usepackage{booktabs}" in src
    # float 提供 [H]，用于把 figure 钉死在源码位置避免最后一张图独占空页
    assert r"\usepackage{float}" in src
    assert r"\begin{figure}[H]" in src


# ---- _pad_math_commands: 分离黏在一起的 LaTeX 命令 ----

def test_pad_math_commands_splits_cdot_stuck_to_letters():
    from math_agent.nodes.latex import _pad_math_commands
    assert _pad_math_commands(r"$\cdotdist_{ij}$") == r"$\cdot\,dist_{ij}$"


def test_pad_math_commands_splits_sum_stuck_to_exp():
    from math_agent.nodes.latex import _pad_math_commands
    assert _pad_math_commands(r"$\sumexp(x)$") == r"$\sum\,exp(x)$"


def test_pad_math_commands_leaves_legit_cdotp_alone():
    """\cdotp 是合法命令名（不是 \cdot+p），不能拆。"""
    from math_agent.nodes.latex import _pad_math_commands
    assert _pad_math_commands(r"$\cdotp$") == r"$\cdotp$"


def test_pad_math_commands_leaves_trailing_subscript_alone():
    from math_agent.nodes.latex import _pad_math_commands
    assert _pad_math_commands(r"$\alpha_i$") == r"$\alpha_i$"
    assert _pad_math_commands(r"$\sum_{i=1}^n a_i$") == r"$\sum_{i=1}^n a_i$"


def test_pad_math_commands_leaves_unknown_macro_alone():
    """写手自定义宏 \myVar 无法判断，保持原样避免误伤。"""
    from math_agent.nodes.latex import _pad_math_commands
    assert _pad_math_commands(r"$\myVar$") == r"$\myVar$"


def test_pad_math_commands_skips_text_span():
    from math_agent.nodes.latex import _pad_math_commands
    # text 里的 \cdotdist 不动（外部 latex 自己处理，且 text 段通常已 escape）
    assert _pad_math_commands("$\cdot$dist") == "$\cdot$dist"


def test_pad_math_commands_covers_equation_block():
    from math_agent.nodes.latex import _pad_math_commands
    src = r"\begin{equation}\cdotdist_{ij}\end{equation}"
    assert _pad_math_commands(src) == r"\begin{equation}\cdot\,dist_{ij}\end{equation}"


def test_pad_math_commands_handles_multiple_in_one_span():
    from math_agent.nodes.latex import _pad_math_commands
    assert _pad_math_commands(r"$a\cdotb\cdotc$") == r"$a\cdot\,b\cdot\,c$"


def test_prepare_section_pipeline_defuses_cdot_dist():
    """端到端：_prepare_section 应把 writer 常见的 `$\cdot$dist_{ij}$`
    最终产出可编译 tex（\cdot 与 dist 之间不粘）。"""
    from math_agent.nodes.latex import _prepare_section
    src = r"目标函数含 $\cdot$dist_{ij}$ 项。"
    out = _prepare_section(src)
    # 关键：不能出现 \cdotdist 这种被 xelatex halt-on-error 的序列
    assert "cdotdist" not in out


# ---- _truncate_caption: 图注在句末/短语边界收尾 ----

def test_truncate_caption_short_input_unchanged():
    from math_agent.nodes.latex import _truncate_caption
    assert _truncate_caption("短标题", max_chars=55) == "短标题"


def test_truncate_caption_cuts_at_period():
    from math_agent.nodes.latex import _truncate_caption
    src = "曲线整体呈下降趋势，表明成本增加。用户满意度下降需要更多调度支持保障"
    # 位置 14 处有句号，落在 [12,20) 窗口内 → 应在句号处收尾
    out = _truncate_caption(src, max_chars=20)
    assert out == "曲线整体呈下降趋势，表明成本增加。", out


def test_truncate_caption_falls_back_to_comma_when_no_period():
    from math_agent.nodes.latex import _truncate_caption
    src = "帕累托前沿图展示，调度方案权衡运营成本与用户满意度分析"
    # 位置 7 有逗号但在 [15*0.6=9, 15) 窗口外；这里 max_chars=15 → [9,15) 无标点 → 硬截
    # 换个例子：max_chars=10，窗口 [6,10) 有逗号在位置 7
    out = _truncate_caption(src, max_chars=10)
    assert out.endswith("，"), out


def test_truncate_caption_hard_cut_when_no_boundary():
    """全是数学符号 / 英文长词、找不到中文标点时退到硬截。"""
    from math_agent.nodes.latex import _truncate_caption
    src = "aaaabbbbccccddddeeeeffffgggghhhhiiiiijjjjkkkk"  # 45 字符
    out = _truncate_caption(src, max_chars=20)
    assert len(out) == 20
