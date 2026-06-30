from math_agent.state import MathModelingState, PaperSections
from math_agent.nodes.latex import (
    latex_node, _wrap_unicode_math, _md_headings_to_latex, _md_inline_code_to_math,
    _wrap_naked_subscripts, _md_bold_to_latex, _md_bullets_to_latex, _md_table_to_latex,
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
    """### basic阶段 → \\subsubsection*{basic阶段}（带 *：不进目录）。"""
    s = "### basic阶段\n内容\n\n## 章节"
    out = _md_headings_to_latex(s)
    assert r"\subsubsection*{basic阶段}" in out
    assert r"\subsection*{章节}" in out
    assert "###" not in out
    assert "## 章节" not in out


def test_md_headings_only_match_line_start():
    """正文里普通 # 不当标题（如 markdown 内联用法、注释）。"""
    s = "成本 #1 是 ## 5\n# title\nbody"
    out = _md_headings_to_latex(s)
    assert "成本 #1 是 ## 5" in out  # 行内 # 不动
    assert r"\section*{title}" in out


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
    assert r"\subsubsection*{basic阶段}" in tex
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
    assert r"\begin{tabular}{|l|l|l|}" in out
    assert r"符号 & 含义 & 单位 \\" in out
    assert r"S_i & 库存 & 辆 \\" in out
    assert r"\end{tabular}" in out
    assert "|------|" not in out
    assert "| 符号 |" not in out
    assert "下文。" in out


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
