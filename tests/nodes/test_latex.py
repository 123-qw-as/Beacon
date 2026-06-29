from math_agent.state import MathModelingState, PaperSections
from math_agent.nodes.latex import (
    latex_node, _wrap_unicode_math, _md_headings_to_latex, _md_inline_code_to_math,
    _wrap_naked_subscripts,
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
