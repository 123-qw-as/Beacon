"""Tests for math_agent.nodes.latex_node — latex_node() orchestration."""
import re

from math_agent.state import MathModelingState, PaperSections, CodeArtifact, FigureArtifact
from math_agent.nodes.latex_node import latex_node


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
        "math_agent.nodes.latex_node.compile_latex",
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
        "math_agent.nodes.latex_node.compile_latex",
        return_value=type("R", (), {"success": True, "pdf_path": str(pdf), "log": ""})(),
    )
    s = _state(workdir)
    delta = latex_node(s)
    assert (workdir / "paper.tex").exists()
    assert delta == {} or "errors" not in delta


def test_latex_node_processes_markdown_in_paper(mocker, workdir):
    """端到端：paper 段含 ### 三级标题与 unicode 数学，渲到 tex 后两者都被处理。"""
    mocker.patch(
        "math_agent.nodes.latex_node.compile_latex",
        return_value=type("R", (), {"success": True, "pdf_path": "", "log": ""})(),
    )
    s = _state(workdir)
    s.paper.model_section = "### basic阶段\n参数 σ 是常数。"
    latex_node(s)
    tex = (workdir / "paper.tex").read_text(encoding="utf-8")
    assert r"\subsubsection{basic阶段}" in tex
    assert r"$\sigma$" in tex
    assert "###" not in tex


def test_latex_node_title_only_first_line(mocker, workdir):
    """title 只取 problem 第一行，避免长问题描述塞进标题。"""
    mocker.patch(
        "math_agent.nodes.latex_node.compile_latex",
        return_value=type("R", (), {"success": True, "pdf_path": "", "log": ""})(),
    )
    s = _state(workdir)
    s.problem = "城市共享单车调度优化\n建立模型预测各站点未来一小时的需求量。\n在不超过 100 辆运力的前提下，设计最优调度方案。"
    latex_node(s)
    tex = (workdir / "paper.tex").read_text(encoding="utf-8")
    assert r"\title{城市共享单车调度优化}" in tex
    assert r"建立模型预测各站点未来" not in tex.split(r"\title{")[1].split("}")[0]


def test_latex_node_figure_caption_truncated(mocker, workdir):
    """figure caption 截到 55 字，避免标题占两行。"""
    mocker.patch(
        "math_agent.nodes.latex_node.compile_latex",
        return_value=type("R", (), {"success": True, "pdf_path": "", "log": ""})(),
    )
    s = _state(workdir)
    s.figures.append(FigureArtifact(
        path="a.png", purpose="test",
        caption="这是一段非常长的图说" * 10,
        analysis="完整分析" * 20,
    ))
    latex_node(s)
    tex = (workdir / "paper.tex").read_text(encoding="utf-8")
    m = re.search(r"\\caption\{([^}]+)\}", tex)
    assert m and len(m.group(1)) <= 56


def test_latex_node_escapes_filename_like_figure_caption(mocker, workdir):
    mocker.patch(
        "math_agent.nodes.latex_node.compile_latex",
        return_value=type("R", (), {"success": True, "pdf_path": "", "log": ""})(),
    )
    state = _state(workdir)
    state.figures.append(FigureArtifact(
        path="figure.png", purpose="cost_rate & result",
        caption="sensitivity_capacity.png & comparison", analysis="分析",
    ))
    latex_node(state)
    tex = (workdir / "paper.tex").read_text(encoding="utf-8")
    assert r"\caption{sensitivity\_capacity.png \& comparison}" in tex


def test_latex_node_escapes_title_without_html_entities(mocker, workdir):
    mocker.patch(
        "math_agent.nodes.latex_node.compile_latex",
        return_value=type("R", (), {"success": True, "pdf_path": "", "log": ""})(),
    )
    s = _state(workdir)
    s.problem = "成本 A&B_{raw} 与参数 α\n正文"
    latex_node(s)
    tex = (workdir / "paper.tex").read_text(encoding="utf-8")
    assert r"\title{成本 A\&B\_\{raw\} 与参数 $\alpha$}" in tex
    assert "&amp;" not in tex


def test_latex_node_escapes_code_artifact_purpose(mocker, workdir):
    mocker.patch(
        "math_agent.nodes.latex_node.compile_latex",
        return_value=type("R", (), {"success": True, "pdf_path": "", "log": ""})(),
    )
    s = _state(workdir)
    s.code_artifacts.append(CodeArtifact(
        purpose="cost_rate & result", code="print(1)", stdout="1", success=True,
    ))
    latex_node(s)
    tex = (workdir / "paper.tex").read_text(encoding="utf-8")
    assert r"\subsection{$cost_rate$ \& result}" in tex


def test_gmcm_references_are_rendered_as_bibitems(mocker, workdir):
    mocker.patch(
        "math_agent.nodes.latex_node.compile_latex",
        return_value=type("R", (), {"success": True, "pdf_path": "", "log": ""})(),
    )
    s = _state(workdir)
    s.latex_template = "gmcm"
    s.paper.references = "[1] Author A. Title A.\n[2] Author B. Title B."
    latex_node(s)
    tex = (workdir / "paper.tex").read_text(encoding="utf-8")
    assert r"\bibitem{ref1} Author A. Title A." in tex
    assert r"\bibitem{ref2} Author B. Title B." in tex
    assert "\\schoolname{None}" not in tex
    assert r"\schoolname{XX大学}" in tex


def test_gmcm_metadata_is_latex_escaped(mocker, workdir):
    mocker.patch(
        "math_agent.nodes.latex_node.compile_latex",
        return_value=type("R", (), {"success": True, "pdf_path": "", "log": ""})(),
    )
    s = _state(workdir)
    s.latex_template = "gmcm"
    s.school = "A&B_University"
    s.team_id = "T#1"
    s.members = "张_三,李&四,王%五"
    s.paper.keywords = "优化_调度 & 鲁棒"
    latex_node(s)
    tex = (workdir / "paper.tex").read_text(encoding="utf-8")
    assert r"\schoolname{A\&B\_University}" in tex
    assert r"\baominghao{T\#1}" in tex
    assert r"\membera{张\_三}" in tex
    assert r"\keywords{优化\_调度 \& 鲁棒}" in tex
