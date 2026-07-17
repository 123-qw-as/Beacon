"""Tests for math_agent.nodes.rendering — shared rendering utilities."""
from pathlib import Path

from math_agent.nodes.rendering import (
    _latex_plain_text, _latex_path, _truncate_caption, _curate_code, _curate_stdout,
)


# ---- _latex_plain_text ----

def test_latex_plain_text_escapes_special_chars():
    out = _latex_plain_text("a_b & c#d %e")
    assert r"a\_b" in out
    assert r"\&" in out
    assert r"\#" in out
    assert r"\%" in out


def test_latex_plain_text_none_returns_none():
    assert _latex_plain_text(None) is None


# ---- _latex_path ----

def test_latex_path_converts_backslashes():
    out = _latex_path(r"C:\Users\test\file.png")
    assert "C:/Users/test/file.png" in out
    assert r"\detokenize{" in out


def test_latex_path_handles_closing_brace():
    out = _latex_path("path/to/file}.png")
    assert r"\char125" in out


# ---- _truncate_caption ----

def test_truncate_caption_short_input_unchanged():
    assert _truncate_caption("短标题", max_chars=55) == "短标题"


def test_truncate_caption_cuts_at_period():
    src = "曲线整体呈下降趋势，表明成本增加。用户满意度下降需要更多调度支持保障"
    out = _truncate_caption(src, max_chars=20)
    assert out == "曲线整体呈下降趋势，表明成本增加。", out


def test_truncate_caption_turns_fallback_comma_into_sentence_end():
    src = "帕累托前沿图展示，调度方案权衡运营成本与用户满意度分析"
    out = _truncate_caption(src, max_chars=10)
    assert out.endswith("。"), out
    assert not out.endswith("，"), out


def test_truncate_caption_hard_cut_when_no_boundary():
    """全是数学符号 / 英文长词、找不到中文标点时退到硬截。"""
    src = "aaaabbbbccccddddeeeeffffgggghhhhiiiiijjjjkkkk"  # 45 字符
    out = _truncate_caption(src, max_chars=20)
    assert len(out) == 20


# ---- _curate_code ----

def test_curate_code_short_unchanged():
    code = "print('hello')\nprint('world')"
    assert _curate_code(code) == code


def test_curate_code_truncates_long():
    lines = [f"line_{i}" for i in range(100)]
    code = "\n".join(lines)
    out = _curate_code(code, max_lines=80)
    assert "line_79" in out
    assert "line_80" not in out
    assert "截取前 80 行" in out


# ---- _curate_stdout ----

def test_curate_stdout_extracts_result_lines():
    stdout = "step 1\nstep 2\nRESULT: score=0.95\nmore output\nlast line"
    out = _curate_stdout(stdout)
    assert "RESULT: score=0.95" in out


def test_curate_stdout_empty_returns_empty():
    assert _curate_stdout("") == ""


# ---- Template declarations ----

def test_default_paper_tex_template_declares_tabularx_and_booktabs():
    r"""回归：default 模板 preamble 必须引入 tabularx/booktabs/float。"""
    tpl = (Path(__file__).resolve().parent.parent.parent
           / "src" / "math_agent" / "templates" / "paper.tex.j2")
    src = tpl.read_text(encoding="utf-8")
    assert r"\usepackage{tabularx}" in src
    assert r"\usepackage{booktabs}" in src
    assert r"\usepackage{float}" in src
    assert r"\begin{figure}[H]" in src


def test_tex_templates_break_long_code_lines():
    template_dir = (Path(__file__).resolve().parent.parent.parent
                    / "src" / "math_agent" / "templates")
    for name in ("paper.tex.j2", "gmcm.tex.j2"):
        src = (template_dir / name).read_text(encoding="utf-8")
        assert r"breaklines=true" in src
        assert r"breakatwhitespace=false" in src
        assert r"columns=fullflexible" in src
