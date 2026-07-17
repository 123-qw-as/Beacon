import shutil
import pytest
from pathlib import Path

from math_agent.tools.latex_compile import compile_latex


HAS_XELATEX = shutil.which("xelatex") is not None


@pytest.mark.skipif(not HAS_XELATEX, reason="xelatex not installed")
def test_compile_latex_minimal(workdir):
    tex = workdir / "main.tex"
    tex.write_text(r"""
\documentclass{article}
\begin{document}
hello
\end{document}
""", encoding="utf-8")
    res = compile_latex(tex)
    assert res.success
    assert Path(res.pdf_path).exists()


def test_compile_latex_returns_failure_when_xelatex_missing(monkeypatch, workdir):
    monkeypatch.setattr("math_agent.tools.latex_compile._xelatex_candidates", lambda: [])
    monkeypatch.setattr("math_agent.tools.latex_compile._tectonic_candidates", lambda: [])
    tex = workdir / "main.tex"
    tex.write_text(r"\documentclass{article}\begin{document}x\end{document}", encoding="utf-8")
    res = compile_latex(tex)
    assert not res.success
    assert "xelatex" in res.log.lower()
    assert "xelatex" in (workdir / "compile.log").read_text(encoding="utf-8").lower()


def test_latex_result_carries_error_kind_when_missing(monkeypatch, workdir):
    monkeypatch.setattr("math_agent.tools.latex_compile._xelatex_candidates", lambda: [])
    monkeypatch.setattr("math_agent.tools.latex_compile._tectonic_candidates", lambda: [])
    tex = workdir / "main.tex"
    tex.write_text(r"\documentclass{article}\begin{document}x\end{document}", encoding="utf-8")
    res = compile_latex(tex)
    assert res.error_kind == "missing_binary"


def test_compile_latex_does_not_accept_stale_pdf(mocker, workdir):
    tex = workdir / "main.tex"
    tex.write_text(r"\documentclass{article}", encoding="utf-8")
    stale = workdir / "main.pdf"
    stale.write_bytes(b"old pdf")
    mocker.patch("math_agent.tools.latex_compile._xelatex_candidates", return_value=["xelatex"])
    mocker.patch("math_agent.tools.latex_compile._tectonic_candidates", return_value=[])
    mocker.patch(
        "math_agent.tools.latex_compile.subprocess.run",
        return_value=mocker.MagicMock(returncode=1, stdout="Emergency stop", stderr=""),
    )
    res = compile_latex(tex)
    assert not res.success
    assert res.error_kind == "compile"
    assert not stale.exists()
    assert "Emergency stop" in (workdir / "compile.log").read_text(encoding="utf-8")
