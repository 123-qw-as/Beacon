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
    monkeypatch.setattr("shutil.which", lambda _: None)
    tex = workdir / "main.tex"
    tex.write_text(r"\documentclass{article}\begin{document}x\end{document}", encoding="utf-8")
    res = compile_latex(tex)
    assert not res.success
    assert "xelatex" in res.log.lower()
