"""xelatex 子进程封装。失败时返回结构化结果，不抛异常。"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LatexResult:
    success: bool
    pdf_path: str = ""
    log: str = ""


def compile_latex(tex_path: str | Path, *, timeout: int = 120) -> LatexResult:
    tex_path = Path(tex_path)
    if shutil.which("xelatex") is None:
        return LatexResult(success=False, log="xelatex not found on PATH")

    workdir = tex_path.parent
    try:
        # 跑两遍以解决交叉引用 / TOC
        log_acc = []
        for _ in range(2):
            proc = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                cwd=workdir, capture_output=True, text=True, timeout=timeout,
            )
            # subprocess.run may leave stdout/stderr as None when stream is closed early;
            # coerce to '' to avoid TypeError in concat (same fix as tools/runner.py).
            log_acc.append((proc.stdout or "") + "\n" + (proc.stderr or ""))
            if proc.returncode != 0:
                return LatexResult(success=False, log="\n".join(log_acc))
        pdf = workdir / (tex_path.stem + ".pdf")
        if not pdf.exists():
            return LatexResult(success=False, log="\n".join(log_acc) + "\nno pdf produced")
        return LatexResult(success=True, pdf_path=str(pdf), log="\n".join(log_acc))
    except subprocess.TimeoutExpired as e:
        return LatexResult(success=False, log=f"timeout after {timeout}s: {e}")
