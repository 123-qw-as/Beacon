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
    error_kind: str = ""  # "" | "missing_binary" | "compile" | "timeout"


def compile_latex(tex_path: str | Path, *, timeout: int = 120) -> LatexResult:
    tex_path = Path(tex_path)
    workdir = tex_path.parent
    pdf = workdir / (tex_path.stem + ".pdf")
    log_path = workdir / "compile.log"
    # 避免上一次运行残留的 PDF 让本次失败被误判为成功或继续展示。
    pdf.unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)
    if shutil.which("xelatex") is None:
        log_path.write_text("xelatex not found on PATH", encoding="utf-8")
        return LatexResult(
            success=False, log="xelatex not found on PATH",
            error_kind="missing_binary",
        )
    try:
        # 跑两遍以解决交叉引用 / TOC。
        # 不用 -halt-on-error：writer 生成的 LaTeX 偶有非致命错误（如
        # 'Extra alignment tab' — tabularx X 列宽计算与中文宽字符冲突），
        # xelatex 能自动修复（把多余 & 改成 \cr）继续排版出完整 PDF。
        # -halt-on-error 会让这些可恢复错误直接停在第 1 页、0 pages of output。
        # 改用 nonstopmode + PDF 是否生成 + log 里 '!' 错误数双条件判 success。
        log_acc = []
        for _ in range(2):
            proc = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", tex_path.name],
                cwd=workdir, capture_output=True, text=True, timeout=timeout,
            )
            log_acc.append((proc.stdout or "") + "\n" + (proc.stderr or ""))

        full_log = "\n".join(log_acc)
        log_path.write_text(full_log, encoding="utf-8", errors="replace")
        # success 双条件：PDF 存在 + log 里没有致命错误（以 '! ' 开头的行）。
        # 'Extra alignment tab has been changed to \cr' 和 'Missing $ inserted'
        # 是 xelatex 已自动修复的非致命错误——xelatex 自己说了 "has been changed" /
        # "inserted"，PDF 内容完整，不算致命。
        _RECOVERABLE = ("has been changed to", "has been converted", "Missing $ inserted")
        fatal_errors = [
            l for l in full_log.split("\n")
            if l.startswith("! ") and not any(r in l for r in _RECOVERABLE)
        ]
        if not pdf.exists():
            return LatexResult(
                success=False, log=full_log + "\nno pdf produced",
                error_kind="compile",
            )
        if fatal_errors:
            # PDF 生成了但有致命错误——标记为 compile 但仍返回 pdf_path
            # 让用户能看到 PDF（内容可能有小瑕疵）+ 知道有错误需修
            return LatexResult(
                success=False, pdf_path=str(pdf),
                log=full_log + "\nfatal errors: " + "; ".join(fatal_errors[:3]),
                error_kind="compile",
            )
        return LatexResult(success=True, pdf_path=str(pdf), log=full_log)
    except subprocess.TimeoutExpired as e:
        log_path.write_text(f"timeout after {timeout}s: {e}", encoding="utf-8")
        return LatexResult(
            success=False, log=f"timeout after {timeout}s: {e}",
            error_kind="timeout",
        )
