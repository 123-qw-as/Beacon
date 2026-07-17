"""LaTeX 编译封装：优先 XeLaTeX，失败后回退到 bundled Tectonic。"""
from __future__ import annotations

import os
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


def _dedupe_existing(candidates: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not item:
            continue
        path = Path(item).expanduser()
        if not path.is_file():
            continue
        norm = os.path.normcase(os.path.abspath(path))
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(str(path))
    return deduped


def _xelatex_candidates() -> list[str]:
    """查找 PATH、显式配置及 Windows 常见 MiKTeX/TeX Live 安装。"""
    candidates: list[str] = []
    for env_name in ("MATH_AGENT_XELATEX", "XELATEX"):
        candidates.append(os.getenv(env_name, "").strip())

    which_path = shutil.which("xelatex")
    if which_path:
        candidates.append(which_path)

    if os.name == "nt":
        local = Path(os.getenv("LOCALAPPDATA", ""))
        program_files = Path(os.getenv("ProgramFiles", r"C:\Program Files"))
        candidates.extend([
            str(local / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64" / "xelatex.exe"),
            str(program_files / "MiKTeX" / "miktex" / "bin" / "x64" / "xelatex.exe"),
        ])
        texlive_root = Path("C:/texlive")
        if texlive_root.is_dir():
            candidates.extend(
                str(path)
                for path in sorted(
                    texlive_root.glob("*/bin/windows/xelatex.exe"), reverse=True
                )
            )

    return _dedupe_existing(candidates)


def _tectonic_candidates() -> list[str]:
    candidates: list[str] = []
    env_path = os.getenv("CODEX_BUNDLED_TECTONIC", "").strip()
    if env_path:
        candidates.append(env_path)

    bundled_root = Path.home() / ".codex" / "plugins" / "cache" / "openai-bundled" / "latex"
    if bundled_root.is_dir():
        for exe in sorted(bundled_root.glob("*/bin/tectonic.exe"), reverse=True):
            candidates.append(str(exe))
        for exe in sorted(bundled_root.glob("*/bin/tectonic"), reverse=True):
            candidates.append(str(exe))

    which_path = shutil.which("tectonic")
    if which_path:
        candidates.append(which_path)

    return _dedupe_existing(candidates)


def _run_subprocess(
    command: list[str], *, cwd: Path, timeout: int, env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
        check=False,
    )


def _write_log(log_path: Path, content: str) -> None:
    log_path.write_text(content, encoding="utf-8", errors="replace")


def _compile_with_xelatex(
    tex_path: Path, *, executable: str, timeout: int
) -> LatexResult:
    workdir = tex_path.parent
    pdf = workdir / f"{tex_path.stem}.pdf"
    log_acc: list[str] = []
    for pass_index in range(2):
        proc = _run_subprocess(
            [
                executable,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                tex_path.name,
            ],
            cwd=workdir,
            timeout=timeout,
        )
        log_acc.append(
            f"[pass {pass_index + 1}] exit={proc.returncode}\n"
            + (proc.stdout or "")
            + "\n"
            + (proc.stderr or "")
        )
        if proc.returncode != 0:
            break
    full_log = "\n".join(log_acc)
    if proc.returncode != 0 or not pdf.exists():
        return LatexResult(
            success=False,
            pdf_path=str(pdf) if pdf.exists() else "",
            log=full_log + ("" if pdf.exists() else "\nno pdf produced"),
            error_kind="compile",
        )
    return LatexResult(success=True, pdf_path=str(pdf), log=full_log)


def _compile_with_tectonic(tex_path: Path, *, timeout: int) -> LatexResult:
    workdir = tex_path.parent
    pdf = workdir / f"{tex_path.stem}.pdf"
    log_parts: list[str] = []
    candidates = _tectonic_candidates()
    if not candidates:
        return LatexResult(success=False, log="tectonic not found", error_kind="missing_binary")

    for candidate in candidates:
        xdg_root = workdir / ".tectonic-runtime"
        (xdg_root / "cache").mkdir(parents=True, exist_ok=True)
        (xdg_root / "config").mkdir(parents=True, exist_ok=True)
        (xdg_root / "data").mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "TECTONIC_UNTRUSTED_MODE": "1",
            "XDG_CACHE_HOME": str((xdg_root / "cache").resolve()),
            "XDG_CONFIG_HOME": str((xdg_root / "config").resolve()),
            "XDG_DATA_HOME": str((xdg_root / "data").resolve()),
        }
        try:
            proc = _run_subprocess(
                [
                    candidate,
                    "-X", "compile",
                    "--outdir", str(workdir),
                    "--outfmt", "pdf",
                    "--print",
                    "--untrusted",
                    tex_path.name,
                ],
                cwd=workdir,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            log_parts.append(f"[tectonic:{candidate}] timeout after {timeout}s: {exc}")
            continue

        full_log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        log_parts.append(f"[tectonic:{candidate}] exit={proc.returncode}\n{full_log}")
        if proc.returncode == 0 and pdf.exists():
            return LatexResult(success=True, pdf_path=str(pdf), log="\n\n".join(log_parts))

    return LatexResult(success=False, log="\n\n".join(log_parts), error_kind="compile")


def compile_latex(tex_path: str | Path, *, timeout: int = 120) -> LatexResult:
    tex_path = Path(tex_path)
    workdir = tex_path.parent
    pdf = workdir / (tex_path.stem + ".pdf")
    log_path = workdir / "compile.log"
    pdf.unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)

    xelatex_paths = _xelatex_candidates()
    logs: list[str] = []
    saw_timeout = False

    if xelatex_paths:
        for xelatex_path in xelatex_paths:
            pdf.unlink(missing_ok=True)
            try:
                res = _compile_with_xelatex(
                    tex_path, executable=xelatex_path, timeout=timeout
                )
                logs.append(f"[xelatex:{xelatex_path}]\n{res.log}")
                _write_log(log_path, "\n\n".join(logs))
                if res.success:
                    return LatexResult(
                        success=True,
                        pdf_path=res.pdf_path,
                        log="\n\n".join(logs),
                    )
            except subprocess.TimeoutExpired as exc:
                saw_timeout = True
                logs.append(
                    f"[xelatex:{xelatex_path}] timeout after {timeout}s: {exc}"
                )
    else:
        logs.append("[xelatex] xelatex not found in PATH or known install locations")

    pdf.unlink(missing_ok=True)
    tectonic_res = _compile_with_tectonic(tex_path, timeout=timeout)
    logs.append("[tectonic]\n" + tectonic_res.log)
    _write_log(log_path, "\n\n".join(logs))
    if tectonic_res.success:
        return LatexResult(success=True, pdf_path=tectonic_res.pdf_path, log="\n\n".join(logs))

    if not xelatex_paths and tectonic_res.error_kind == "missing_binary":
        return LatexResult(success=False, log="\n\n".join(logs), error_kind="missing_binary")
    if saw_timeout and tectonic_res.error_kind == "missing_binary":
        return LatexResult(success=False, log="\n\n".join(logs), error_kind="timeout")
    if xelatex_paths and tectonic_res.error_kind == "missing_binary":
        return LatexResult(success=False, log="\n\n".join(logs), error_kind="compile")
    return LatexResult(
        success=False,
        pdf_path=tectonic_res.pdf_path,
        log="\n\n".join(logs),
        error_kind=tectonic_res.error_kind or "compile",
    )
