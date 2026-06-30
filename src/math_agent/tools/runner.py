"""subprocess 代码执行器。

**不是隔离沙箱**：LLM 写的代码以当前用户权限运行。本模块只做：
- 工作目录隔离（每次新建临时目录）
- 超时强杀
- 清空环境变量（仅传递最小 PATH / PYTHONPATH / SystemRoot），避免顺手读到 OPENAI_API_KEY 之类。

真正的隔离要靠 docker/firejail/nsjail，放 Plan C。
仅在本机可信使用前提下使用本模块。
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    artifact_paths: list[str] = field(default_factory=list)
    error_kind: str = ""  # "" | "timeout" | "runtime"


def _minimal_env() -> dict[str, str]:
    """只透传子进程跑 Python 必需的变量。"""
    keys = [
        "PATH", "PYTHONPATH", "PYTHONHOME", "SystemRoot", "TEMP", "TMP", "LANG", "LC_ALL",
        # matplotlib 等库要求能解析 Path.home()
        "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "HOME",
    ]
    return {k: os.environ[k] for k in keys if k in os.environ}


def run_python(code: str, *, workdir: Path, timeout: int = 60) -> RunResult:
    # 必须 resolve()：subprocess 切换 cwd 后，python 解释器把 argv 里的
    # 相对 script 解释为相对于新 cwd，导致路径双重前缀。
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / "_run.py"
    script.write_text(code, encoding="utf-8")

    before = {p.name for p in workdir.iterdir()}

    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_minimal_env(),
        )
    except subprocess.TimeoutExpired as e:
        return RunResult(
            success=False,
            stdout=e.stdout or "",
            stderr=f"timeout after {timeout}s",
            error_kind="timeout",
        )

    after = {p.name for p in workdir.iterdir()}
    new_files = sorted(after - before - {"_run.py"})
    return RunResult(
        success=proc.returncode == 0,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        artifact_paths=[str(workdir / n) for n in new_files],
        error_kind="" if proc.returncode == 0 else "runtime",
    )
