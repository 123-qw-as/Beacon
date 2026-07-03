"""subprocess 代码执行器。

**不是隔离沙箱**：LLM 写的代码以当前用户权限运行。本模块只做：
- 工作目录隔离（每次新建临时目录）
- 超时强杀
- 清空环境变量（仅传递最小 PATH / PYTHONPATH / SystemRoot），避免顺手读到 OPENAI_API_KEY 之类。

**为什么不接 docker/firejail/nsjail**（Plan C 复盘决定）：
- 威胁模型：单用户本机跑，配置好的 LLM（DeepSeek / GLM 等）没有主动越狱动机
- 已有防护够用：minimal env 防了 API key 泄露，tempdir + timeout 挡住大部分误伤
- docker 边际成本：镜像+依赖打包、每次调用多 3-5s 冷启、Windows Docker Desktop 资源占用
- **换来的边际安全在本使用场景下近乎为零** → 主动 skip，不做过度工程
- 使用前提：仅在本机可信环境使用。多租户/云部署另议。
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


import re as _re

_RESULT_LINE_RE = _re.compile(
    r"^RESULT:\s*(?:baseline|scenario|method|config)=(\S+)\s+(.+)$",
    _re.MULTILINE,
)
_RESULT_PAIR_RE = _re.compile(r"(\w+)=(-?\d+\.?\d*(?:[eE][+-]?\d+)?)")


def extract_numeric_results(stdout: str) -> dict[str, dict[str, float]]:
    """从 stdout 提取所有 RESULT: 行，返回 {identifier: {metric: value}} 映射。

    支持格式（与 sensitivity.py 的 RESULT: parameter=... 互补，不冲突）：
      RESULT: baseline=no_schedule total_cost=1245.3 service_rate=0.82
      RESULT: scenario=high_demand objective=9876 solve_time=12.5
    """
    results: dict[str, dict[str, float]] = {}
    for m in _RESULT_LINE_RE.finditer(stdout):
        identifier = m.group(1)
        pairs_str = m.group(2)
        metrics: dict[str, float] = {}
        for pm in _RESULT_PAIR_RE.finditer(pairs_str):
            metrics[pm.group(1)] = float(pm.group(2))
        if metrics:
            results[identifier] = metrics
    return results
