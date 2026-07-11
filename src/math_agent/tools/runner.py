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
    env = {k: os.environ[k] for k in keys if k in os.environ}
    # 强制子进程用 UTF-8 输出，避免 Windows 默认 GBK 编码导致 subprocess 解码崩溃
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


import re as _re

# LLM 生成代码常缺的 import。扫描代码体，缺哪个补哪个。
# ponytail: 只补最常见 5 个库；复杂缺漏靠 retry 兜底，不做全量 ast 分析。
_IMPORT_FIXES: list[tuple[str, str, str]] = [
    # (检测标志, 缺时补的行, already_marker — 若已在代码中存在则跳过)
    #
    # matplotlib Agg 后端：用 "matplotlib.use('Agg')" 做 already_marker，
    # 而不是 "import matplotlib.pyplot"——旧逻辑会因为 LLM 已 import pyplot
    # 就跳过 Agg 注入，导致 plt.show() 在子进程中尝试弹 GUI 窗口阻塞。
    # 条目 2、3 的 fix_line 仍保留 import matplotlib.pyplot，
    # 防止 LLM 只写了 plt.plot() 却忘了 import pyplot 的遗漏场景。
    ("matplotlib.rcParams", "import matplotlib; matplotlib.use('Agg')", "matplotlib.use('Agg')"),
    ("matplotlib\\.pyplot", "import matplotlib; matplotlib.use('Agg')\nimport matplotlib.pyplot", "matplotlib.use('Agg')"),
    ("plt\\.", "import matplotlib; matplotlib.use('Agg')\nimport matplotlib.pyplot as plt", "matplotlib.use('Agg')"),
    ("np\\.", "import numpy as np", "import numpy"),
    ("pd\\.", "import pandas as pd", "import pandas"),
]


def _auto_fix_imports(code: str) -> str:
    """检测 LLM 生成代码缺的 import 并自动补上。

    只补常见库（matplotlib/numpy/pandas），避免 NameError 导致整段代码白跑。
    已有对应 import 的不重复补。
    """
    prefix_lines: list[str] = []
    for pattern, fix_line, already_marker in _IMPORT_FIXES:
        if _re.search(pattern, code) and already_marker not in code:
            if fix_line not in prefix_lines:
                prefix_lines.append(fix_line)
    if not prefix_lines:
        return code
    return "\n".join(prefix_lines) + "\n" + code


# 危险调用：在非交互子进程中会弹 GUI 窗口阻塞。Agg 后端让它们变 no-op，
# 但显式移除更安全——万一 matplotlib 内部在 Agg 下仍走 event loop 呢。
# 覆盖 plt.show() / plt.ion() / fig.show()（fig 是 Figure 对象）。
_DANGEROUS_CALL_RE = _re.compile(
    r'^\s*(?:plt|fig)\.(?:show|ion)\s*\(.*\)\s*(?:#.*)?$',
    _re.MULTILINE,
)


def _strip_dangerous_calls(code: str) -> str:
    """移除 plt.show() / plt.ion() / fig.show() 等会阻塞子进程的 GUI 调用。"""
    return _DANGEROUS_CALL_RE.sub(
        '# [auto-removed: plt.show()/ion()/fig.show() blocks subprocess]',
        code,
    )


def run_python(code: str, *, workdir: Path, timeout: int = 60) -> RunResult:
    # 必须 resolve()：subprocess 切换 cwd 后，python 解释器把 argv 里的
    # 相对 script 解释为相对于新 cwd，导致路径双重前缀。
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    code = _auto_fix_imports(code)   # 补缺的 import + matplotlib.use('Agg')
    code = _strip_dangerous_calls(code)  # 移除 plt.show()/ion()/fig.show()，防止 GUI 阻塞
    script = workdir / "_run.py"
    script.write_text(code, encoding="utf-8")

    before = {p.name for p in workdir.iterdir()}

    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=workdir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",   # 子进程可能输出 GBK/混合编码，replace 避免解码崩溃
            timeout=timeout,
            env=_minimal_env(),
        )
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        return RunResult(
            success=False,
            stdout=stdout,
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
