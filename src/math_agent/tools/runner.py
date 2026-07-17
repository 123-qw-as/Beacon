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

import ast
import json
import os
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil


@dataclass
class RunResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    artifact_paths: list[str] = field(default_factory=list)
    read_paths: list[str] = field(default_factory=list)
    error_kind: str = ""  # "" | "timeout" | "runtime" | "resource"


_DATA_LOAD_CALLS = {
    "open", "read_csv", "read_excel", "excelfile", "load_workbook",
    "load", "loadtxt", "genfromtxt", "read_text", "read_bytes",
}
_DATA_DISCOVERY_CALLS = {"glob", "rglob", "iterdir", "listdir", "walk"}


def validate_code_data_usage(code: str, filenames: list[str]) -> tuple[bool, str]:
    """验证数值证据脚本确实从附件读取数据，而不是只打印硬编码结果。

    这是轻量级数据血缘门禁：AST 中必须同时出现读取调用，以及真实附件名/扩展名
    或目录枚举调用。它不声称证明算法正确，但能拒绝本次真实运行暴露出的纯硬编码脚本。
    """
    if not filenames:
        return True, ""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"主数值脚本无法解析，不能验证数据血缘: {exc}"

    calls: set[str] = set()
    string_literals: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.add(func.id.casefold())
            elif isinstance(func, ast.Attribute):
                calls.add(func.attr.casefold())
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            string_literals.append(node.value.replace("\\", "/").casefold())

    has_loader = bool(calls & _DATA_LOAD_CALLS)
    has_discovery = bool(calls & _DATA_DISCOVERY_CALLS)
    normalized_names = [Path(name).name.casefold() for name in filenames if name]
    suffixes = {Path(name).suffix.casefold() for name in filenames if Path(name).suffix}
    has_named_source = any(
        filename in literal or (
            Path(filename).stem and Path(filename).stem in literal
            and Path(filename).suffix in suffixes
        )
        for filename in normalized_names
        for literal in string_literals
    )
    if not has_loader:
        return False, "主数值脚本未调用任何数据读取函数，结果疑似硬编码"
    if not (has_named_source or has_discovery):
        return False, "主数值脚本的数据读取未关联真实附件名或附件目录枚举"
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "print":
            continue
        for arg in node.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and "RESULT:" in arg.value
                and _RESULT_PAIR_RE.search(arg.value)
            ):
                return False, "RESULT 使用常量数值字符串，结果疑似硬编码"
    return True, ""


def _input_audit_prefix(expected_paths: list[Path], audit_log: Path) -> str:
    """在生成脚本最前注册只读审计钩子，记录真实打开的声明附件。"""
    normalized = sorted({str(path.resolve()).casefold() for path in expected_paths})
    if not normalized:
        return ""
    return (
        "import sys as __beacon_sys, os as __beacon_os, json as __beacon_json\n"
        f"__beacon_expected = set({normalized!r})\n"
        f"__beacon_audit_log = {str(audit_log)!r}\n"
        "__beacon_audit_busy = False\n"
        "def __beacon_audit(event, args):\n"
        "    global __beacon_audit_busy\n"
        "    if event != 'open' or __beacon_audit_busy or not args:\n"
        "        return\n"
        "    try:\n"
        "        path = __beacon_os.path.abspath(__beacon_os.fspath(args[0])).casefold()\n"
        "    except (TypeError, ValueError):\n"
        "        return\n"
        "    if path not in __beacon_expected:\n"
        "        return\n"
        "    __beacon_audit_busy = True\n"
        "    try:\n"
        "        with open(__beacon_audit_log, 'a', encoding='utf-8') as fh:\n"
        "            fh.write(__beacon_json.dumps(path, ensure_ascii=False) + '\\n')\n"
        "    finally:\n"
        "        __beacon_audit_busy = False\n"
        "__beacon_sys.addaudithook(__beacon_audit)\n"
    )


def _read_audited_paths(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return []
    values: list[str] = []
    for line in lines:
        try:
            value = str(json.loads(line))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if value not in values:
            values.append(value)
    return values


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


def _normalize_common_import_mistakes(code: str) -> str:
    """修正常见但确定无歧义的生成式 import 拼写错误。"""
    return _re.sub(
        r"^\s*import\s+matplotlib\.rcparams\s+as\s+(\w+)\s*$",
        r"from matplotlib import rcParams as \1",
        code,
        flags=_re.MULTILINE,
    )


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


def _normalize_escaped_source(code: str) -> str:
    """将被双重转义成字面量 ``\\n``/``\\t`` 的整段源码还原回真实换行。

    仅在原始源码编译失败、且表现出典型的“line continuation character”语法错误时尝试。
    这样可以修复 LLM 把整段 Python 代码塞进 JSON 字符串后再次转义的情况，同时避免
    误伤本来语法正确、只是字符串字面量里包含 ``\\n`` 的脚本。
    """
    if "\\n" not in code and "\\t" not in code and "\\r" not in code:
        return code
    try:
        compile(code, "<llm_code>", "exec")
        return code
    except SyntaxError as exc:
        msg = str(exc).lower()
        if "line continuation character" not in msg and "unexpected character after line continuation" not in msg:
            return code
    normalized = code.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    try:
        compile(normalized, "<llm_code_normalized>", "exec")
    except SyntaxError:
        return code
    return normalized


def _terminate_process_tree(pid: int, *, grace: float = 2.0) -> None:
    """终止 pid 及全部后代；Windows venv launcher 会额外派生真实解释器。"""
    try:
        parent = psutil.Process(pid)
        processes = [*parent.children(recursive=True), parent]
    except psutil.NoSuchProcess:
        return
    for process in reversed(processes):
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(processes, timeout=grace)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if alive:
        psutil.wait_procs(alive, timeout=grace)


def _process_tree_rss(pid: int) -> int:
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def run_python(
    code: str,
    *,
    workdir: Path,
    timeout: int = 60,
    memory_limit_mb: int | None = None,
    expected_input_paths: list[Path] | None = None,
) -> RunResult:
    # 必须 resolve()：subprocess 切换 cwd 后，python 解释器把 argv 里的
    # 相对 script 解释为相对于新 cwd，导致路径双重前缀。
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    code = _normalize_escaped_source(code)  # 修复整段源码被双重转义成字面量 \n/\t
    code = _normalize_common_import_mistakes(code)
    code = _auto_fix_imports(code)   # 补缺的 import + matplotlib.use('Agg')
    code = _strip_dangerous_calls(code)  # 移除 plt.show()/ion()/fig.show()，防止 GUI 阻塞
    script = workdir / "_run.py"
    audit_log = workdir / "_input_reads.jsonl"
    audit_log.unlink(missing_ok=True)
    expected_paths = [Path(path) for path in (expected_input_paths or [])]
    script.write_text(_input_audit_prefix(expected_paths, audit_log) + code, encoding="utf-8")

    before = {p.name for p in workdir.iterdir()}

    if memory_limit_mb is None:
        try:
            memory_limit_mb = int(os.getenv("MATH_AGENT_CODE_MEMORY_LIMIT_MB", "2048"))
        except ValueError:
            memory_limit_mb = 2048
    memory_limit_bytes = max(0, memory_limit_mb) * 1024 * 1024
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_minimal_env(),
    )
    started = time.monotonic()
    stdout = ""
    stderr = ""
    failure_kind = ""
    failure_message = ""
    while True:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            failure_kind = "timeout"
            failure_message = f"timeout after {timeout}s; process tree terminated"
            break
        try:
            stdout, stderr = proc.communicate(timeout=min(0.5, remaining))
            break
        except subprocess.TimeoutExpired:
            if memory_limit_bytes and _process_tree_rss(proc.pid) > memory_limit_bytes:
                used_mb = _process_tree_rss(proc.pid) / (1024 * 1024)
                failure_kind = "resource"
                failure_message = (
                    f"memory limit exceeded: {used_mb:.0f} MB > {memory_limit_mb} MB; "
                    "process tree terminated"
                )
                break

    if failure_kind:
        _terminate_process_tree(proc.pid)
        try:
            tail_out, tail_err = proc.communicate(timeout=2)
            stdout = tail_out or stdout
            stderr = tail_err or stderr
        except (subprocess.TimeoutExpired, OSError):
            pass
        return RunResult(
            success=False,
            stdout=stdout or "",
            stderr="\n".join(part for part in (stderr, failure_message) if part),
            error_kind=failure_kind,
            read_paths=_read_audited_paths(audit_log),
        )

    after = {p.name for p in workdir.iterdir()}
    new_files = sorted(after - before - {"_run.py", "_input_reads.jsonl"})
    return RunResult(
        success=proc.returncode == 0,
        stdout=stdout or "",
        stderr=stderr or "",
        artifact_paths=[str(workdir / n) for n in new_files],
        read_paths=_read_audited_paths(audit_log),
        error_kind="" if proc.returncode == 0 else "runtime",
    )


import re as _re

_RESULT_LINE_RE = _re.compile(
    r"^RESULT:\s*(?:baseline|scenario|method|config)=(\S+)\s+(.+)$",
    _re.MULTILINE,
)
_RESULT_PAIR_RE = _re.compile(r"(\w+)=(-?\d+\.?\d*(?:[eE][+-]?\d+)?)")

_OUTPUT_FAILURE_RE = _re.compile(
    r"^\s*(?:"
    r"traceback\s*\(most recent call last\)\s*:|"
    r"data\s+(?:loading|load|reading|read)\s+failed\b|"
    r"error(?:\s+during\s+execution)?\s*:|"
    r"exception\s*:|"
    r"failed\s+to\s+(?:load|read|execute|solve)\b"
    r")",
    _re.IGNORECASE | _re.MULTILINE,
)
_NONFINITE_OUTPUT_RE = _re.compile(
    r"(?<![A-Za-z0-9_])(?:nan|[+-]?inf)(?![A-Za-z0-9_])",
    _re.IGNORECASE,
)
_RATE_WORDS = ("rate", "ratio", "share", "proportion", "utilization")
_COUNT_KEYS = {
    "vehicle_count", "veh_count", "vehicles", "vehicle_num", "fleet_size",
    "truck_count", "route_count", "used_vehicles", "num_vehicles",
}
_NONNEGATIVE_WORDS = (
    "cost", "count", "vehicle", "carbon", "emission", "distance", "time",
    "demand", "load", "energy", "fuel", "penalty", "objective",
)


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


def detect_output_failure(stdout: str, stderr: str = "") -> str:
    """识别被生成代码吞掉、但写进 stdout/stderr 的显式失败。

    子进程退出码只能说明 Python 解释器正常结束；LLM 代码常见 ``except`` 后打印
    ``Data loading failed`` 再返回 0。这里把这种输出恢复为可重试的失败信号。
    """
    for stream in (stdout or "", stderr or ""):
        match = _OUTPUT_FAILURE_RE.search(stream)
        if match:
            line = stream[match.start():].splitlines()[0].strip()
            return line[:500]
        nonfinite = _NONFINITE_OUTPUT_RE.search(stream)
        if nonfinite:
            line = stream[nonfinite.start():].splitlines()[0].strip()
            return f"输出包含非有限数值：{line[:450]}"
    return ""


def validate_numeric_results(
    stdout: str,
    *,
    stderr: str = "",
    require_result: bool = False,
    expected_identifier: str | None = None,
    max_entity_count: int | None = None,
    min_metrics_per_result: int = 0,
) -> tuple[bool, str, dict[str, dict[str, float]]]:
    """验证生成代码的 RESULT 协议及最基本的数值合理性。

    这是跨 coder、表格、writer、critic 共享的确定性证据门禁。它不替代题目专属
    求解校验，但能阻止“打印错误后 exit 0”、全零伪结果、比例越界和实体数量明显
    超出输入规模等已观测故障进入论文。
    """
    failure = detect_output_failure(stdout, stderr)
    parsed = extract_numeric_results(stdout)
    if failure:
        return False, f"输出声明执行失败：{failure}", parsed
    if require_result and not parsed:
        return False, "stdout 缺少可解析的 RESULT 数值行", parsed
    if expected_identifier is not None and expected_identifier not in parsed:
        return (
            False,
            f"RESULT 标识不匹配：要求 {expected_identifier}，实际 {sorted(parsed)}",
            parsed,
        )

    for identifier, metrics in parsed.items():
        if len(metrics) < min_metrics_per_result:
            return (
                False,
                f"RESULT {identifier} 仅含 {len(metrics)} 个指标，至少需要 {min_metrics_per_result} 个",
                parsed,
            )
        if metrics and all(abs(value) <= 1e-12 for value in metrics.values()):
            return False, f"RESULT {identifier} 的全部指标均为 0", parsed
        for name, value in metrics.items():
            key = name.lower()
            if not math.isfinite(value):
                return False, f"指标 {name} 不是有限数值", parsed
            if any(word in key for word in _NONNEGATIVE_WORDS) and value < 0:
                return False, f"指标 {name} 不应为负数：{value}", parsed
            if any(key == word or key.endswith(f"_{word}") for word in _RATE_WORDS):
                if not 0 <= value <= 1:
                    return False, f"比例指标 {name} 必须位于 [0, 1]：{value}", parsed
            if max_entity_count and key in _COUNT_KEYS:
                if value > max_entity_count:
                    return (
                        False,
                        f"实体数量指标 {name}={value:g} 超过输入规模上界 {max_entity_count}",
                        parsed,
                    )
                if abs(value - round(value)) > 1e-9:
                    return False, f"计数指标 {name} 必须为整数：{value}", parsed
    return True, "", parsed


def infer_entity_upper_bound(data_files) -> int | None:
    """从 DataFileInfo 或等价字典摘要中提取保守的最大行数。"""
    rows: list[int] = []
    for info in data_files or []:
        summary = getattr(info, "summary", None)
        if summary is None and isinstance(info, dict):
            summary = info.get("summary")
        summary = summary or {}
        direct = summary.get("rows")
        if isinstance(direct, (int, float)) and direct > 0:
            rows.append(int(direct))
        sheets = summary.get("sheets", [])
        if isinstance(sheets, list):
            for sheet in sheets:
                count = sheet.get("rows") if isinstance(sheet, dict) else getattr(sheet, "rows", None)
                if isinstance(count, (int, float)) and count > 0:
                    rows.append(int(count))
    return max(rows) if rows else None


def extract_valid_result_lines(
    stdout: str,
    *,
    stderr: str = "",
    expected_identifier: str | None = None,
    max_entity_count: int | None = None,
) -> list[str]:
    """只返回通过统一证据门禁的 RESULT 原始行。"""
    valid, _, parsed = validate_numeric_results(
        stdout,
        stderr=stderr,
        require_result=True,
        expected_identifier=expected_identifier,
        max_entity_count=max_entity_count,
    )
    if not valid or not parsed:
        return []
    return [match.group(0).strip() for match in _RESULT_LINE_RE.finditer(stdout or "")]
