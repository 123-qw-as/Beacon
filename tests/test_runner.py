from pathlib import Path

from math_agent.tools.runner import run_python, RunResult


def test_runner_runs_simple_code(workdir):
    res = run_python("print(1 + 1)", workdir=workdir)
    assert res.success
    assert "2" in res.stdout


def test_runner_captures_stderr_on_error(workdir):
    res = run_python("raise ValueError('boom')", workdir=workdir)
    assert not res.success
    assert "boom" in res.stderr


def test_runner_times_out(workdir):
    res = run_python("import time; time.sleep(30)", workdir=workdir, timeout=1)
    assert not res.success
    assert "timeout" in res.stderr.lower() or "killed" in res.stderr.lower()


def test_runner_lists_produced_files(workdir):
    code = "open('out.txt','w').write('x')"
    res = run_python(code, workdir=workdir)
    assert res.success
    assert any(p.endswith("out.txt") for p in res.artifact_paths)


def test_runner_strips_host_env(workdir, monkeypatch):
    """父进程的敏感环境变量不应继承到子进程。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leaked")
    res = run_python(
        "import os; print(os.environ.get('OPENAI_API_KEY', 'MISSING'))",
        workdir=workdir,
    )
    assert res.success
    assert "MISSING" in res.stdout


def test_runner_accepts_relative_workdir(tmp_path, monkeypatch):
    """回归：workdir 传相对路径时不应触发 cwd+script 双重前缀。

    历史 bug：subprocess 切换 cwd 后，python 解释器把 argv 里的相对 script
    解释为相对于新 cwd，导致 "attempt_0/runs/.../attempt_0/_run.py"
    （所有平台都有此行为，不是 Windows 特有）。
    """
    monkeypatch.chdir(tmp_path)
    rel = Path("runs/x/attempt_0")
    res = run_python("print('relpath_ok')", workdir=rel)
    assert res.success, f"stderr={res.stderr!r}"
    assert "relpath_ok" in res.stdout


def test_runner_result_carries_error_kind_on_timeout(workdir):
    res = run_python("import time; time.sleep(30)", workdir=workdir, timeout=1)
    assert not res.success
    assert res.error_kind == "timeout"


def test_runner_result_carries_error_kind_on_runtime(workdir):
    res = run_python("raise ValueError('x')", workdir=workdir)
    assert not res.success
    assert res.error_kind == "runtime"


def test_runner_result_error_kind_empty_on_success(workdir):
    res = run_python("print('hi')", workdir=workdir)
    assert res.success
    assert res.error_kind == ""


def test_extract_numeric_results_baseline_format():
    """RESULT: baseline=X metric1=Y metric2=Z 格式。"""
    from math_agent.tools.runner import extract_numeric_results
    stdout = (
        "一些输出...\n"
        "RESULT: baseline=no_schedule total_cost=1245.3 service_rate=0.82\n"
        "更多输出\n"
        "RESULT: baseline=greedy total_cost=980.0 service_rate=0.91 solve_time=12.5\n"
    )
    results = extract_numeric_results(stdout)
    assert "no_schedule" in results
    assert results["no_schedule"]["total_cost"] == 1245.3
    assert results["no_schedule"]["service_rate"] == 0.82
    assert results["greedy"]["solve_time"] == 12.5


def test_extract_numeric_results_scenario_format():
    """RESULT: scenario=X metric=Y 格式也支持。"""
    from math_agent.tools.runner import extract_numeric_results
    stdout = "RESULT: scenario=high_demand objective=9876.5\n"
    results = extract_numeric_results(stdout)
    assert "high_demand" in results
    assert results["high_demand"]["objective"] == 9876.5


def test_extract_numeric_results_no_result_lines():
    """没有 RESULT 行返回空 dict。"""
    from math_agent.tools.runner import extract_numeric_results
    assert extract_numeric_results("普通输出\n没有结果行") == {}
    assert extract_numeric_results("") == {}


def test_extract_numeric_results_ignores_malformed():
    """RESULT 行缺 key=value 对时跳过。"""
    from math_agent.tools.runner import extract_numeric_results
    stdout = "RESULT: baseline=test\n"  # 没有 metric=value
    results = extract_numeric_results(stdout)
    assert results == {} or results.get("test") == {}


def test_auto_fix_imports_handles_qualified_pyplot_after_matplotlib_import():
    from math_agent.tools.runner import _auto_fix_imports
    code = "import matplotlib\nmatplotlib.pyplot.plot([1, 2])"
    fixed = _auto_fix_imports(code)
    assert "import matplotlib.pyplot" in fixed.splitlines()[:3]


def test_runner_decodes_timeout_output_bytes(mocker, workdir):
    import subprocess
    mocker.patch(
        "math_agent.tools.runner.subprocess.run",
        side_effect=subprocess.TimeoutExpired("python", 1, output=b"partial\xff"),
    )
    res = run_python("print('x')", workdir=workdir, timeout=1)
    assert isinstance(res.stdout, str)
    assert res.stdout.startswith("partial")


# ── Agg 注入 + plt.show 剥离 测试 ──


def test_auto_fix_injects_agg_when_pyplot_already_imported():
    """LLM 已 import pyplot 但未设置 Agg → 仍应注入 matplotlib.use('Agg')。"""
    from math_agent.tools.runner import _auto_fix_imports
    code = (
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3])\n"
        "plt.show()\n"
    )
    fixed = _auto_fix_imports(code)
    assert "matplotlib.use('Agg')" in fixed


def test_auto_fix_skips_agg_when_already_present():
    """LLM 已显式设置 Agg → 不重复注入。"""
    from math_agent.tools.runner import _auto_fix_imports
    code = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3])\n"
    )
    fixed = _auto_fix_imports(code)
    assert fixed.count("matplotlib.use('Agg')") == 1


def test_auto_fix_handles_rcParams_without_pyplot():
    """仅使用 matplotlib.rcParams 也应注入 Agg。"""
    from math_agent.tools.runner import _auto_fix_imports
    code = (
        "import matplotlib\n"
        "matplotlib.rcParams['font.size'] = 10\n"
    )
    fixed = _auto_fix_imports(code)
    assert "matplotlib.use('Agg')" in fixed


def test_strip_dangerous_calls_removes_plt_show():
    """plt.show() 被移除。"""
    from math_agent.tools.runner import _strip_dangerous_calls
    code = (
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3])\n"
        "plt.show()\n"
    )
    stripped = _strip_dangerous_calls(code)
    assert "# [auto-removed:" in stripped
    # 确认原始调用行（以 plt.show 开头的行）被替换
    assert not any(line.strip().startswith("plt.show(") for line in stripped.splitlines())


def test_strip_dangerous_calls_removes_plt_ion():
    """plt.ion() 被移除。"""
    from math_agent.tools.runner import _strip_dangerous_calls
    code = (
        "import matplotlib.pyplot as plt\n"
        "plt.ion()\n"
        "plt.plot([1, 2, 3])\n"
    )
    stripped = _strip_dangerous_calls(code)
    assert "# [auto-removed:" in stripped
    assert not any(line.strip().startswith("plt.ion(") for line in stripped.splitlines())


def test_strip_dangerous_calls_removes_fig_show():
    """fig.show() 被移除（Figure 对象调用 show 同理会阻塞）。"""
    from math_agent.tools.runner import _strip_dangerous_calls
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig = plt.figure()\n"
        "fig.show()\n"
        "plt.savefig('out.png')\n"
    )
    stripped = _strip_dangerous_calls(code)
    assert "# [auto-removed:" in stripped
    assert not any(line.strip().startswith("fig.show(") for line in stripped.splitlines())


def test_strip_dangerous_calls_preserves_savefig():
    """savefig 不受影响。"""
    from math_agent.tools.runner import _strip_dangerous_calls
    code = (
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3])\n"
        "plt.savefig('out.png')\n"
    )
    stripped = _strip_dangerous_calls(code)
    assert "savefig" in stripped
    assert "auto-removed" not in stripped


def test_runner_end_to_end_no_hang_from_plt_show(workdir):
    """端到端：含 plt.show() 的代码在 run_python 中不阻塞，正常完成。"""
    code = (
        "import matplotlib\n"
        "matplotlib.use('TkAgg')  # LLM 可能显式设 GUI backend\n"
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3])\n"
        "plt.show()\n"
        "print('DONE')\n"
    )
    res = run_python(code, workdir=workdir, timeout=10)
    # 即使 LLM 设了 TkAgg，_auto_fix_imports 会前置 Agg（因为
    # already_marker 检查 Agg 而非 pyplot import），随后 _strip_dangerous_calls
    # 移除 plt.show() → 必须不阻塞，不超时
    assert res.success, f"timed out or crashed: {res.stderr[:200]}"
    assert "DONE" in res.stdout
