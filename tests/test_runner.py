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
