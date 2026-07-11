from pathlib import Path
from math_agent.state import MathModelingState, ModelVersion, Assumption
from math_agent.nodes.sensitivity import (
    sensitivity_node, SensitivityPlan, SensitivityCode, Interpretations,
)


def _ok_state(workdir):
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.assumptions.append(Assumption(statement="lambda 是常数", rationale="r", sensitivity_relevant=True))
    s.model_versions.append(ModelVersion(stage="final", description="d"*200, equations=["x=lambda"]))
    return s


def test_sensitivity_runs_plan_then_code_then_interpret(mocker, workdir):
    plan = SensitivityPlan(runs=[{"parameter": "lambda", "values": [0.5, 1, 1.5, 2, 2.5],
                                  "metric": "y", "rationale": "核心参数"}])
    code = SensitivityCode(code=(
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "vals=[0.5,1,1.5,2,2.5]; res=[v*2 for v in vals]\n"
        "plt.plot(vals,res); plt.savefig('lambda.png')\n"
        "print(f'RESULT: parameter=lambda values={vals} results={res}')\n"
    ))
    interp = Interpretations(interpretations=["参数 lambda 上升时 y 线性增长，敏感度中等。"])
    mocker.patch("math_agent.nodes.sensitivity.complete", side_effect=[plan, code, interp])

    delta = sensitivity_node(_ok_state(workdir))
    assert len(delta["sensitivity_runs"]) == 1
    run = delta["sensitivity_runs"][0]
    assert run.parameter == "lambda"
    assert run.results == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert run.interpretation.startswith("参数 lambda")
    assert run.figure_path and Path(run.figure_path).exists()


def test_sensitivity_records_error_when_no_final_model(mocker, workdir):
    s = MathModelingState(problem="p", output_dir=str(workdir))
    delta = sensitivity_node(s)
    assert delta["errors"]
    assert delta.get("sensitivity_runs", []) == []


def test_sensitivity_falls_back_when_code_fails(mocker, workdir):
    plan = SensitivityPlan(runs=[{"parameter": "lambda", "values": [1, 2, 3, 4, 5],
                                  "metric": "y", "rationale": "x"}])
    bad = SensitivityCode(code="raise RuntimeError('x')")
    # retry: plan + 2 次失败的 code = 3 次 complete 调用
    mocker.patch("math_agent.nodes.sensitivity.complete", side_effect=[plan, bad, bad])
    delta = sensitivity_node(_ok_state(workdir))
    assert delta["errors"]
    assert delta.get("sensitivity_runs", []) == []


def test_sensitivity_parse_handles_numpy_repr():
    """RESULT 行里 np.float64(...) 形式的数字不应让节点崩（eval_v6_1 实测）。"""
    from math_agent.nodes.sensitivity import _parse_results
    line = ("RESULT: parameter=lambda values=[0.5, 1.0, 1.5] "
            "results=[np.float64(2.1), np.float64(3.5), np.float64(8.0)]")
    parsed = _parse_results(line)
    assert len(parsed) == 1
    param, vals, res = parsed[0]
    assert param == "lambda"
    assert vals == [0.5, 1.0, 1.5]
    assert res == [2.1, 3.5, 8.0]


def test_sensitivity_retries_after_failure_then_succeeds(mocker, workdir):
    plan = SensitivityPlan(runs=[{"parameter": "lambda", "values": [0.5, 1, 1.5, 2, 2.5],
                                  "metric": "y", "rationale": "x"}])
    # 第一次：NameError（与 eval_v5 真实失败一致）
    bad = SensitivityCode(code="matplotlib.rcParams['x']=1")
    # 第二次：成功
    good = SensitivityCode(code=(
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "vals=[0.5,1,1.5,2,2.5]; res=[v*2 for v in vals]\n"
        "plt.plot(vals,res); plt.savefig('lambda.png')\n"
        "print(f'RESULT: parameter=lambda values={vals} results={res}')\n"
    ))
    interp = Interpretations(interpretations=["参数 lambda 上升时 y 线性增长。"])
    mocker.patch("math_agent.nodes.sensitivity.complete",
                 side_effect=[plan, bad, good, interp])
    delta = sensitivity_node(_ok_state(workdir))
    assert delta.get("errors") is None or "sensitivity:" not in str(delta.get("errors", ""))
    assert len(delta["sensitivity_runs"]) == 1
    assert delta["sensitivity_runs"][0].results == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_sensitivity_prompt_on_timeout_asks_to_shrink(mocker, workdir):
    """attempt_0 扫参超时时，attempt_1 的 code prompt 应命中"缩小规模"，而非"stderr 节选"。"""
    from math_agent.tools.runner import RunResult

    plan = SensitivityPlan(runs=[{"parameter": "lambda", "values": [0.5, 1, 1.5, 2, 2.5],
                                  "metric": "y", "rationale": "r"}])
    ok_code = SensitivityCode(code=(
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "vals=[1,2,3]; res=[v*2 for v in vals]\n"
        "plt.plot(vals,res); plt.savefig('lambda.png')\n"
        "print(f'RESULT: parameter=lambda values={vals} results={res}')\n"
    ))
    interp = Interpretations(interpretations=["ok"])
    spy = mocker.patch("math_agent.nodes.sensitivity.complete",
                       side_effect=[plan, ok_code, ok_code, interp])

    # 第 1 次 run_python 超时，第 2 次成功
    from unittest.mock import patch
    real = None
    call_i = {"n": 0}

    def _fake_run(code, *, workdir, timeout):
        call_i["n"] += 1
        if call_i["n"] == 1:
            return RunResult(success=False, stderr="timeout after 300s",
                             error_kind="timeout")
        # 第 2 次调真的跑，让 stdout parseable
        from math_agent.tools.runner import run_python as _rp
        return _rp(code, workdir=workdir, timeout=timeout)

    mocker.patch("math_agent.nodes.sensitivity.run_python", side_effect=_fake_run)

    sensitivity_node(_ok_state(workdir))

    # complete 调用顺序：plan(0) → code attempt0(1) → code attempt1(2) → interpret(3)
    second_code_prompt = spy.call_args_list[2].args[0]
    assert "缩小扫参规模" in second_code_prompt
    assert "stderr 节选" not in second_code_prompt


def test_sensitivity_prompt_on_runtime_feeds_stderr(mocker, workdir):
    """attempt_0 runtime 失败时，attempt_1 的 prompt 应喂 stderr。"""
    from math_agent.tools.runner import RunResult

    plan = SensitivityPlan(runs=[{"parameter": "lambda", "values": [0.5, 1, 1.5, 2, 2.5],
                                  "metric": "y", "rationale": "r"}])
    ok_code = SensitivityCode(code=(
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "vals=[1,2,3]; res=[v*2 for v in vals]\n"
        "plt.plot(vals,res); plt.savefig('lambda.png')\n"
        "print(f'RESULT: parameter=lambda values={vals} results={res}')\n"
    ))
    interp = Interpretations(interpretations=["ok"])
    spy = mocker.patch("math_agent.nodes.sensitivity.complete",
                       side_effect=[plan, ok_code, ok_code, interp])

    call_i = {"n": 0}

    def _fake_run(code, *, workdir, timeout):
        call_i["n"] += 1
        if call_i["n"] == 1:
            return RunResult(success=False,
                             stderr="Traceback ... ZeroDivisionError: x",
                             error_kind="runtime")
        from math_agent.tools.runner import run_python as _rp
        return _rp(code, workdir=workdir, timeout=timeout)

    mocker.patch("math_agent.nodes.sensitivity.run_python", side_effect=_fake_run)

    sensitivity_node(_ok_state(workdir))
    second_code_prompt = spy.call_args_list[2].args[0]
    assert "stderr 节选" in second_code_prompt
    assert "ZeroDivisionError" in second_code_prompt
    assert "缩小扫参规模" not in second_code_prompt


def test_sensitivity_rejects_mismatched_result_lengths(mocker, workdir):
    from math_agent.tools.runner import RunResult
    plan = SensitivityPlan(runs=[{
        "parameter": "lambda", "values": [1, 2], "metric": "y", "rationale": "r",
    }])
    code = SensitivityCode(code="print('unused')")
    mocker.patch("math_agent.nodes.sensitivity.complete", side_effect=[plan, code])
    mocker.patch("math_agent.nodes.sensitivity.run_python", return_value=RunResult(
        success=True,
        stdout="RESULT: parameter=lambda values=[1, 2] results=[3]",
    ))
    delta = sensitivity_node(_ok_state(workdir))
    assert delta["errors"]
    assert "sensitivity_runs" not in delta


def test_sensitivity_rejects_incomplete_interpretations(mocker, workdir):
    from math_agent.tools.runner import RunResult
    plan = SensitivityPlan(runs=[
        {"parameter": "a", "values": [1, 2], "metric": "y"},
        {"parameter": "b", "values": [1, 2], "metric": "z"},
    ])
    code = SensitivityCode(code="print('unused')")
    interpretations = Interpretations(interpretations=["only one"])
    mocker.patch(
        "math_agent.nodes.sensitivity.complete",
        side_effect=[plan, code, interpretations],
    )
    mocker.patch("math_agent.nodes.sensitivity.run_python", return_value=RunResult(
        success=True,
        stdout=(
            "RESULT: parameter=a values=[1, 2] results=[3, 4]\n"
            "RESULT: parameter=b values=[1, 2] results=[5, 6]"
        ),
    ))
    delta = sensitivity_node(_ok_state(workdir))
    assert delta["errors"]
    assert "sensitivity_runs" not in delta


def test_sensitivity_code_prompt_includes_data_paths():
    from math_agent.state import DataFileInfo
    from math_agent.prompts.sensitivity import build_code_prompt

    model = ModelVersion(
        stage="improved", description="VRP with time windows",
        equations=["min total_cost"], variables={},
    )
    plan_runs = [{"parameter": "alpha", "values": [0.1, 0.3, 0.5], "metric": "total_cost"}]
    data_files = [DataFileInfo(
        filename="distances.xlsx", file_type="xlsx", path="distances.xlsx",
        summary={},
    )]
    prompt = build_code_prompt(model, plan_runs, data_dir="/data/run1", data_files=data_files)
    assert "/data/run1" in prompt
    assert "distances.xlsx" in prompt
