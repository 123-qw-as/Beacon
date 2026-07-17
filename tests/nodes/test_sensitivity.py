from pathlib import Path
from math_agent.state import MathModelingState, ModelVersion, Assumption, SensitivityRun
from math_agent.nodes.sensitivity import (
    sensitivity_node, SensitivityPlan, SensitivityCode, Interpretations,
    sensitivity_interpret_node,
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
        "plt.plot(vals,res); plt.savefig('sensitivity_lambda.png')\n"
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
    from math_agent.config import MAX_CODE_RETRIES
    plan = SensitivityPlan(runs=[{"parameter": "lambda", "values": [1, 2, 3, 4, 5],
                                  "metric": "y", "rationale": "x"}])
    bad = SensitivityCode(code="raise RuntimeError('x')")
    mocker.patch(
        "math_agent.nodes.sensitivity.complete",
        side_effect=[plan] + [bad] * (MAX_CODE_RETRIES + 1),
    )
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


def test_sensitivity_parse_preserves_parameter_names_with_spaces():
    from math_agent.nodes.sensitivity import _parse_results

    parsed = _parse_results(
        "RESULT: parameter=碳排放成本系数 (beta) values=[0.5, 1.0] results=[10, 12]"
    )

    assert parsed == [("碳排放成本系数 (beta)", [0.5, 1.0], [10.0, 12.0])]


def test_sensitivity_rejects_center_point_that_does_not_match_primary_metric():
    from math_agent.nodes.sensitivity import _center_alignment_error

    runs = [SensitivityRun(
        parameter="penalty", values=[50, 100, 150],
        metric="total_cost", results=[500000, 1050000, 1500000],
    )]

    reason = _center_alignment_error(runs, {"total_cost": 300524.02})

    assert "基准点口径不一致" in reason


def test_extract_python_source_removes_unclosed_opening_fence():
    from math_agent.nodes.sensitivity import _extract_python_source

    assert _extract_python_source("```python\nprint('ok')") == "print('ok')"


def test_sensitivity_retries_exit_zero_output_failure_and_aligns_alias(mocker, workdir):
    """真实故障回归：Data loading failed + exit 0 必须重试，beta 应对齐完整参数名。"""
    from math_agent.tools.runner import RunResult

    plan = SensitivityPlan(runs=[{
        "parameter": "碳排放成本系数 (beta)",
        "values": [0.5, 1.0],
        "metric": "total_cost",
    }])
    bad = SensitivityCode(code="print('bad')")
    good = SensitivityCode(code="print('good')")
    interp = Interpretations(interpretations=["成本随碳价上升。"])
    spy = mocker.patch(
        "math_agent.nodes.sensitivity.complete",
        side_effect=[plan, bad, good, interp],
    )
    mocker.patch(
        "math_agent.nodes.sensitivity.run_python",
        side_effect=[
            RunResult(success=True, stdout="Data loading failed: '纬度'"),
            RunResult(
                success=True,
                stdout="RESULT: parameter=beta values=[0.5, 1.0] results=[10, 12]",
            ),
        ],
    )

    delta = sensitivity_node(_ok_state(workdir))

    assert len(delta["sensitivity_runs"]) == 1
    assert delta["sensitivity_runs"][0].parameter == "碳排放成本系数 (beta)"
    retry_prompt = spy.call_args_list[2].args[0]
    assert "Data loading failed" in retry_prompt
    assert "上一版扫参脚本" in retry_prompt
    assert "print('bad')" in retry_prompt


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
    from math_agent.config import MAX_CODE_RETRIES
    from math_agent.tools.runner import RunResult
    plan = SensitivityPlan(runs=[{
        "parameter": "lambda", "values": [1, 2], "metric": "y", "rationale": "r",
    }])
    code = SensitivityCode(code="print('unused')")
    mocker.patch(
        "math_agent.nodes.sensitivity.complete",
        side_effect=[plan] + [code] * (MAX_CODE_RETRIES + 1),
    )
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
    assert "errors" not in delta
    assert len(delta["sensitivity_runs"]) == 2
    assert delta["sensitivity_runs"][0].interpretation == "only one"
    assert "参数 b" in delta["sensitivity_runs"][1].interpretation


def test_sensitivity_interpret_node_falls_back_for_missing_entries(mocker, workdir):
    from math_agent.state import SensitivityRun

    state = _ok_state(workdir)
    state.sensitivity_pending_runs = [
        SensitivityRun(parameter="alpha", values=[1, 2, 3], metric="gap", results=[0.3, 0.2, 0.1]),
        SensitivityRun(parameter="beta", values=[1, 2, 3], metric="cpu", results=[10, 12, 15]),
    ]
    mocker.patch(
        "math_agent.nodes.sensitivity.complete",
        return_value=Interpretations(interpretations=["alpha 解读"]),
    )
    delta = sensitivity_interpret_node(state)
    assert len(delta["sensitivity_runs"]) == 2
    assert delta["sensitivity_runs"][0].interpretation == "alpha 解读"
    assert "参数 beta" in delta["sensitivity_runs"][1].interpretation


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


def test_sensitivity_code_generation_uses_code_timeout_profile(mocker, workdir):
    from math_agent.nodes.sensitivity import sensitivity_code_generate_node

    state = _ok_state(workdir)
    state.sensitivity_plan_dump = SensitivityPlan(runs=[{
        "parameter": "beta", "values": [0.8, 1.0], "metric": "cost",
    }]).model_dump()
    spy = mocker.patch(
        "math_agent.nodes.sensitivity.complete",
        return_value=SensitivityCode(code="print('x')"),
    )

    sensitivity_code_generate_node(state)
    assert spy.call_args.kwargs["profile"] == "code"


def test_canonical_replay_supports_model_plan_aliases(workdir):
    from math_agent.nodes.sensitivity import _build_canonical_replay_code
    from math_agent.tools.runner import run_python

    plan = SensitivityPlan(runs=[
        {"parameter": "c_late", "values": [75, 100], "metric": "Z"},
        {"parameter": "beta_v(fuel)", "values": [0.05, 0.1], "metric": "Z"},
        {"parameter": "green_zone_radius", "values": [7.5, 10], "metric": "Z"},
    ])
    main_code = """EARLY_PENALTY = 0.2
LATE_PENALTY = 1.0
GREEN_ZONE_RADIUS = 10.0
carbon = 100.0
total_carbon = carbon
total_cost = 1000.0 + LATE_PENALTY + total_carbon * 0.1
print(f'RESULT: baseline=ours total_cost={total_cost} total_carbon={total_carbon} fuel_ratio=0.5 vehicles=1 service_rate=1.0')
"""

    result = run_python(
        _build_canonical_replay_code(plan, main_code),
        workdir=workdir / "aliases",
        timeout=30,
    )

    assert result.success, result.stderr
    assert "parameter=c_late" in result.stdout
    assert "parameter=beta_v(fuel)" in result.stdout
    assert "parameter=green_zone_radius" in result.stdout


def test_canonical_replay_supports_chinese_green_logistics_plan(workdir):
    from math_agent.nodes.sensitivity import _build_canonical_replay_code
    from math_agent.tools.runner import run_python

    plan = SensitivityPlan(runs=[
        {"parameter": "速度时变函数的比例因子（整体速度水平）", "values": [0.8, 1.0], "metric": "总成本 (Z)"},
        {"parameter": "绿色区限行时段开始时间（小时）", "values": [7.0, 8.0], "metric": "总成本 (Z)"},
        {"parameter": "软时间窗单位惩罚成本系数（元/分钟）", "values": [0.75, 1.0], "metric": "总成本 (Z)"},
    ])
    main_code = '''import numpy as np
BAN_START = 480.0
def speed(minute):
    points_t = np.array([0, 1440], dtype=float)
    points_v = np.array([40, 40], dtype=float)
    return float(np.interp(float(minute) % 1440.0, points_t, points_v))
late = 2.0
total_cost = 1000.0 + 1.0 * late + speed(10) + BAN_START
print(f"RESULT: baseline=ours total_cost={total_cost} vehicles=1 service_rate=1 total_carbon=1")
'''

    result = run_python(
        _build_canonical_replay_code(plan, main_code),
        workdir=workdir / "chinese-plan", timeout=30,
    )

    assert result.success, result.stderr
    assert "速度时变函数的比例因子" in result.stdout
    assert "绿色区限行时段开始时间" in result.stdout
    assert "软时间窗单位惩罚成本系数" in result.stdout

def test_data_hint_uses_safe_windows_paths_and_real_columns():
    from math_agent.prompts._data_hint import build_data_hint
    from math_agent.state import DataFileInfo

    hint = build_data_hint(
        r"C:\题目\附件",
        [DataFileInfo(
            filename="客户坐标.xlsx", file_type="xlsx", path="客户坐标.xlsx",
            summary={"rows": 100, "columns": ["类型", "ID", "X (km)", "Y (km)"]},
        )],
    )

    assert "C:/题目/附件/客户坐标.xlsx" in hint
    assert "X (km), Y (km)" in hint
    assert "不得猜测" in hint
    assert "单个反斜杠结尾" in hint


def test_data_hint_profiles_actual_dtypes_and_samples(workdir):
    from math_agent.prompts._data_hint import build_data_hint
    from math_agent.state import DataFileInfo

    csv = workdir / "时间窗.csv"
    csv.write_text("客户编号,开始时间,结束时间\n1,11:33,12:22\n", encoding="utf-8")
    hint = build_data_hint(str(workdir), [DataFileInfo(
        filename=csv.name, file_type="csv", path=csv.name, summary={},
    )])

    assert "真实读取契约" in hint
    assert "开始时间" in hint
    assert "11:33" in hint
    assert "HH:MM" in hint
    assert "customer_id" in hint
