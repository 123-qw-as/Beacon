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
