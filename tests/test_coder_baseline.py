"""对照方案 prompt 构建器测试。"""
from math_agent.prompts.coder_baseline import BASELINE_SPECS, build_baseline_prompt


def test_baseline_specs_has_three():
    assert len(BASELINE_SPECS) == 3
    names = [s[0] for s in BASELINE_SPECS]
    assert "无调度" in names
    assert "简单平均预测" in names
    assert "贪婪启发式" in names


def test_baseline_prompt_contains_main_code():
    prompt = build_baseline_prompt(
        problem="共享单车调度",
        main_code="import numpy as np\nprint('main')",
        name="无调度",
        category="no_schedule",
        instruction="把优化步骤删除",
    )
    assert "import numpy as np" in prompt
    assert "无调度" in prompt
    assert "RESULT: baseline=no_schedule" in prompt


def test_baseline_prompt_contains_output_contract():
    prompt = build_baseline_prompt(
        problem="test",
        main_code="print(1)",
        name="贪婪",
        category="greedy",
        instruction="用贪心替换",
    )
    assert "RESULT: baseline=greedy" in prompt
    assert "JSON" in prompt


def test_coder_node_produces_baseline_artifacts(monkeypatch):
    """coder_node 应在主方案后追加 3 个 category='baseline:...' 的 artifacts。"""
    from math_agent.nodes.coder import coder_node, CoderDraft
    from math_agent.state import MathModelingState, ModelVersion
    from math_agent.tools.runner import RunResult

    s = MathModelingState(problem="test", output_dir="/tmp/test_coder_baseline")
    s.model_versions.append(ModelVersion(
        stage="final", description="test model",
        variables={"x": "v"}, figure_purposes=["plot1"],
    ))

    call_count = {"n": 0}
    def mock_complete(prompt, *, schema=None, **kw):
        call_count["n"] += 1
        if call_count["n"] <= 1:
            return CoderDraft(purpose="main plot", code="print('main')")
        specs = ["no_schedule", "simple_pred", "greedy"]
        idx = call_count["n"] - 2
        return CoderDraft(
            purpose=f"baseline {specs[idx]}",
            code=f"print('RESULT: baseline={specs[idx]} total_cost=100.0 service_rate=0.9')",
        )

    def mock_run(code, *, workdir, timeout=60, **kw):
        return RunResult(success=True, stdout=code.replace("print(", "").replace("')", "").replace("'", ""),
                         artifact_paths=[])

    monkeypatch.setattr("math_agent.nodes.coder.complete", mock_complete)
    monkeypatch.setattr("math_agent.nodes.coder.run_python", mock_run)

    result = coder_node(s)
    artifacts = result["code_artifacts"]
    assert len(artifacts) == 4  # 1 主方案 + 3 对照方案
    baseline_arts = [a for a in artifacts if a.category.startswith("baseline:")]
    assert len(baseline_arts) == 3
    categories = [a.category for a in baseline_arts]
    assert "baseline:no_schedule" in categories
    assert "baseline:simple_pred" in categories
    assert "baseline:greedy" in categories


def test_main_figure_prompt_includes_ours_result_contract():
    """I1 回归：主方案 prompt 必须要求输出 RESULT: baseline=ours，否则对比表缺本文方案行。"""
    from math_agent.prompts.coder_figure_one import build_prompt_figure_one
    from math_agent.state import ModelVersion
    m = ModelVersion(stage="final", description="test", variables={"x": "v"})
    prompt = build_prompt_figure_one(m, "plot1")
    assert "RESULT: baseline=ours" in prompt
