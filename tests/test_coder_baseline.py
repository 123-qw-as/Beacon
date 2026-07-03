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
