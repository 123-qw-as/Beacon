"""table_assembler 端到端验证：mock state → run node → verify output。

不依赖 LLM / xelatex。验证：
1) 变量表注入到 notation
2) 敏感性表注入到 sensitivity
3) 禁用词被清洗
4) table_warnings 非空
5) 空 state 不崩
"""
from __future__ import annotations

from math_agent.state import (
    MathModelingState, ModelVersion, PaperSections, SensitivityRun,
)
from math_agent.nodes.table_assembler import table_assembler_node


def main():
    # 构造 mock state
    s = MathModelingState(problem="共享单车调度问题")
    s.model_versions.append(ModelVersion(
        stage="final",
        description="XGBoost + 随机规划滚动优化",
        variables={
            "x_i": "第i时段调度量(辆)",
            "d_i": "第i时段需求量(辆)",
            "s_i": "第i时段初始存量",
            "alpha": "学习率",
            "beta": "惩罚系数",
        },
        equations=["\\min \\sum_i (c_i x_i + p_i (d_i - x_i)^+)"],
    ))
    s.sensitivity_runs = [
        SensitivityRun(parameter="alpha", values=[0.01, 0.1, 0.5],
                       metric="MAE", results=[3.2, 2.4, 2.8]),
        SensitivityRun(parameter="beta", values=[0.5, 1.0, 2.0],
                       metric="total_cost", results=[1200, 1245, 1180]),
    ]
    s.paper = PaperSections(
        abstract="本文研究共享单车调度问题。",
        problem_restatement="问题分析：需要决策变量 x_i。",
        assumptions="**假设1**：需求服从泊松分布。",
        notation="| 符号 | 含义 |\n|---|---|\n| x_i | 调度量 |",
        model_section="## 基础预测模型\n使用 XGBoost。PaperCritic 评分 8 分。",
        solution="## 求解算法与流程\n见代码1。Claim: 成本下降 15%。",
        sensitivity="对参数 alpha 做敏感性分析。",
        conclusion="模型优点：精度高。",
        references="[1] 作者. 标题. 期刊, 2024.",
    )

    result = table_assembler_node(s)
    paper = result["paper"]
    warnings = result["table_warnings"]

    # 1) 变量表
    assert "| 符号 | 含义 | 单位 |" in paper.notation, "变量表未注入 notation"
    assert "x_i" in paper.notation
    print("✓ 变量表注入 notation")

    # 2) 敏感性表
    assert "| 参数 | 取值范围 |" in paper.sensitivity, "敏感性表未注入"
    assert "alpha" in paper.sensitivity
    print("✓ 敏感性表注入 sensitivity")

    # 3) 禁用词清洗
    assert "PaperCritic" not in paper.model_section, "PaperCritic 未清洗"
    assert "Claim" not in paper.solution, "Claim 未清洗"
    assert "代码1" not in paper.solution, "代码1 未清洗"
    print("✓ 禁用词已清洗")

    # 4) warnings
    assert len(warnings) >= 3, f"warnings 不足: {warnings}"
    print(f"✓ table_warnings: {len(warnings)} 条")

    # 5) 空 state
    empty = MathModelingState(problem="empty")
    empty.paper = PaperSections()
    result2 = table_assembler_node(empty)
    assert "paper" in result2
    print("✓ 空 state 不崩")

    # 6) Phase 2: 对比表注入 solution
    from math_agent.state import CodeArtifact
    s.code_artifacts = [
        CodeArtifact(
            purpose="无调度", code="", success=True,
            stdout="RESULT: baseline=no_schedule total_cost=1500.0 service_rate=0.70",
            category="baseline:no_schedule",
        ),
        CodeArtifact(
            purpose="贪婪", code="", success=True,
            stdout="RESULT: baseline=greedy total_cost=1100.0 service_rate=0.85",
            category="baseline:greedy",
        ),
    ]
    result_p2 = table_assembler_node(s)
    assert "| 方案 |" in result_p2["paper"].solution, "对比表未注入 solution"
    assert "无调度" in result_p2["paper"].solution
    print("✓ 对比表注入 solution（Phase 2）")

    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
