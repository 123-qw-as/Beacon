"""对照方案 prompt 构建器。

3 个预设对照方案，用一个参数化 prompt 模板 × 3 次调用。
每个方案基于主方案代码做最小修改（删优化 / 换预测 / 换求解器），
保留数据加载和评估逻辑不变，输出 RESULT: 行供 extract_numeric_results 解析。
"""
from __future__ import annotations

# (name, category, instruction) — 3 个预设对照方案
BASELINE_SPECS: list[tuple[str, str, str]] = [
    (
        "无调度",
        "no_schedule",
        "把主方案代码中的优化/调度求解步骤全部删除，改为'不调整/保持现状'（所有决策变量取默认值 0 或保持初始值）。"
        "保留数据加载、需求计算和评估逻辑不变。最终用相同的指标函数计算成本和服务率。",
    ),
    (
        "简单平均预测",
        "simple_pred",
        "把主方案代码中的预测模型（XGBoost/STGNN/回归等）替换为简单历史均值预测："
        "prediction = np.mean(historical_data, axis=0)。保留调度/优化代码和评估逻辑不变。",
    ),
    (
        "贪婪启发式",
        "greedy",
        "把主方案代码中的优化求解器（MILP/随机规划/滚动优化等）替换为贪心策略："
        "while 循环 + 每次取当前需求最大（或缺口最大）的站点优先分配，直到资源耗尽。"
        "保留数据加载、预测和评估逻辑不变。",
    ),
]


def build_baseline_prompt(
    problem: str,
    main_code: str,
    name: str,
    category: str,
    instruction: str,
) -> str:
    """构造对照方案代码生成 prompt。

    一个模板，3 次调用不同 (name, category, instruction)。
    输出 RESULT: baseline={category} metric1=v1 metric2=v2 格式。
    """
    return (
        f"# 题目\n{problem[:500]}\n\n"
        f"# 对照方案：{name}\n"
        f"## 修改指令\n{instruction}\n\n"
        f"# 主方案代码（参考基础）\n```python\n{main_code[:3000]}\n```\n\n"
        f"# 输出要求\n"
        f"基于主方案代码做上述修改，生成一段**独立可运行**的 Python 脚本。\n"
        f"脚本末尾必须用 print 输出至少 2 个指标，格式严格如下：\n"
        f"print(f'RESULT: baseline={category} total_cost={{total_cost}} service_rate={{service_rate}}')\n"
        f"（指标名可按题目调整，但必须以 RESULT: baseline={category} 开头）\n\n"
        f"请输出 JSON：{{\"purpose\": \"{name}对照方案\", \"code\": str}}，code 字段是完整的 Python 源码。"
    )
