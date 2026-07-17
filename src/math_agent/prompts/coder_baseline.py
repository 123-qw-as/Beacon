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
        "把主方案的优化排序替换为先到先服务（按订单或客户原始顺序）的可行构造法。"
        "仍须逐单满足载重、容积和基本时间递推，车辆满载后开启下一辆；不得把决策变量全置 0。"
        "保留真实数据加载与主方案相同的成本、车辆、服务率、碳排放评估口径。",
    ),
    (
        "简单平均预测",
        "simple_pred",
        "把主方案代码中的预测模型（XGBoost/STGNN/回归等）替换为简单历史均值预测："
        "prediction = np.nanmean(historical_data, axis=0)。若题目没有独立预测子问题，则用附件中非空需求的全局均值"
        "作为排序/容量估计的基线特征，但可行性和最终指标必须用真实需求复算。不得输出 NaN/Inf，"
        "也不得原样复用主方案 RESULT。保留调度/优化代码和评估逻辑不变。",
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
    prev_failure: str | None = None,
    prev_error_kind: str = "",
    previous_code: str = "",
) -> str:
    """构造对照方案代码生成 prompt。

    一个模板，3 次调用不同 (name, category, instruction)。
    输出 RESULT: baseline={category} metric1=v1 metric2=v2 格式。
    """
    retry = ""
    if prev_failure:
        retry = (
            f"\n# 上次运行或输出校验失败（{prev_error_kind or 'runtime'}）\n"
            f"{prev_failure[:1000]}\n"
            "请修正代码后重试。不得捕获异常后继续返回 0，也不得输出全零伪结果。\n"
        )
    repair = ""
    if previous_code:
        repair = (
            "\n# 上一版对照脚本（只做定向修复）\n"
            f"```python\n{previous_code[:24000]}\n```\n"
            "保留正确部分，只修复上面的运行/校验错误，不要从零重写。\n"
        )
    return (
        f"# 题目\n{problem[:500]}\n\n"
        f"# 对照方案：{name}\n"
        f"## 修改指令\n{instruction}\n\n"
        f"# 主方案代码（参考基础）\n```python\n{main_code[:24000]}\n```\n\n"
        f"# 输出要求\n"
        f"基于主方案代码做上述修改，生成一段**独立可运行**的 Python 脚本。\n"
        f"脚本末尾必须用 print 输出至少 2 个指标，格式严格如下：\n"
        f"print(f'RESULT: baseline={category} total_cost={{total_cost}} service_rate={{service_rate}}')\n"
        f"（指标名可按题目调整，但必须以 RESULT: baseline={category} 开头）\n"
        f"重要：对照方案必须沿用主方案相同的指标名，不得改用不同的指标名，以便对比表汇总。\n"
        f"stdout 不允许只输出自然语言总结，必须包含 RESULT: 行带具体数值。\n\n"
        f"发生数据读取或求解异常时必须 raise 并以非零状态退出，不能打印错误后伪装成功。\n"
        f"{retry}{repair}\n"
        f"请输出 JSON：{{\"purpose\": \"{name}对照方案\", \"code\": str}}，code 字段是完整的 Python 源码。"
    )
