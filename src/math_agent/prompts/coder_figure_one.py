"""Coder：单图代码生成 prompt。"""

from math_agent.prompts.coder import SYSTEM  # noqa: F401


def _blueprint_metrics_hint(blueprint) -> str:
    """从 blueprint 提取关键指标，提醒代码至少输出这些数值。"""
    if blueprint is None or not blueprint.metrics:
        return ""
    metric_names = ", ".join(m.name for m in blueprint.metrics)
    return (
        f"\n# Blueprint 指标（代码至少输出以下关键指标）\n{metric_names}\n"
        f"代码需要覆盖 final model 的核心变量、目标函数与关键约束。\n"
    )


def _truncate_model_context(model) -> tuple[str, str, str]:
    """压缩 coder prompt，避免把完整建模细节都喂给画图代码生成。"""
    desc = (model.description or "")[:1200]

    equations = list(model.equations or [])
    eq_preview = equations[:12]
    eqs = "\n".join(f"- {e}" for e in eq_preview) or "（暂时无）"
    if len(equations) > len(eq_preview):
        eqs += f"\n- ……其余 {len(equations) - len(eq_preview)} 条方程省略"

    variables = list((model.variables or {}).items())
    var_preview = variables[:20]
    vars_ = "\n".join(f"- {k}: {v}" for k, v in var_preview) or "（暂时无）"
    if len(variables) > len(var_preview):
        vars_ += f"\n- ……其余 {len(variables) - len(var_preview)} 个变量省略"

    return desc, eqs, vars_


def build_prompt_figure_one(model, purpose: str, prev_failure=None, prev_error_kind: str = "",
                            blueprint=None, data_dir=None, data_files=None,
                            canonical_evidence: str = "", previous_code: str = ""):
    """构造单图代码生成 prompt。"""
    desc, eqs, vars_ = _truncate_model_context(model)
    fb = ""
    if prev_failure:
        if prev_error_kind == "timeout":
            fb = (
                "\n# 上次运行超时\n"
                f"标记：{prev_failure[:200]}\n"
                "请大幅缩小数据规模、迭代次数或求解精度，确保单脚本 60 秒内完成；"
                "优先保留图所需的核心逻辑，不必追求完整大规模最优求解。\n"
            )
        else:
            fb = (
                "\n# 上次运行失败（runtime）\n"
                f"stderr 节选：\n{prev_failure[:1000]}\n"
                "请修复后重试。\n"
            )

    metrics_hint = _blueprint_metrics_hint(blueprint)
    data_hint = ""
    if data_dir and data_files:
        from math_agent.prompts._data_hint import build_data_hint
        data_hint = build_data_hint(data_dir, data_files)
    canonical_hint = ""
    if canonical_evidence:
        canonical_hint = (
            "\n# 已通过校验的唯一主方案证据\n"
            f"{canonical_evidence[:1500]}\n"
            "本任务是补充可视化：必须复用上述数值，不得重新求解出另一套主方案指标；"
            "脚本末尾原样输出同一组 RESULT。\n"
            "补充图代码控制在 100 行以内，只读取绘图所需列；禁止重新运行路径优化。\n"
        )
    else:
        canonical_hint = (
            "\n# 首次主求解的资源约束\n"
            "这是本批次唯一的主数值证据。必须基于真实数据完成一个可复现的轻量启发式，"
            "代码建议不超过 180 行、运行不超过 60 秒、内存不超过 1 GB。\n"
            "允许按客户聚合订单后使用最近邻/贪心插入/滚动局部搜索；"
            "算法复杂度应不高于 O(n^2)；禁止 permutations、全路径枚举、"
            "指数分支搜索、超大三维数组和无上限 while。\n"
            "必须明确解析跨表 customer_id 与 HH:MM 时间窗，并输出车辆数、总成本、"
            "服务率、碳排放等可核验指标。\n"
            "客户聚合需求可能超过单车容量：必须拆成容量可行的多次访问，或保持订单级求解；"
            "禁止 `while unserved` 在无可行客户时原地循环。每轮外层循环必须移除至少一个任务，"
            "否则立即 raise 报错。大型 Excel 距离矩阵应使用 read_only 流式读取或按需距离访问，"
            "禁止复制成 `(i,j)->distance` 的数百万项 Python 字典。\n"
        )

    repair_hint = ""
    if previous_code:
        repair_hint = (
            "\n# 上一版可运行脚本（本次应做最小定向修复）\n"
            f"```python\n{previous_code[:24000]}\n```\n"
            "保留其中已经正确的数据读取、算法和输出结构，只修复 stderr 指向的问题；"
            "不要从零重写，也不要改变已有 RESULT 指标口径。\n"
        )

    return (
        f"# 模型摘要\n{desc}\n\n"
        f"# 核心方程（节选）\n{eqs}\n\n"
        f"# 核心变量（节选）\n{vars_}\n\n"
        f"# 当前绘图任务\n{purpose}\n"
        f"{metrics_hint}{data_hint}{canonical_hint}{fb}{repair_hint}\n"
        f"请为上述绘图任务生成一段独立可运行的 Python 脚本。\n"
        f"优先使用标准库 + numpy + matplotlib；除非确有必要，不要依赖 pandas、seaborn、networkx 等额外库。\n"
        f"如果任务属于鲁棒性或敏感性图，请用小规模、轻量级实验设计，保证单脚本 60 秒内完成。\n"
        f"如果完整 Branch-and-Cut 太重，必须使用上述轻量启发式处理完整客户集合，"
        f"不得通过指数枚举追求精确最优。\n"
        f"请直接输出正常的多行 Python 源码，不要把整段 code 写成带字面量 \\n 的转义字符串。\n"
        f"若标题、注释、docstring 里需要反斜杠或 LaTeX 记号，请使用原始字符串或双反斜杠，避免非法转义。\n"
        f"脚本末尾必须用 print 输出关键指标，格式严格如下：\n"
        f"print(f'RESULT: baseline=ours total_cost={{total_cost}} vehicles={{vehicles}} "
        f"service_rate={{service_rate}} total_carbon={{total_carbon}} "
        f"avg_delivery_time={{avg_delivery_time}}')\n"
        f"主方案 RESULT 至少包含 4 个可核验指标；题目涉及的车辆数、碳排放、"
        f"配送时间不得只写在普通日志中。\n"
        f"stdout 不允许只输出自然语言总结，必须包含 RESULT: 行并带具体数值。\n\n"
        f"发生数据读取或求解异常时必须 raise 并以非零状态退出，不能打印错误后继续输出 RESULT。\n"
        f"所有比例指标必须位于 [0,1]，车辆数等计数必须为整数且不得超过输入订单/节点规模。\n\n"
        f"请输出 JSON：{{\"purpose\": str, \"code\": str}}，code 字段是完整的 Python 源码。"
    )
