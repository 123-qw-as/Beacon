"""Sensitivity：选择关键参数 + 给出扫参代码（仍由沙箱执行）+ 解读结果。

设计：把"选参数+造代码"和"读结果"拆成两个 prompt，避免一次返回过大的 JSON。
"""

PLAN_SYSTEM = (
    "你是国赛评委关心的敏感性分析专家。请基于已有的最终模型和已确认假设，"
    "选出 1-3 个最值得做敏感性分析的参数。优先选标记了 sensitivity_relevant=True 的假设里出现的参数。"
)


def build_plan_prompt(model, assumptions):
    asum = "\n".join(
        f"- [{'敏感' if a.sensitivity_relevant else '常规'}] {a.statement}"
        for a in assumptions
    )
    eqs = "\n".join(f"- {e}" for e in model.equations)
    return (
        f"# 最终模型\n{model.description}\n方程：\n{eqs}\n\n# 假设\n{asum}\n\n"
        f"请输出 JSON：{{\"runs\": [{{\"parameter\": str, \"values\": [float, ...], "
        f"\"metric\": str, \"rationale\": str}}, ...]}}，"
        f"每个 run 的 values 至少 5 个点，跨度合理（涵盖参数典型范围的 ±30%~50%）。"
    )


CODE_SYSTEM = (
    "你是建模队工程师。根据敏感性分析计划，写一段独立可运行的 Python，"
    "对每个 run 计算 metric 随 parameter 变化的曲线，并保存 PNG 到当前目录。"
    "约束：只用 numpy/scipy/matplotlib；为每个 run 单独保存一张 *.png；"
    "中文字体设置：开头加 `matplotlib.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']; matplotlib.rcParams['axes.unicode_minus']=False`；"
    "用 print 输出 `RESULT: parameter=<名称字面量> values=<list> results=<list>` 行（每个 run 一行），"
    "**parameter 必须是字符串字面量（如 `parameter=alpha`），不要写 `parameter={alpha}` 这种把变量值代入的写法**，"
    "values/results 用 Python 列表的 repr（例如 `[0.1, 0.2, 0.3]`），方便正则解析。"
)


def build_code_prompt(model, plan_runs):
    desc = "\n".join(
        f"- parameter={r['parameter']}, values={r['values']}, metric={r['metric']}"
        for r in plan_runs
    )
    return (
        f"# 最终模型\n{model.description}\n方程：\n{chr(10).join(model.equations)}\n\n"
        f"# 敏感性分析计划\n{desc}\n\n"
        f"请输出 JSON：{{\"code\": str}}。"
    )


INTERPRET_SYSTEM = (
    "你是国赛主笔。根据敏感性分析的数值结果，写出每个参数的解读（趋势 + 含义 + 对结论的影响），"
    "每条 80-150 字，避免空话。"
)


def build_interpret_prompt(runs):
    rows = "\n".join(
        f"- {r.parameter}={r.values} → {r.metric}={r.results}"
        for r in runs
    )
    return (
        f"# 数值结果\n{rows}\n\n"
        f"请输出 JSON：{{\"interpretations\": [str, ...]}}，长度与上面行数一致。"
    )
