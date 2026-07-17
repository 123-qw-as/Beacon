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
    "**IRON RULE：代码开头必须显式 import 所有用到的库**——"
    "`import numpy as np; import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt`，"
    "不得依赖外部已导入的变量或隐式导入。缺 import 会直接导致 NameError；"
    "用 print 输出 `RESULT: parameter=<名称字面量> values=<list> results=<list>` 行（每个 run 一行），"
    "**parameter 必须逐字使用计划中的完整名称（包括空格和括号）；也可只使用括号内英文别名。**"
    "不要写 `parameter={alpha}` 这种把变量值代入的写法。"
    "values/results 用 Python 列表的 repr（例如 `[0.1, 0.2, 0.3]`），方便正则解析。\n"
    "不得用手工构造的线性/二次曲线冒充重求解结果；每个扰动值都必须重新运行同一目标函数与约束。"
    "不得捕获数据读取或求解异常后继续返回 0；发生异常必须 raise，使进程以非零状态退出。"
    "完整脚本应复用函数并控制在 4500 字符左右，避免重复展开同一套求解逻辑。"
    # 图表质量（参考 nature-figure 准则，与 coder 一致）：
    "绘图质量要求："
    "(1) 每张图只论证一个参数的敏感性，title 写 `Sensitivity of <metric> to <parameter>`；"
    "(2) 必备：title、x 轴=parameter 名+单位、y 轴=metric 名+单位、legend（如多曲线）；"
    "(3) 发表级 rcParams：savefig dpi≥300、font.size≥10、axes.linewidth=0.8、关闭右上脊柱；"
    "(4) 配色克制：单参数曲线用一种主色；网格 alpha≤0.3。"
)


def build_code_prompt(model, plan_runs, prev_failure: str | None = None,
                      prev_error_kind: str = "",
                      data_dir=None, data_files=None, previous_code: str = "",
                      main_code: str = "", canonical_metrics: dict | None = None):
    """构造敏感性扫参代码 prompt。

    prev_error_kind: RunResult.error_kind，∈ {"", "timeout", "runtime", "output_validation"}
      timeout → 让 LLM 缩规模（值列表变短、MC 次数变少）而不是修 bug
      runtime → 喂 stderr 让它修
    """
    desc = "\n".join(
        f"- parameter={r['parameter']}, values={r['values']}, metric={r['metric']}"
        for r in plan_runs
    )
    fb = ""
    if prev_failure:
        if prev_error_kind == "timeout":
            fb = (
                "\n# 上次扫参超时\n"
                f"标记：{prev_failure[:200]}\n"
                "请**大幅缩小扫参规模**（示例：每个参数的 values 缩到 3-5 个点、"
                "内层仿真步数减半），保证 5 分钟内跑完；扫参逻辑不必改。\n"
            )
        else:
            fb = (
                f"\n# 上次运行或输出校验失败（{prev_error_kind or 'runtime'}）\n"
                f"stderr 节选（输出校验失败时为校验信息）：\n{prev_failure[:1000]}\n"
                "请修正后重试；不得吞掉异常或伪造 RESULT。\n"
            )
    data_hint = ""
    if data_dir and data_files:
        from math_agent.prompts._data_hint import build_data_hint
        data_hint = build_data_hint(data_dir, data_files)
    repair = ""
    if previous_code:
        repair = (
            "\n# 上一版扫参脚本（只做定向修复）\n"
            f"```python\n{previous_code[:24000]}\n```\n"
            "保留正确的数据读取、重求解和绘图逻辑，只修复 stderr 指向的问题；不要从零重写。\n"
        )
    canonical = ""
    if main_code:
        canonical = (
            "\n# 正式主方案代码与基准指标（必须复用同一目标函数和约束）\n"
            f"基准指标：{canonical_metrics or {}}\n"
            f"```python\n{main_code[:16000]}\n```\n"
            "每条敏感性曲线的中心参数值必须复现上述对应基准指标（允许数值算法小误差）；"
            "不得另写一套口径、缩放单位或手工校准曲线。\n"
        )
    return (
        f"# 最终模型\n{model.description}\n方程：\n{chr(10).join(model.equations)}\n\n"
        f"# 敏感性分析计划\n{desc}\n{data_hint}{canonical}{fb}{repair}\n"
        f"请只输出完整 Python 源码，不要 JSON、解释文字或 Markdown 围栏。"
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
