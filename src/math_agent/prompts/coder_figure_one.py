"""Coder：单图代码生成 prompt。

Plan D Phase 3：modeler 在 final 阶段产出 figure_purposes（5-10 个图任务），
coder_node 对每个 purpose 调用一次本 prompt，生成一段**独立可运行**的 Python 脚本。
复用 coder.py 的 SYSTEM（含发表级绘图质量要求），不重复维护。
"""

from math_agent.prompts.coder import SYSTEM  # noqa: F401  (re-exported for coder_node)


def _blueprint_metrics_hint(blueprint) -> str:
    """从 blueprint 提取指标清单，要求代码至少输出这些指标。"""
    if blueprint is None or not blueprint.metrics:
        return ""
    metric_names = ", ".join(m.name for m in blueprint.metrics)
    return (
        f"\n# Blueprint 指标（代码至少输出以下关键指标）\n{metric_names}\n"
        f"代码必须实现 final model 的核心变量、目标函数、约束。\n"
    )


def build_prompt_figure_one(model, purpose: str, prev_failure=None, prev_error_kind: str = "",
                            blueprint=None, data_dir=None, data_files=None):
    """构造单图代码生成 prompt。

    model: 当前 ModelVersion（提供 description / equations / variables 上下文）
    purpose: 当前要画的图的描述（来自 model.figure_purposes）
    prev_failure: 上一次运行的 stderr 节选（None 表示首次）
    prev_error_kind: RunResult.error_kind，∈ {"", "timeout", "runtime"}
                     用来给 LLM 分流建议--超时要缩规模，runtime 才要看 stderr 修 bug
    blueprint: ProblemBlueprint，提供指标/变量/目标/约束对齐信息
    """
    eqs = "\n".join(f"- {e}" for e in model.equations) or "（暂无）"
    vars_ = "\n".join(f"- {k}: {v}" for k, v in model.variables.items()) or "（暂无）"
    fb = ""
    if prev_failure:
        if prev_error_kind == "timeout":
            # timeout 时 stderr 是 "timeout after Ns"，没修 bug 的价值；重点让 LLM 缩规模
            fb = (
                "\n# 上次运行超时\n"
                f"标记：{prev_failure[:200]}\n"
                "请**大幅缩小数据规模 / 迭代次数 / 求解精度**（示例：n_stations 5->3, "
                "MC 仿真次数 1000->100, 时间步 96->24），保证 5 分钟内跑完；"
                "算法逻辑不必改，只调超参与规模。"
            )
        else:
            # runtime 或未标记 -> 保持原行为：把 stderr 喂回让 LLM 修
            fb = f"\n# 上次运行失败（runtime）\nstderr 节选：\n{prev_failure[:1000]}\n请修正后重试。"
    metrics_hint = _blueprint_metrics_hint(blueprint)
    data_hint = ""
    if data_dir and data_files:
        from math_agent.prompts._data_hint import build_data_hint
        data_hint = build_data_hint(data_dir, data_files)
    return (
        f"# 模型描述\n{model.description}\n\n# 方程\n{eqs}\n\n# 变量\n{vars_}\n\n"
        f"# 当前绘图任务\n{purpose}\n{metrics_hint}{data_hint}{fb}\n\n"
        f"请为上述绘图任务生成一段**独立可运行**的 Python 脚本。\n"
        f"脚本末尾必须用 print 输出关键指标，格式严格如下：\n"
        f"print(f'RESULT: baseline=ours total_cost={{total_cost}} service_rate={{service_rate}}')\n"
        f"（指标名按题目调整，但必须以 RESULT: baseline=ours 开头，供对比表使用）\n"
        f"stdout 不允许只输出自然语言总结，必须包含 RESULT: 行带具体数值。\n\n"
        f"请输出 JSON：{{\"purpose\": str, \"code\": str}}，code 字段是完整的 Python 源码。"
    )
