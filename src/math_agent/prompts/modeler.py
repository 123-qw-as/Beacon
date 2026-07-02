"""Modeler：依据当前 stage 产出对应版本的模型。"""

SYSTEM = (
    "你是数学建模队的主建模手。请在给定假设下构建数学模型。"
    "你必须按照 stage 渐进：basic（最简可解模型）→ improved（加入更多现实因素）"
    "→ final（综合性最强、可被敏感性分析的最终模型）。"
)


def build_prompt(problem, assumptions, prev_model, stage, critic_feedback=None,
                 retrieved_context: str = ""):
    asum = "\n".join(f"- {a.statement}（依据：{a.rationale}）" for a in assumptions) or "（暂无）"
    prev = "（无前一版本）"
    if prev_model is not None:
        prev = f"[{prev_model.stage}] {prev_model.description}\n方程：" + " ; ".join(prev_model.equations)
    fb = ""
    if critic_feedback:
        fb = "\n# 上一版 Critic 反馈\n" + "\n".join(
            f"- 问题: {i.problem}" for i in critic_feedback.issues
        ) + "\n" + "\n".join(f"- 建议: {s}" for s in critic_feedback.suggestions)
    ctx = f"\n{retrieved_context}\n" if retrieved_context else ""

    # Plan D Phase 3：final 阶段才要求 figure_purposes（basic/improved 不需要图，
    # 字段在 ModelVersion 里默认空 list，prompt 也不提及，避免污染早期建模）
    figure_clause = ""
    if stage == "final":
        figure_clause = (
            f"  \"figure_purposes\": [str, ...], # 5-10 个图任务，每个是一句话描述要画的图，"
            f"如 '需求时序图', '调度路径图', '成本构成饼图', '敏感性曲线'\n"
        )

    return (
        f"# 题目\n{problem}\n\n# 当前阶段\n{stage}\n\n"
        f"# 已确认假设\n{asum}\n\n# 上一版模型\n{prev}\n{fb}\n"
        f"{ctx}\n"
        f"请输出 JSON：{{\n"
        f"  \"stage\": \"{stage}\",\n"
        f"  \"description\": str,        # 模型定位与核心思路，>= 200 字\n"
        f"  \"equations\": [str, ...],   # LaTeX 字符串\n"
        f"  \"variables\": {{name: meaning}},\n"
        f"{figure_clause}"
        f"  \"notes\": str               # 与上一版的区别（basic 阶段可为空）\n"
        f"}}"
    )
