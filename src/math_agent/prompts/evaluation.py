"""Evaluation Module：与 PaperCritic 解耦的独立打分，更结构化（对齐国赛四大标准 + 加分项）。"""

SYSTEM = (
    "你是国赛阅卷打分官。请独立、严格地按下列维度打分（每项 0-10，整数）："
    "assumption_reasonableness（假设合理性）、modeling_creativity（建模创造性）、"
    "result_correctness（结果正确性）、writing_clarity（文字清晰度）、"
    "extra_depth（加分项：分析深度/敏感性/创新点）。"
    "overall = round("
    "0.2*assumption_reasonableness + 0.25*modeling_creativity + "
    "0.25*result_correctness + 0.2*writing_clarity + 0.1*extra_depth, 2)。"
    "请认真给出 issues 和 suggestions，但不要重复 PaperCritic 已经说过的内容。"
)


def build_prompt(paper, figures, sensitivity_runs, paper_critic):
    crit_summary = "（无 PaperCritic 报告）"
    if paper_critic:
        crit_summary = (
            f"score={paper_critic.score}; issues={[i.problem for i in paper_critic.issues[:5]]}; "
            f"suggestions={paper_critic.suggestions[:5]}"
        )
    return (
        f"# 论文摘要\n{paper.abstract[:1000]}\n\n"
        f"# 主体（截断）\n模型：{paper.model_section[:800]}\n\n"
        f"求解：{paper.solution[:800]}\n\n敏感性：{paper.sensitivity[:800]}\n\n"
        f"结论：{paper.conclusion[:500]}\n\n"
        f"# 客观信号\n图数={len(figures)}; 平均图质量="
        f"{sum(f.quality_score for f in figures)/max(1,len(figures)):.1f}; "
        f"sensitivity 数={len(sensitivity_runs)}\n\n"
        f"# PaperCritic 摘要\n{crit_summary}\n\n"
        f"请按 schema 输出 JSON。"
    )
