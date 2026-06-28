"""PaperCritic：对组装好的论文初稿做整体评审，输出 CriticReport(target='paper')。"""

SYSTEM = (
    "你是国赛资深评委。请审阅一份建模论文初稿。要点："
    "（1）摘要是否凸显方法和结论；（2）假设是否被正文承接；"
    "（3）模型与求解是否一致、可复现；（4）是否有敏感性分析；"
    "（5）图表是否被正文引用并解读；（6）整体行文是否专业。"
    "总评 0-10，>=8 approved。"
)


def build_prompt(paper, n_figures, n_sensitivity):
    sections = {
        "abstract": paper.abstract, "problem_restatement": paper.problem_restatement,
        "assumptions": paper.assumptions, "notation": paper.notation,
        "model_section": paper.model_section, "solution": paper.solution,
        "sensitivity": paper.sensitivity, "conclusion": paper.conclusion,
    }
    body = "\n\n".join(f"## {k}\n{v[:1000]}" for k, v in sections.items())
    return (
        f"# 章节素材\n{body}\n\n"
        f"# 客观信号\n- 图表数：{n_figures}\n- 敏感性 run 数：{n_sensitivity}\n\n"
        f"请输出 JSON：{{\"target\":\"paper\",\"score\":int,\"issues\":[str],"
        f"\"suggestions\":[str],\"approved\":bool}}。"
    )
