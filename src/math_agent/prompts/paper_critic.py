"""PaperCritic：对组装好的论文初稿做整体评审，输出 CriticReport(target='paper')。"""

SYSTEM = (
    "你是国赛资深评委。请审阅一份建模论文初稿。要点："
    "（1）摘要是否凸显方法和结论；（2）假设是否被正文承接；"
    "（3）模型与求解是否一致、可复现；（4）是否有敏感性分析；"
    "（5）图表是否被正文引用并解读；（6）整体行文是否专业。"
    "总评 0-10，>=8 approved。"
    "\n\n"
    "**关键事实核查**：若下文给出『代码运行真实输出』区块，请把它当作唯一可靠的数字事实源。"
    "用语义判断正文中的关键定量结论（成本、占比、敏感度幅度、性能指标等）是否与 stdout 相符。"
    "明显与 stdout 不符的数字（如 stdout 显示 52.7174 但正文写 718）视为编造，"
    "把它逐条列入 issues 并把 approved 设为 False。"
    "合理四舍五入（如 52.7174→52.6、53.7718→53.8）不算编造，不要因此扣分。"
)


def build_prompt(paper, n_figures, n_sensitivity, code_stdout: str = ""):
    sections = {
        "abstract": paper.abstract, "problem_restatement": paper.problem_restatement,
        "assumptions": paper.assumptions, "notation": paper.notation,
        "model_section": paper.model_section, "solution": paper.solution,
        "sensitivity": paper.sensitivity, "conclusion": paper.conclusion,
    }
    body = "\n\n".join(f"## {k}\n{v[:1000]}" for k, v in sections.items())
    stdout_block = ""
    if code_stdout.strip():
        stdout_block = (
            f"\n# 代码运行真实输出（事实源；用于核对正文数字）\n"
            f"```\n{code_stdout[:4000]}\n```\n"
        )
    return (
        f"# 章节素材\n{body}\n\n"
        f"# 客观信号\n- 图表数：{n_figures}\n- 敏感性 run 数：{n_sensitivity}\n"
        f"{stdout_block}\n"
        f"请输出 JSON：{{\"target\":\"paper\",\"score\":int,\"issues\":[str],"
        f"\"suggestions\":[str],\"approved\":bool}}。"
    )
