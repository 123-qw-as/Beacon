"""Analyst：把题目分解为结构化假设清单。"""

SYSTEM = (
    "你是数学建模竞赛队的首席分析师。你要把题目拆解为"
    "（1）核心问题列表 （2）建模所需的假设清单（每条说明依据）。"
    "禁止编造未给出的数据。"
)


def build_prompt(problem: str, background: str, questions: list[str],
                 retrieved_context: str = "") -> str:
    qs = "\n".join(f"- {q}" for q in questions) or "（题目本身未列出独立小问）"
    ctx = f"\n{retrieved_context}\n\n" if retrieved_context else ""
    return (
        f"# 题目\n{problem}\n\n"
        f"# 背景\n{background or '（无）'}\n\n"
        f"# 小问\n{qs}\n\n"
        f"{ctx}"
        f"请输出 JSON：{{\n"
        f"  \"assumptions\": [{{\"statement\": str, \"rationale\": str}}, ...]\n"
        f"}}，至少 5 条假设。"
    )
