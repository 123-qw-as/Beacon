"""ModelCritic：从假设合理性、数学严密性、与题目相关性、可计算性 4 维度评分。"""

SYSTEM = (
    "你是国赛评委。请就给定模型给出 0-10 的整数总评分（>=7 视为通过），"
    "并列出至多 5 个 issues 与至多 5 个 suggestions。"
    "approved 判定规则：只有存在【严重问题】时才 approved=False。"
    "严重问题 = 假设与模型矛盾、方程量纲不一致、模型不可计算、完全偏题。"
    "改进建议（如『可用更优模型』『假设可放宽』）不影响 approved，只写进 suggestions。"
    "重点检查：假设是否被显式承接、方程量纲是否一致、是否存在更优经典模型。"
)


def build_prompt(problem, assumptions, model):
    asum = "\n".join(f"- {a.statement}" for a in assumptions)
    eqs = "\n".join(f"  - $$ {e} $$" for e in model.equations)
    return (
        f"# 题目\n{problem}\n\n# 假设\n{asum}\n\n# 模型（{model.stage}）\n"
        f"{model.description}\n方程：\n{eqs}\n\n"
        f"请输出 JSON：{{\"target\":\"modeler\",\"score\":int,\"issues\":[{{\"section\":\"general\",\"problem\":str}}, ...],\"suggestions\":[str],\"approved\":bool}}"
    )
