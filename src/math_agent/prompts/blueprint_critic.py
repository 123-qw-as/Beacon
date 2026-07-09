"""BlueprintCritic：审查 analyst 产出的 ProblemBlueprint，确保题目理解无误后才进入建模。"""

SYSTEM = (
    "你是数学建模竞赛队的审题专家。你的职责是审查 ProblemBlueprint 是否准确、完整地理解了题目。"
    "只有存在【严重问题】时才 approved=False。"
    "严重问题 = 遗漏小问、任务类型判断错误、关键变量/目标/约束缺失或空泛、"
    "数据需求虚构（题目给出的标为 missing、缺失的标为 given）、"
    "推荐路线无法回答全部小问、缺少 baseline 或验证计划。"
    "改进建议（如『假设可更精确』）不影响 approved，只写进 suggestions。"
)


def build_prompt(blueprint_json: str, problem: str, questions: list[str]) -> str:
    qs = "\n".join(f"- {q}" for q in questions) or "（题目本身未列出独立小问）"
    return (
        f"# 原始题目\n{problem}\n\n"
        f"# 题目小问\n{qs}\n\n"
        f"# 待审查的 ProblemBlueprint\n{blueprint_json}\n\n"
        f"请审查上述蓝图，重点检查：\n"
        f"1. 是否遗漏了题目中的任何小问\n"
        f"2. 每个 subquestion 的 task_type 是否判断正确\n"
        f"3. decision_variables / objectives / constraints 是否空泛或缺失\n"
        f"4. data_requirements 中的 given_or_missing 是否与题目实际相符（有无虚构数据）\n"
        f"5. recommended_route 是否能回答全部小问\n"
        f"6. 是否缺少 baseline 或 validation_plan\n\n"
        f"请输出 JSON：{{"
        f"\"target\":\"analyst\","
        f"\"score\":int,  # 0-10"
        f"\"issues\":[{{\"section\":\"general\",\"problem\":str}}, ...],"
        f"\"suggestions\":[str, ...],"
        f"\"approved\":bool"
        f"}}"
    )
