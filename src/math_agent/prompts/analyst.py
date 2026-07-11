"""Analyst：把题目分解为 ProblemBlueprint（结构化题目理解资产）。"""

SYSTEM = (
    "你是数学建模竞赛队的首席分析师。你的职责是把题目拆解为一份结构化的 ProblemBlueprint，"
    "覆盖：核心任务、每个小问的任务类型、决策变量、目标、约束、指标、数据需求、假设、"
    "建模候选路线、推荐路线与验证计划。"
    "禁止编造未给出的数据；数据需求中 given_or_missing 必须如实标注。"
    "problem_domains 从下列固定集合中选取 1-3 个："
    "['optimization', 'time_series', 'machine_learning', 'graph', "
    "'probability', 'queueing', 'simulation', 'generic']。"
)

# ProblemBlueprint 的 JSON schema 描述（供 LLM 参照输出）
_SCHEMA_HINT = """请输出 JSON，结构与 ProblemBlueprint 一致：
{
  "core_task": str,                          # 题目的核心任务一句话概括
  "subquestions": [                          # 每个小问一条
    {"id": str, "original_text": str,
     "task_type": "evaluation|prediction|optimization|simulation|classification|explanation|strategy|generic",
     "depends_on": [str, ...], "expected_output": str, "success_criteria": [str, ...]}
  ],
  "decision_variables": [                    # 决策变量
    {"name": str, "meaning": str, "domain": str, "unit": str}
  ],
  "objectives": [                            # 目标
    {"direction": "minimize|maximize|evaluate|predict|explain", "description": str, "math_hint": str}
  ],
  "constraints": [                           # 约束
    {"description": str, "math_hint": str, "source": "given|derived|assumed"}
  ],
  "metrics": [                               # 评估指标
    {"name": str, "meaning": str, "direction": "higher_better|lower_better|target|neutral", "unit": str}
  ],
  "data_requirements": [                     # 数据需求（given_or_missing 必须如实标注）
    {"field": str, "meaning": str, "needed_for": str,
     "given_or_missing": "given|missing|derived|unknown", "handling_strategy": str}
  ],
  "assumptions": [                           # 建模假设（至少 5 条）
    {"statement": str, "rationale": str}
  ],
  "problem_domains": [str, ...],
  "modeling_candidates": [                   # 建模候选路线（至少 2 条）
    {"name": str, "route": str, "suitable_for": [str, ...], "data_required": [str, ...],
     "pros": [str, ...], "cons": [str, ...], "risk": str, "recommendation_score": int}
  ],
  "recommended_route": {"route": str, "reason": str, "baseline": str, "fallback": str},
  "validation_plan": [                       # 验证计划
    {"target": str, "method": str, "pass_criteria": str}
  ],
  "risks": [str, ...]
}"""


def build_prompt(problem: str, background: str, questions: list[str],
                 retrieved_context: str = "",
                 critic_feedback=None,
                 data_files=None) -> str:
    qs = "\n".join(f"- {q}" for q in questions) or "（题目本身未列出独立小问）"
    ctx = f"\n{retrieved_context}\n\n" if retrieved_context else ""
    fb = ""
    if critic_feedback is not None:
        issues = "\n".join(f"- {i.problem}" for i in critic_feedback.issues)
        sugs = "\n".join(f"- {s}" for s in critic_feedback.suggestions)
        fb = f"\n# 上一轮 Blueprint Critic 反馈\n问题：\n{issues}\n建议：\n{sugs}\n请据此修正蓝图。\n"
    data_hint = ""
    if data_files:
        from math_agent.prompts._data_hint import build_data_summary_hint
        data_hint = build_data_summary_hint(data_files)
    return (
        f"# 题目\n{problem}\n\n"
        f"# 背景\n{background or '（无）'}\n\n"
        f"# 小问\n{qs}\n\n"
        f"{ctx}{data_hint}{fb}{_SCHEMA_HINT}"
    )
