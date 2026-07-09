"""ModelCritic：从假设合理性、数学严密性、与题目相关性、可计算性 4 维度评分，
并检查模型与 ProblemBlueprint 的对齐情况。"""

SYSTEM = (
    "你是国赛评委。请就给定模型给出 0-10 的整数总评分（>=7 视为通过），"
    "并列出至多 5 个 issues 与至多 5 个 suggestions。"
    "approved 判定规则：只有存在【严重问题】时才 approved=False。"
    "严重问题 = 假设与模型矛盾、方程量纲不一致、模型不可计算、完全偏题、"
    "模型未覆盖 blueprint 中的小问、模型变量/目标/约束与 blueprint 明显不一致。"
    "改进建议（如『可用更优模型』『假设可放宽』）不影响 approved，只写进 suggestions。"
    "重点检查：假设是否被显式承接、方程量纲是否一致、是否存在更优经典模型。"
    "Blueprint 对齐检查：是否覆盖所有小问、变量是否对应 blueprint、"
    "目标函数是否对应 blueprint、约束是否对应 blueprint、"
    "final model 是否包含 baseline、validation plan 是否可执行。"
    "交叉验证 question_coverage：如果模型声称覆盖了某小问，但 equations 中找不到对应的公式，记为 issue。"
)


def _blueprint_context(blueprint) -> str:
    if blueprint is None:
        return ""
    lines = []
    if blueprint.subquestions:
        sq = "\n".join(f"  - [{s.id}] ({s.task_type}) {s.original_text}" for s in blueprint.subquestions)
        lines.append(f"## Blueprint 小问\n{sq}")
    if blueprint.decision_variables:
        dv = "\n".join(f"  - {v.name}: {v.meaning}" for v in blueprint.decision_variables)
        lines.append(f"## Blueprint 决策变量\n{dv}")
    if blueprint.objectives:
        ob = "\n".join(f"  - [{o.direction}] {o.description}" for o in blueprint.objectives)
        lines.append(f"## Blueprint 目标\n{ob}")
    if blueprint.constraints:
        cs = "\n".join(f"  - {c.description}" for c in blueprint.constraints)
        lines.append(f"## Blueprint 约束\n{cs}")
    if blueprint.validation_plan:
        vp = "\n".join(f"  - {v.target}: {v.method}" for v in blueprint.validation_plan)
        lines.append(f"## Blueprint 验证计划\n{vp}")
    return "\n".join(lines)


def _coverage_context(model) -> str:
    if not model.question_coverage:
        return ""
    lines = ["## 模型声称的 question_coverage"]
    for cov in model.question_coverage:
        lines.append(f"  - [{cov.question_id}] {cov.how_answered}")
        if cov.related_equations:
            lines.append(f"    related_equations: {cov.related_equations}")
    return "\n".join(lines)


def build_prompt(problem, assumptions, model, blueprint=None):
    asum = "\n".join(f"- {a.statement}" for a in assumptions)
    eqs = "\n".join(f"  - $$ {e} $$" for e in model.equations)
    vars_ = "\n".join(f"  - {k}: {v}" for k, v in model.variables.items())
    bp = ""
    if blueprint is not None:
        bp = f"\n\n{_blueprint_context(blueprint)}"
    cov = ""
    if model.stage == "final":
        cov = f"\n\n{_coverage_context(model)}"
        bp += "\n\n## 对齐检查要求\nfinal 模型必须包含 baseline，validation plan 必须可执行。"
    return (
        f"# 题目\n{problem}\n\n# 假设\n{asum}\n\n# 模型（{model.stage}）\n"
        f"{model.description}\n方程：\n{eqs}\n变量：\n{vars_}{bp}{cov}\n\n"
        f"请输出 JSON：{{\"target\":\"modeler\",\"score\":int,\"issues\":[{{\"section\":\"general\",\"problem\":str}}, ...],\"suggestions\":[str],\"approved\":bool}}"
    )
