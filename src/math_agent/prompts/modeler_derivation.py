"""Modeler 推导链 prompt：6 步推导，每步把已完成的前序步骤喂回 LLM。

Plan D Phase 4：
- final 阶段才运行；basic / improved 不运行（derivation_steps 默认空）。
- 每步一次 LLM 调用（schema=DerivationStep），共 6 步。
- 跑完 6 步后做一次 self-consistency gate（schema=ConsistencyCheck），
  若不连贯则把问题写入 ModelVersion.derivation_notes。
"""

from pydantic import BaseModel, Field

from math_agent.state import ModelVersion, DerivationStep

# 6 步推导的固定顺序（feed-forward：每步看到前面所有步的结果）
DERIVATION_STEPS = [
    "motivation",        # 1. 动机：为什么用这个模型族
    "math_statement",    # 2. 数学陈述：模型族形式化
    "param_estimation",  # 3. 参数估计：MLE / 矩估计
    "constraints",       # 4. 约束推导：定常性 / 可解性 → 参数约束
    "transformation",    # 5. 等价变换：Markov 形式 / 状态空间
    "solution",          # 6. 求解：解析解 / 数值方法 / 滤波
]

# Human-readable labels for each step kind, shown to the LLM
_STEP_LABELS = {
    "motivation": "模型选择动机",
    "math_statement": "数学陈述",
    "param_estimation": "参数估计",
    "constraints": "约束推导",
    "transformation": "等价变换",
    "solution": "求解",
}

_STEP_GUIDANCE = {
    "motivation": "为什么选择这个模型族？与题目结构的对应关系是什么？相比朴素模型有何优势？",
    "math_statement": "给出模型族的严格数学形式化（含下标、求和、条件），用 LaTeX。",
    "param_estimation": "参数如何估计？MLE / 矩估计 / 贝叶斯？给出目标函数或估计方程。",
    "constraints": "从模型性质（定常性 / 可解性 / 稳定性）推导出参数约束条件。",
    "transformation": "等价变换：是否能化为 Markov 形式 / 状态空间 / 标准型？给出变换式。",
    "solution": "如何求解？解析解 / 数值方法 / 滤波递推？给出求解公式或算法步骤。",
}


class ConsistencyCheck(BaseModel):
    """推导链自洽性检查结果（self-consistency gate 产出）。"""
    coherent: bool
    issues: list[str] = Field(default_factory=list)


def build_derivation_prompt(model: ModelVersion, step_kind: str,
                            completed_steps: list[DerivationStep]) -> str:
    """构造单步推导 prompt。把已完成的前序步骤喂回，保证 step 间逻辑连贯。"""
    prev = "\n".join(
        f"[{i+1}] {s.title}: {s.statement} → {s.result}"
        for i, s in enumerate(completed_steps)
    ) or "（这是第一步，无前序）"
    label = _STEP_LABELS.get(step_kind, step_kind)
    guidance = _STEP_GUIDANCE.get(step_kind, "")
    return (
        f"# 模型\n{model.description}\n\n方程：{'; '.join(model.equations)}\n\n"
        f"# 已完成推导步骤\n{prev}\n\n"
        f"# 当前步骤：{label}\n{guidance}\n\n"
        f"请输出 JSON：{{\"title\": str, \"motivation\": str, \"statement\": str, \"result\": str}}。"
        f"title 用简短中文标签，statement 含 inline LaTeX，result 给出推导结论。"
    )


def build_consistency_prompt(model: ModelVersion,
                             completed_steps: list[DerivationStep]) -> str:
    """Self-consistency gate：回看整个推导链，检查逻辑连贯性。"""
    chain = "\n".join(
        f"[{i+1}] {s.title}: {s.statement} → {s.result}"
        for i, s in enumerate(completed_steps)
    )
    return (
        f"# 模型\n{model.description}\n\n方程：{'; '.join(model.equations)}\n\n"
        f"# 完整推导链\n{chain}\n\n"
        f"请审查上述推导链的逻辑连贯性：步骤间是否有矛盾？假设是否一致？结论是否由前提推出？\n"
        f"请输出 JSON：{{\"coherent\": bool, \"issues\": [str, ...]}}。"
        f"coherent=true 表示逻辑连贯；issues 列出发现的问题（为空则 coherent=true）。"
    )
