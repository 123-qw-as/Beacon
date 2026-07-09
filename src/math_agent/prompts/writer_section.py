"""Writer 分章节 prompt 派发器（Plan D Phase 2）。

设计：
- Pass 1（大纲）：一次 LLM 调用产出 WriterOutline，每章 2-3 句论点锚点。
- Pass 2（分章）：7 个分组，每组一次 LLM 调用，只输出该组对应的 PaperSections 子集。
- 重试时（writer_iteration > 0）跳过大纲，只重跑 paper_critic 标记的分组。

本模块只负责 prompt 构建与分组元数据，不做 LLM 调用（调用在 writer_node 内）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel

from math_agent.state import CriticIssue, MathModelingState


# ---------------------------------------------------------------------------
# 分组元数据
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SectionGroup:
    name: str                       # "abstract_problem"
    template: str                   # "writer_section_abstract_problem.md.j2"
    fields: tuple[str, ...]         # ("abstract", "problem_restatement", "keywords")


# 7 个分组（决策 C，Plan Task 2.1）
_WRITER_SECTIONS: list[SectionGroup] = [
    SectionGroup("abstract_problem", "writer_section_abstract_problem.md.j2",
                 ("abstract", "problem_restatement", "keywords")),
    SectionGroup("assumptions_notation", "writer_section_assumptions_notation.md.j2",
                 ("assumptions", "notation")),
    SectionGroup("model", "writer_section_model.md.j2", ("model_section",)),
    SectionGroup("solution", "writer_section_solution.md.j2", ("solution",)),
    SectionGroup("sensitivity", "writer_section_sensitivity.md.j2", ("sensitivity",)),
    SectionGroup("conclusion", "writer_section_conclusion.md.j2", ("conclusion",)),
    SectionGroup("references", "writer_section_references.md.j2", ("references",)),
]


def writer_sections() -> list[SectionGroup]:
    """返回 7 个分组的有序列表（副本语义，元素本身 frozen）。"""
    return list(_WRITER_SECTIONS)


# CriticIssue.section（paper 章节字段名）→ SectionGroup.name
# "general" 与未识别值由 _sections_to_rewrite 兜底返回全部。
_SECTION_FIELD_TO_GROUP: dict[str, str] = {
    "abstract": "abstract_problem",
    "problem_restatement": "abstract_problem",
    "keywords": "abstract_problem",
    "assumptions": "assumptions_notation",
    "notation": "assumptions_notation",
    "model_section": "model",
    "solution": "solution",
    "sensitivity": "sensitivity",
    "conclusion": "conclusion",
    "references": "references",
}


# ---------------------------------------------------------------------------
# 输出 schema（每组一个精简 Pydantic 模型，只含该组字段）
# ---------------------------------------------------------------------------

class _AbstractProblemOut(BaseModel):
    abstract: str = ""
    problem_restatement: str = ""
    keywords: str = ""


class _AssumptionsNotationOut(BaseModel):
    assumptions: str = ""
    notation: str = ""


class _ModelOut(BaseModel):
    model_section: str = ""


class _SolutionOut(BaseModel):
    solution: str = ""


class _SensitivityOut(BaseModel):
    sensitivity: str = ""


class _ConclusionOut(BaseModel):
    conclusion: str = ""


class _ReferencesOut(BaseModel):
    references: str = ""


_SCHEMA_FOR_GROUP: dict[str, type[BaseModel]] = {
    "abstract_problem": _AbstractProblemOut,
    "assumptions_notation": _AssumptionsNotationOut,
    "model": _ModelOut,
    "solution": _SolutionOut,
    "sensitivity": _SensitivityOut,
    "conclusion": _ConclusionOut,
    "references": _ReferencesOut,
}


def schema_for_group(group_name: str) -> type[BaseModel]:
    return _SCHEMA_FOR_GROUP[group_name]


# ---------------------------------------------------------------------------
# 大纲 schema
# ---------------------------------------------------------------------------

class WriterOutline(BaseModel):
    """Pass 1 产出：每章 2-3 句论点锚点。9 个字符串字段，默认空。"""
    abstract: str = ""
    problem_restatement: str = ""
    assumptions: str = ""
    notation: str = ""
    model_section: str = ""
    solution: str = ""
    sensitivity: str = ""
    conclusion: str = ""
    references: str = ""


# ---------------------------------------------------------------------------
# Jinja2 环境（与 prompts/writer.py 共用 templates/ 目录）
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=select_autoescape([]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _group_by_name(name: str) -> SectionGroup:
    for g in _WRITER_SECTIONS:
        if g.name == name:
            return g
    raise KeyError(f"unknown section group: {name}")


import re as _re
# 提取 stdout 中的 RESULT: 行和 key=value 数值对，供 writer 引用
_RESULT_NUM_RE = _re.compile(
    r"(?:RESULT:\s*\S+\s+)?((?:\w+=\s*-?\d+\.?\d*(?:[eE][+-]?\d+)?\s*)+)",
    _re.MULTILINE,
)


def _extract_available_numbers(state: MathModelingState) -> str:
    """从 code_artifacts.stdout 和 sensitivity_runs 提取可用数值清单。

    让 writer "只能用这些数值"，比 IRON RULE 的禁令更有效。
    """
    lines: list[str] = []
    for a in state.code_artifacts:
        if not a.success or not a.stdout:
            continue
        # 提取 RESULT: 行
        for m in _RESULT_NUM_RE.finditer(a.stdout):
            line = m.group(0).strip()
            if line:
                lines.append(f"  [{a.purpose}] {line}")
    for r in state.sensitivity_runs:
        lines.append(f"  [sensitivity] {r.parameter}={r.values} → {r.metric}={r.results}")
    if not lines:
        return ""
    return "\n".join(lines[:30])  # 上限 30 行，控 prompt 长度


# ---------------------------------------------------------------------------
# prompt 构建器
# ---------------------------------------------------------------------------

def build_outline_prompt(state: MathModelingState, *, retrieved_context: str = "") -> str:
    """渲染大纲模板（Pass 1）。"""
    tmpl = _env.get_template("writer_outline.md.j2")
    rendered = tmpl.render(
        problem=state.problem,
        model_versions=state.model_versions,
        problem_blueprint=state.problem_blueprint,
    )
    if retrieved_context:
        rendered = rendered + "\n\n" + retrieved_context
    return rendered


def build_section_prompt(
    group_name: str,
    state: MathModelingState,
    outline: WriterOutline,
    *,
    prior_critic=None,
    retrieved_context: str = "",
    references_list=None,
) -> str:
    """渲染某分组模板（Pass 2 单组）。

    prior_critic: CriticReport 或 None。模板内 {% if prior_critic %} 控制是否渲染反馈块。
    references_list: list[Reference] 或 None，仅 references 分组需要（Plan D Phase 5）。
    """
    group = _group_by_name(group_name)
    # 标准视图 dict：与原 writer_prompt 一致的全量素材，由各模板按需取用。
    latest_model = state.latest_model()
    view = {
        "problem": state.problem,
        "assumptions": state.assumptions,
        "model_versions": state.model_versions,
        "code_artifacts": state.code_artifacts,
        "sensitivity_runs": state.sensitivity_runs,
        "figures": state.figures,
        "prior_critic": prior_critic if prior_critic is not None else state.latest_critic("paper"),
        "problem_domains": state.problem_domains,  # Plan D：analyst 输出
        "references_list": references_list or [],  # Plan D：检索到的真实文献
        "available_numbers": _extract_available_numbers(state),  # 可引用的数值清单
        # Problem Blueprint 对齐（P2 Step 13）
        "problem_blueprint": state.problem_blueprint,
        "question_coverage": latest_model.question_coverage if latest_model else [],
        "objective_mapping": latest_model.objective_mapping if latest_model else [],
        "constraint_mapping": latest_model.constraint_mapping if latest_model else [],
        "validation_mapping": latest_model.validation_mapping if latest_model else [],
        "model_code_reports": state.model_code_reports,
        # 大纲锚点：每章一个 outline_<field> 变量
        "outline_abstract": outline.abstract,
        "outline_problem_restatement": outline.problem_restatement,
        "outline_assumptions": outline.assumptions,
        "outline_notation": outline.notation,
        "outline_model_section": outline.model_section,
        "outline_solution": outline.solution,
        "outline_sensitivity": outline.sensitivity,
        "outline_conclusion": outline.conclusion,
        "outline_references": outline.references,
    }
    tmpl = _env.get_template(group.template)
    rendered = tmpl.render(**view)
    # 注入 Blueprint 写作约束（P2 Step 13）
    if state.problem_blueprint is not None:
        bp = state.problem_blueprint
        constraints_lines = [
            "\n---\n## Blueprint 写作约束",
            f"- 核心任务：{bp.core_task}",
        ]
        if bp.subquestions:
            sq = "; ".join(f"[{s.id}] {s.original_text}" for s in bp.subquestions)
            constraints_lines.append(f"- 必须覆盖的小问：{sq}")
            constraints_lines.append("- 如果某个小问没有被模型或代码支持，必须写成局限性，不能编造结果。")
        if bp.decision_variables:
            dv = ", ".join(v.name for v in bp.decision_variables)
            constraints_lines.append(f"- 符号说明优先使用决策变量：{dv}")
        if bp.objectives:
            ob = "; ".join(o.description for o in bp.objectives)
            constraints_lines.append(f"- 模型建立必须对应目标：{ob}")
        if bp.constraints:
            cs = "; ".join(c.description for c in bp.constraints)
            constraints_lines.append(f"- 模型建立必须对应约束：{cs}")
        if bp.metrics:
            mt = ", ".join(m.name for m in bp.metrics)
            constraints_lines.append(f"- 求解与结果必须对应指标：{mt}")
        if bp.validation_plan:
            vp = "; ".join(v.target for v in bp.validation_plan)
            constraints_lines.append(f"- 求解与结果必须对应验证计划：{vp}")
        rendered = rendered + "\n".join(constraints_lines)
    # 注入可用数值清单：让 writer 只能引用这些数值，不得编造
    numbers = view["available_numbers"]
    if numbers:
        rendered = rendered + "\n\n---\n## 可引用的数值清单（只能用以下数值，不得使用其他数值）\n" + numbers
    if retrieved_context:
        rendered = rendered + "\n\n" + retrieved_context
    return rendered


# ---------------------------------------------------------------------------
# 重试分组计算
# ---------------------------------------------------------------------------

def _sections_to_rewrite(issues: list[CriticIssue]) -> list[str]:
    """把 CriticIssue.section 映射到需重跑的 SectionGroup.name 列表。

    - 任一 issue.section == "general" 或未被 _SECTION_FIELD_TO_GROUP 覆盖 → 返回全部 7 组（安全兜底）。
    - 否则：逐条映射并去重，保持 _WRITER_SECTIONS 原顺序。
    """
    if not issues:
        return [g.name for g in _WRITER_SECTIONS]

    group_names: list[str] = []
    seen: set[str] = set()
    for iss in issues:
        mapped = _SECTION_FIELD_TO_GROUP.get(iss.section)
        if mapped is None:
            # general 或未识别 → 安全兜底，全部重跑
            return [g.name for g in _WRITER_SECTIONS]
        if mapped not in seen:
            seen.add(mapped)
            group_names.append(mapped)

    # 保持 _WRITER_SECTIONS 原始顺序，便于可预测的调用序列
    ordered = [g.name for g in _WRITER_SECTIONS if g.name in seen]
    return ordered
