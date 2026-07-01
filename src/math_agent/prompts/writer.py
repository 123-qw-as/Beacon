"""Writer prompt：模板化版本。

设计原则：
- 模板与 paper.md.j2 共用 templates/ 目录与同一 Jinja2 Environment
- Python 侧只负责构建 view dict；规约/字数预算/禁用词等全部写在模板里
- 输出契约与 PaperSections schema 严格一致
"""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from math_agent.state import MathModelingState

SYSTEM = (
    "你是国赛 O 奖论文主笔。严格遵守用户消息中的 IRON RULES 与字数预算。"
    "禁止编造数据；禁止占位符；每段必须有可被证伪的论点。"
    "输出严格的 JSON，键名与给定 schema 完全一致，不得增减键。"
)

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=select_autoescape([]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def build_prompt(state: MathModelingState, *, retrieved_context: str = "") -> str:
    tmpl = _env.get_template("writer_prompt.md.j2")
    rendered = tmpl.render(
        problem=state.problem,
        assumptions=state.assumptions,
        model_versions=state.model_versions,
        code_artifacts=state.code_artifacts,
        sensitivity_runs=state.sensitivity_runs,
        figures=state.figures,
        prior_critic=state.latest_critic("paper"),
    )
    if retrieved_context:
        # 追加到模板渲染结果末尾，避免改动模板文件
        rendered = rendered + "\n\n" + retrieved_context
    return rendered
