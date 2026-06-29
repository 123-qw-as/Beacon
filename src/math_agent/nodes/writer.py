from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from math_agent.llm import complete
from math_agent.config import MODEL_ROUTING
from math_agent.prompts.writer import SYSTEM, build_prompt
from math_agent.state import MathModelingState, PaperSections


def writer_node(state: MathModelingState) -> dict:
    out: PaperSections = complete(
        build_prompt(state),
        schema=PaperSections,
        system=SYSTEM,
        model=MODEL_ROUTING["writer"],
    )
    return {
        "paper": out,
        "writer_iteration": state.writer_iteration + 1,
    }


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=select_autoescape([]))


def render_markdown(state: MathModelingState) -> str:
    tmpl = _env.get_template("paper.md.j2")
    return tmpl.render(
        problem=state.problem,
        paper=state.paper,
        code_artifacts=state.code_artifacts,
        figures=state.figures,
        sensitivity_runs=state.sensitivity_runs,
    )
