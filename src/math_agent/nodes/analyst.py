from pydantic import BaseModel
from math_agent.llm import complete
from math_agent.config import MODEL_ROUTING
from math_agent.prompts.analyst import SYSTEM, build_prompt
from math_agent.state import Assumption, MathModelingState


class AnalystOutput(BaseModel):
    assumptions: list[Assumption]


def analyst_node(state: MathModelingState) -> dict:
    prompt = build_prompt(state.problem, state.background, state.questions)
    out: AnalystOutput = complete(
        prompt,
        schema=AnalystOutput,
        system=SYSTEM,
        model=MODEL_ROUTING["analyst"],
    )
    return {"assumptions": out.assumptions}
