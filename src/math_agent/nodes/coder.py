from __future__ import annotations
from pathlib import Path
import tempfile

from pydantic import BaseModel
from math_agent.llm import complete
from math_agent.config import MODEL_ROUTING
from math_agent.prompts.coder import SYSTEM, build_prompt
from math_agent.state import MathModelingState, CodeArtifact
from math_agent.tools.runner import run_python


class CoderDraft(BaseModel):
    purpose: str
    code: str


MAX_CODE_RETRIES = 1  # 一次失败后再给一次机会，避免成本失控


def coder_node(state: MathModelingState) -> dict:
    model = state.latest_model()
    workdir = Path(state.output_dir) if state.output_dir else Path(tempfile.mkdtemp(prefix="math_agent_"))
    workdir.mkdir(parents=True, exist_ok=True)

    artifacts: list[CodeArtifact] = []
    prev_err: str | None = None
    for attempt in range(MAX_CODE_RETRIES + 1):
        draft: CoderDraft = complete(
            build_prompt(model, prev_err),
            schema=CoderDraft,
            system=SYSTEM,
            model=MODEL_ROUTING["coder"],
        )
        result = run_python(draft.code, workdir=workdir / f"attempt_{attempt}")
        artifacts.append(
            CodeArtifact(
                purpose=draft.purpose,
                code=draft.code,
                stdout=result.stdout,
                stderr=result.stderr,
                success=result.success,
                artifact_paths=result.artifact_paths,
            )
        )
        if result.success:
            break
        prev_err = result.stderr

    return {"code_artifacts": artifacts}
