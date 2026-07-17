from __future__ import annotations

import json
from pathlib import Path

from math_agent.config import MODEL_ROUTING
from math_agent.prompts.analyst import SYSTEM, build_prompt
from math_agent.state import DataFileInfo
from math_agent.transport import CompletionRequest, LiteLLMWorkerTransport


def main() -> int:
    problem_path = Path("runs/problem-specs/huazhong_green_logistics_a.json")
    spec = json.loads(problem_path.read_text(encoding="utf-8"))
    prompt = build_prompt(
        spec["title"] + "\n" + "\n".join(spec.get("questions", [])),
        spec.get("background", ""),
        spec.get("questions", []),
        data_files=[DataFileInfo(**item) for item in spec.get("data_files", [])],
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    transport = LiteLLMWorkerTransport()
    try:
        resp = transport.send(
            CompletionRequest(
                model=MODEL_ROUTING["analyst"],
                messages=messages,
                temperature=0.3,
                response_format={"type": "json_object"},
            ),
            timeout_s=90,
        )
        out = Path("runs/problem-specs/analyst_raw_response.txt")
        out.write_text(resp.content, encoding="utf-8")
        print(f"OK {out}")
        print(resp.content[:2000])
        return 0
    finally:
        transport.close()


if __name__ == "__main__":
    raise SystemExit(main())
