from __future__ import annotations

import json
from pathlib import Path

from math_agent.nodes.analyst import analyst_node
from math_agent.state import DataFileInfo, MathModelingState


def main() -> int:
    problem_path = Path("runs/problem-specs/huazhong_green_logistics_a.json")
    spec = json.loads(problem_path.read_text(encoding="utf-8"))
    state = MathModelingState(
        problem=spec["title"] + "\n" + "\n".join(spec.get("questions", [])),
        background=spec.get("background", ""),
        questions=spec.get("questions", []),
        data_dir=spec.get("data_dir"),
        data_files=[DataFileInfo(**item) for item in spec.get("data_files", [])],
    )
    delta = analyst_node(state)
    blueprint = delta["problem_blueprint"]
    print("OK")
    print(blueprint.model_dump_json(indent=2, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
