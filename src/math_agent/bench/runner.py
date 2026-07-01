"""回归基准 runner：真跑 build_graph，按 expectations.json 判定通过。

不引入测试库；mock 模式由 tests/bench/conftest.py 提供 fixture，本模块对此无感。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from math_agent.graph import build_graph
from math_agent.state import HumanDecision


_BENCH_ROOT = Path(__file__).resolve().parent
_PROBLEMS_DIR = _BENCH_ROOT / "problems"
_EXPECTATIONS = _BENCH_ROOT / "expectations.json"


@dataclass
class BenchCase:
    problem_id: str
    overall: float
    passed: bool
    failures: list[str] = field(default_factory=list)


@dataclass
class BenchReport:
    cases: list[BenchCase]


def _get(state, key, default=None):
    """final state 既可能是 dict 也可能是 pydantic-like；统一访问。"""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _evaluate(case_id: str, final_state, expect: dict) -> BenchCase:
    failures: list[str] = []
    evaluation = _get(final_state, "evaluation")
    overall = float(evaluation.overall) if evaluation is not None else 0.0
    if overall < expect["min_overall"]:
        failures.append(f"overall {overall} < {expect['min_overall']}")

    paper = _get(final_state, "paper")
    if paper is None:
        text = ""
    else:
        text = " ".join([
            getattr(paper, "abstract", "") or "",
            getattr(paper, "model_section", "") or "",
            getattr(paper, "solution", "") or "",
            getattr(paper, "conclusion", "") or "",
        ])
    for kw in expect.get("must_contain_keywords", []):
        if kw not in text:
            failures.append(f"missing keyword: {kw}")
    if expect.get("must_have_sensitivity") and not _get(final_state, "sensitivity_runs"):
        failures.append("missing sensitivity_runs")
    if expect.get("must_have_figures") and not _get(final_state, "figures"):
        failures.append("missing figures")
    return BenchCase(problem_id=case_id, overall=overall,
                     passed=not failures, failures=failures)


def run_bench(*, out_dir: str | Path) -> BenchReport:
    """真跑每道题；caller（pytest mock fixture 或 CLI 真 API key）负责提供 LLM。

    本函数不知道 LLM 是真是假——它只 invoke graph 并判定结果。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expectations = json.loads(_EXPECTATIONS.read_text(encoding="utf-8"))
    cases: list[BenchCase] = []

    for problem_path in sorted(_PROBLEMS_DIR.glob("*.json")):
        case_id = problem_path.stem
        expect = expectations[case_id]
        problem = json.loads(problem_path.read_text(encoding="utf-8"))
        case_out = out_dir / case_id
        case_out.mkdir(parents=True, exist_ok=True)

        graph = build_graph()  # bench 不带 checkpointer / interrupt
        final = graph.invoke({
            "problem": problem["title"] + "\n" + "\n".join(problem["questions"]),
            "background": problem.get("background", ""),
            "questions": problem["questions"],
            "stage_target": "basic", "iteration": 0,
            "output_dir": str(case_out),
            "human_decision": HumanDecision(approved=True).model_dump(),
        })
        cases.append(_evaluate(case_id, final, expect))

    report = BenchReport(cases=cases)
    (out_dir / "bench_report.json").write_text(
        json.dumps({"cases": [asdict(c) for c in cases]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
