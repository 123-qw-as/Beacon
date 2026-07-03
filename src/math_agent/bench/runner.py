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


def _evaluate(case_id: str, final_state, expect: dict, *,
              output_dir: Path | None = None) -> BenchCase:
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

    # ---- Plan D Task 6.2: 可选验证字段（automation gate）----
    # 以下检查全部可选：expect dict 不含对应 key 时跳过，保持向后兼容。
    _check_plan_d_fields(final_state, expect, output_dir, failures)

    return BenchCase(problem_id=case_id, overall=overall,
                     passed=not failures, failures=failures)


def _check_plan_d_fields(final_state, expect: dict,
                         output_dir: Path | None, failures: list[str]) -> None:
    """Plan D 验证：code_artifacts / figures / derivation / references / PDF / xelatex / paper critic。

    所有字段可选；缺失 expect key 则跳过。output_dir 为空时跳过需要读文件的检查。
    """
    artifacts = _get(final_state, "code_artifacts") or []

    if "min_code_artifacts_total" in expect:
        total = len(artifacts)
        if total < expect["min_code_artifacts_total"]:
            failures.append(
                f"code_artifacts_total {total} < {expect['min_code_artifacts_total']}")

    if "min_code_artifacts_success" in expect:
        success = sum(1 for a in artifacts if _get(a, "success", False))
        if success < expect["min_code_artifacts_success"]:
            failures.append(
                f"code_artifacts_success {success} < {expect['min_code_artifacts_success']}")

    if "min_figures" in expect:
        fig_count = len(_get(final_state, "figures") or [])
        if fig_count < expect["min_figures"]:
            failures.append(f"figures {fig_count} < {expect['min_figures']}")

    if "min_derivation_steps" in expect:
        models = _get(final_state, "model_versions") or []
        if models:
            last = models[-1]
            steps = len(getattr(last, "derivation_steps", []) or [])
            if steps < expect["min_derivation_steps"]:
                failures.append(
                    f"derivation_steps {steps} < {expect['min_derivation_steps']}")
        # 无模型版本时优雅跳过（不记失败）——basic 阶段可能尚无推导链。

    if "min_references" in expect:
        paper = _get(final_state, "paper")
        refs_text = getattr(paper, "references", "") if paper is not None else ""
        refs_text = refs_text or ""
        # 优先按 [N] 模式计数；无匹配则按非空行计数。
        ref_lines = [l.strip() for l in refs_text.split("\n")
                     if l.strip() and l.strip()[0] == "["]
        if not ref_lines:
            ref_lines = [l.strip() for l in refs_text.split("\n") if l.strip()]
        ref_count = len(ref_lines)
        if ref_count < expect["min_references"]:
            failures.append(f"references {ref_count} < {expect['min_references']}")

    if "max_paper_critic_issues" in expect:
        critics = _get(final_state, "critic_reports") or []
        paper_critic = next(
            (r for r in reversed(critics)
             if _get(r, "target", "") == "paper"), None)
        if paper_critic is not None:
            issues = len(getattr(paper_critic, "issues", []) or [])
            if issues > expect["max_paper_critic_issues"]:
                failures.append(
                    f"paper_critic_issues {issues} > {expect['max_paper_critic_issues']}")
        # 无 paper critic 时优雅跳过。

    # 以下两项需要 output_dir 指向真实编译产物；缺文件则跳过（不记失败）。
    if output_dir is not None:
        _check_paper_pdf(output_dir, expect, failures)
        _check_xelatex_log(output_dir, expect, failures)


def _check_paper_pdf(output_dir: Path, expect: dict,
                     failures: list[str]) -> None:
    if "min_paper_pdf_pages" not in expect:
        return
    pdf_path = output_dir / "paper.pdf"
    if not pdf_path.exists():
        return  # 无 PDF（latex 可能被 mock）→ 跳过
    try:
        from pypdf import PdfReader
        pages = len(PdfReader(str(pdf_path)).pages)
    except Exception as exc:  # 损坏的 PDF 等异常 → 记失败而非崩
        failures.append(f"paper_pdf unreadable: {exc}")
        return
    if pages < expect["min_paper_pdf_pages"]:
        failures.append(
            f"paper_pdf_pages {pages} < {expect['min_paper_pdf_pages']}")


def _check_xelatex_log(output_dir: Path, expect: dict,
                       failures: list[str]) -> None:
    if "xelatex_must_have_zero_errors" not in expect:
        return
    if not expect["xelatex_must_have_zero_errors"]:
        return
    # 在 output_dir 下寻找 compile.log / xelatex.log / *.log。
    log_path = None
    for name in ("compile.log", "xelatex.log"):
        cand = output_dir / name
        if cand.exists():
            log_path = cand
            break
    if log_path is None:
        log_logs = sorted(output_dir.glob("*.log"))
        if log_logs:
            log_path = log_logs[0]
    if log_path is None:
        return  # 无日志 → 跳过
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        failures.append(f"compile.log unreadable: {exc}")
        return
    # TeX 错误行以 "! " 开头，或日志含 "Error:"。
    error_lines = [ln for ln in log_text.splitlines()
                   if ln.startswith("! ") or "Error:" in ln]
    if error_lines:
        failures.append(
            f"xelatex errors: {len(error_lines)} (e.g. {error_lines[0][:80]})")


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
        cases.append(_evaluate(case_id, final, expect, output_dir=case_out))

    report = BenchReport(cases=cases)
    (out_dir / "bench_report.json").write_text(
        json.dumps({"cases": [asdict(c) for c in cases]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
