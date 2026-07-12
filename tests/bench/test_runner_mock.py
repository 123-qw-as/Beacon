import json

from math_agent.bench.runner import run_bench, BenchReport


def test_run_bench_returns_one_case_per_problem(workdir, install_bench_mocks):
    rep = run_bench(out_dir=workdir)
    assert isinstance(rep, BenchReport)
    assert {c.problem_id for c in rep.cases} == {"2022_A", "2023_B"}


def test_run_bench_marks_pass_when_expectation_met(workdir, install_bench_mocks):
    rep = run_bench(out_dir=workdir)
    for case in rep.cases:
        assert case.passed, f"{case.problem_id}: {case.failures}"


def test_run_bench_marks_fail_when_keyword_missing(workdir, install_bench_mocks_missing_keyword):
    rep = run_bench(out_dir=workdir)
    fails = [c for c in rep.cases if not c.passed]
    assert fails, "expected at least one FAIL when keyword missing"
    assert any("missing keyword" in " ".join(c.failures) for c in fails)


def test_run_bench_marks_fail_when_overall_below_threshold(workdir, install_bench_mocks_low_overall):
    rep = run_bench(out_dir=workdir)
    fails = [c for c in rep.cases if not c.passed]
    assert fails
    assert any("overall" in " ".join(c.failures) for c in fails)


def test_run_bench_writes_report_json(workdir, install_bench_mocks):
    run_bench(out_dir=workdir)
    blob = json.loads((workdir / "bench_report.json").read_text(encoding="utf-8"))
    assert "cases" in blob and len(blob["cases"]) == 2


def test_evaluate_supports_fully_serialized_state():
    from math_agent.bench.runner import _evaluate
    state = {
        "evaluation": {"overall": 8.0},
        "paper": {
            "abstract": "覆盖 无人机 鲁棒",
            "model_section": "覆盖",
            "solution": "无人机",
            "conclusion": "鲁棒",
        },
        "sensitivity_runs": [{"parameter": "x"}],
        "figures": [{"path": "x.png"}],
    }
    case = _evaluate("dict", state, {
        "min_overall": 6.5,
        "must_contain_keywords": ["覆盖", "无人机", "鲁棒"],
        "must_have_sensitivity": True,
        "must_have_figures": True,
    })
    assert case.passed, case.failures
