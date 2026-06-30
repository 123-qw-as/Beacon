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
