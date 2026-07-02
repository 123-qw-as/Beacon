"""Plan D Task 6.2：_evaluate 的 Plan D 验证字段单测。

直接调用 _evaluate（不经 run_bench），用精心构造的 final_state dict 验证
新增的可选 expect 字段（code_artifacts / figures / derivation / references /
paper_critic / PDF / xelatex log）。这些检查全部可选——expect 不含 key 则跳过，
保证对旧 expectations（2022_A / 2023_B）向后兼容。
"""
from __future__ import annotations

from math_agent.bench.runner import _evaluate
from math_agent.state import (
    CriticReport, CriticIssue, DerivationStep, ModelVersion, PaperSections,
)


def _art(*, success: bool):
    """单个 code_artifact：带 .success 属性的轻量对象。"""
    return type("A", (), {"success": success})()


def _evaluation(overall: float = 8.0):
    """带 .overall 的 evaluation，使 min_overall 检查通过，隔离新字段检查。"""
    return type("E", (), {"overall": overall})()


def test_evaluate_min_code_artifacts_success_fail():
    state = {"code_artifacts": [_art(success=True), _art(success=False)],
             "evaluation": _evaluation()}
    # 1 个成功，阈值 2 → 失败
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "min_code_artifacts_success": 2})
    assert not case.passed
    assert any("success" in f or "code_artifacts_success" in f
               for f in case.failures)


def test_evaluate_min_code_artifacts_success_pass():
    state = {"code_artifacts": [_art(success=True), _art(success=False)],
             "evaluation": _evaluation()}
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "min_code_artifacts_success": 1})
    assert case.passed


def test_evaluate_min_code_artifacts_total_fail():
    state = {"code_artifacts": [_art(success=True)]}
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "min_code_artifacts_total": 5})
    assert not case.passed
    assert any("total" in f.lower() for f in case.failures)


def test_evaluate_min_code_artifacts_total_empty():
    # 无 code_artifacts 字段 → 当作空列表
    state = {"evaluation": _evaluation()}
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "min_code_artifacts_total": 1})
    assert not case.passed


def test_evaluate_min_figures_fail():
    state = {"figures": [1, 2]}  # 2 个 figure
    case = _evaluate("t", state, {"min_overall": 6.5, "min_figures": 5})
    assert not case.passed


def test_evaluate_min_figures_pass():
    state = {"figures": [1, 2, 3], "evaluation": _evaluation()}
    case = _evaluate("t", state, {"min_overall": 6.5, "min_figures": 3})
    assert case.passed


def test_evaluate_min_derivation_steps_fail():
    model = ModelVersion(
        stage="final", description="d" * 200,
        derivation_steps=[DerivationStep(title="s", motivation="m",
                                         statement="st") for _ in range(3)])
    state = {"model_versions": [model]}
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "min_derivation_steps": 4})
    assert not case.passed
    assert any("derivation" in f.lower() for f in case.failures)


def test_evaluate_min_derivation_steps_pass():
    model = ModelVersion(
        stage="final", description="d" * 200,
        derivation_steps=[DerivationStep(title="s", motivation="m",
                                         statement="st") for _ in range(3)])
    state = {"model_versions": [model], "evaluation": _evaluation()}
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "min_derivation_steps": 2})
    assert case.passed


def test_evaluate_no_model_versions_skips_derivation_check():
    # 无模型版本 → 优雅跳过，不记失败
    state = {"model_versions": [], "evaluation": _evaluation()}
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "min_derivation_steps": 4})
    assert case.passed


def test_evaluate_max_paper_critic_issues_fail():
    critic = CriticReport(
        target="paper", score=9, approved=True,
        issues=[CriticIssue(problem="x") for _ in range(5)])
    state = {"critic_reports": [critic]}
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "max_paper_critic_issues": 3})
    assert not case.passed
    assert any("paper_critic" in f or "critic_issues" in f
               for f in case.failures)


def test_evaluate_max_paper_critic_issues_pass():
    critic = CriticReport(
        target="paper", score=9, approved=True,
        issues=[CriticIssue(problem="x") for _ in range(2)])
    state = {"critic_reports": [critic], "evaluation": _evaluation()}
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "max_paper_critic_issues": 3})
    assert case.passed


def test_evaluate_max_paper_critic_issues_no_paper_critic_skips():
    # 只有 modeler critic，无 paper critic → 优雅跳过
    critic = CriticReport(target="modeler", score=9, approved=True)
    state = {"critic_reports": [critic], "evaluation": _evaluation()}
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "max_paper_critic_issues": 0})
    assert case.passed


def test_evaluate_min_references_bracket_pattern_fail():
    paper = PaperSections(
        references="[1] Author. Title. Venue, 2020.\n"
                   "[2] Author2. Title2. Venue2, 2019.\n"
                   "[3] Author3. Title3. 2021.")
    state = {"paper": paper}
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "min_references": 5})
    assert not case.passed  # 仅 3 条


def test_evaluate_min_references_bracket_pattern_pass():
    paper = PaperSections(
        references="[1] Author. Title. Venue, 2020.\n"
                   "[2] Author2. Title2. Venue2, 2019.\n"
                   "[3] Author3. Title3. 2021.")
    state = {"paper": paper, "evaluation": _evaluation()}
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "min_references": 2})
    assert case.passed  # 3 >= 2


def test_evaluate_min_references_plain_lines():
    # 无 [N] 模式 → 按非空行计数
    paper = PaperSections(
        references="Author A. Title. 2020.\nAuthor B. Title2. 2019.")
    state = {"paper": paper, "evaluation": _evaluation()}
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "min_references": 2})
    assert case.passed  # 2 非空行


def test_evaluate_min_references_dash_placeholder():
    # mock paper references="-" → 1 非空行
    paper = PaperSections(references="-")
    state = {"paper": paper}
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "min_references": 2})
    assert not case.passed  # 1 < 2


def test_evaluate_skips_missing_optional_fields():
    """expect 不含任何新字段时 _evaluate 仍正常工作（向后兼容）。"""
    paper = PaperSections(abstract="x")
    state = {"evaluation": _evaluation(), "paper": paper}
    case = _evaluate("t", state, {"min_overall": 6.5})
    assert case.passed


def test_evaluate_min_paper_pdf_pages_missing_file_skips(tmp_path):
    # output_dir 无 paper.pdf → 跳过（不记失败）
    state = {"evaluation": _evaluation()}
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "min_paper_pdf_pages": 10},
                     output_dir=tmp_path)
    assert case.passed


def test_evaluate_min_paper_pdf_pages_real_pdf(tmp_path):
    # 用 pypdf 生成一个真实多页 PDF（依赖 reportlab 可用，否则跳过）。
    pdf_path = tmp_path / "paper.pdf"
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        # 没装 reportlab：用 pypdf 的底层对象造一个空多页 PDF。
        try:
            from pypdf import PdfWriter
            writer = PdfWriter()
            for _ in range(3):
                writer.add_blank_page(width=200, height=200)
            with open(pdf_path, "wb") as fh:
                writer.write(fh)
        except Exception:
            return  # 无法造 PDF → 跳过本测试
    else:
        c = canvas.Canvas(str(pdf_path))
        for _ in range(3):
            c.showPage()
        c.save()

    state = {"evaluation": _evaluation()}
    # 3 页，阈值 5 → 失败
    case_fail = _evaluate("t", state, {"min_overall": 6.5,
                                       "min_paper_pdf_pages": 5},
                          output_dir=tmp_path)
    assert not case_fail.passed
    assert any("paper_pdf" in f.lower() for f in case_fail.failures)
    # 3 页，阈值 2 → 通过
    case_pass = _evaluate("t", state, {"min_overall": 6.5,
                                       "min_paper_pdf_pages": 2},
                          output_dir=tmp_path)
    assert case_pass.passed


def test_evaluate_xelatex_zero_errors_missing_log_skips(tmp_path):
    state = {"evaluation": _evaluation()}
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "xelatex_must_have_zero_errors": True},
                     output_dir=tmp_path)
    assert case.passed  # 无 log → 跳过


def test_evaluate_xelatex_zero_errors_clean_log_pass(tmp_path):
    (tmp_path / "compile.log").write_text(
        "This is pdfTeX, Version 3.14\nOutput written on paper.pdf\n",
        encoding="utf-8")
    state = {"evaluation": _evaluation()}
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "xelatex_must_have_zero_errors": True},
                     output_dir=tmp_path)
    assert case.passed


def test_evaluate_xelatex_zero_errors_with_errors_fail(tmp_path):
    (tmp_path / "compile.log").write_text(
        "! Undefined control sequence.\nError: Emergency stop.\n",
        encoding="utf-8")
    state = {"evaluation": _evaluation()}
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "xelatex_must_have_zero_errors": True},
                     output_dir=tmp_path)
    assert not case.passed
    assert any("xelatex" in f.lower() for f in case.failures)


def test_evaluate_xelatex_finds_generic_log_file(tmp_path):
    # 无 compile.log/xelatex.log，但有任意 *.log
    (tmp_path / "run.log").write_text("! Some TeX error.\n", encoding="utf-8")
    state = {"evaluation": _evaluation()}
    case = _evaluate("t", state, {"min_overall": 6.5,
                                  "xelatex_must_have_zero_errors": True},
                     output_dir=tmp_path)
    assert not case.passed
