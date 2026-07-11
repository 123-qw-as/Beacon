"""bench mock harness：把 LLM / LaTeX 全 mock，runner 跑结构性流程。

放在 tests/ 下，避免 src/math_agent/bench/runner.py import unittest.mock。
所有节点 mock 用 itertools.cycle，足以撑住 bench 多道题的循环调用。
"""
from __future__ import annotations

import itertools
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

from math_agent.state import (
    Assumption, ModelVersion, CriticReport, PaperSections,
    EvaluationReport, DerivationStep, ProblemBlueprint, ModelCodeConsistencyReport,
)
from math_agent.nodes.analyst import AnalystOutput
from math_agent.nodes.coder import CoderDraft
from math_agent.nodes.figure_pipeline import FigureCriticOut, FigureAnalysisOut
from math_agent.nodes.sensitivity import SensitivityPlan, SensitivityCode, Interpretations
from math_agent.prompts.modeler_derivation import ConsistencyCheck


# 两道 bench 题的关键词全集；默认 paper 含全部，保证 PASS。
_ALL_KEYWORDS = ["覆盖", "无人机", "鲁棒", "内涝", "排水", "风险"]


def _make_paper(keywords: list[str]) -> PaperSections:
    """生成必含给定 keywords 的 paper（每段都含一遍）。"""
    kw_line = "、".join(keywords) if keywords else "（无）"
    body = f"本研究围绕 {kw_line} 展开，模型经过 basic->improved->final 演化。" * 5
    return PaperSections(
        abstract=body, problem_restatement=body, assumptions=body,
        notation=body, model_section=body, solution=body,
        sensitivity=body, conclusion=body, references="-",
    )


def _setup_mocks(stack: ExitStack, *, paper: PaperSections,
                 evaluation: EvaluationReport):
    def _patch(target, **kw):
        stack.enter_context(patch(target, **kw))

    _patch("math_agent.nodes.analyst.complete",
           side_effect=itertools.cycle([ProblemBlueprint(
               core_task="bench task",
               assumptions=[
                   Assumption(statement="A", rationale="r", sensitivity_relevant=True)],
               problem_domains=["optimization"],
           )]))

    # blueprint_critic 审查通过
    _patch("math_agent.nodes.blueprint_critic.complete",
           side_effect=itertools.cycle([CriticReport(target="analyst", score=9, approved=True, critic_type="blueprint")]))

    def _modeler_complete(prompt, *, schema, **kw):
        # final 阶段会额外调用 derivation steps + consistency gate，
        # 需按请求的 schema 返回正确类型，否则解析出错。
        if schema is ModelVersion:
            return ModelVersion(stage="basic", description="d" * 200)
        if schema is DerivationStep:
            return DerivationStep(title="step", motivation="m", statement="s", result="r")
        if schema is ConsistencyCheck:
            return ConsistencyCheck(coherent=True, issues=[])
        return ModelVersion(stage="basic", description="d" * 200)

    _patch("math_agent.nodes.modeler.complete", side_effect=_modeler_complete)

    _patch("math_agent.nodes.model_critic.complete",
           side_effect=itertools.cycle([CriticReport(target="modeler", score=9, approved=True)]))

    _patch("math_agent.nodes.coder.complete",
           side_effect=itertools.cycle([CoderDraft(purpose="主结果", code="print('done')")]))

    # model_code_consistency 审查通过
    _patch("math_agent.nodes.model_code_consistency.complete",
           side_effect=itertools.cycle([ModelCodeConsistencyReport(score=9, approved=True)]))

    sens_plan = SensitivityPlan(runs=[{"parameter": "lambda", "values": [1, 2, 3, 4, 5],
                                       "metric": "y", "rationale": "r"}])
    sens_code = SensitivityCode(code=(
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "v=[1,2,3,4,5]; r=[x*2 for x in v]\n"
        "plt.plot(v,r); plt.savefig('lambda.png')\n"
        "print(f'RESULT: parameter=lambda values={v} results={r}')\n"
    ))
    sens_interp = Interpretations(interpretations=["lambda 越大 y 线性增长。"])
    _patch("math_agent.nodes.sensitivity.complete",
           side_effect=itertools.cycle([sens_plan, sens_code, sens_interp]))

    _patch("math_agent.nodes.figure_pipeline.complete",
           side_effect=itertools.cycle([
               FigureCriticOut(score=9, approved=True),
               FigureAnalysisOut(analysis="趋势单调。"),
           ]))

    # writer: 按 schema 分派（WriterOutline vs 各 section schema）
    from math_agent.prompts.writer_section import (
        WriterOutline, _AbstractProblemOut, _AssumptionsNotationOut,
        _ModelOut, _SolutionOut, _SensitivityOut, _ConclusionOut, _ReferencesOut,
    )
    _section_payloads = {
        WriterOutline: WriterOutline(),
        _AbstractProblemOut: _AbstractProblemOut(
            abstract=paper.abstract, problem_restatement=paper.problem_restatement, keywords=""),
        _AssumptionsNotationOut: _AssumptionsNotationOut(assumptions=paper.assumptions, notation=paper.notation),
        _ModelOut: _ModelOut(model_section=paper.model_section),
        _SolutionOut: _SolutionOut(solution=paper.solution),
        _SensitivityOut: _SensitivityOut(sensitivity=paper.sensitivity),
        _ConclusionOut: _ConclusionOut(conclusion=paper.conclusion),
        _ReferencesOut: _ReferencesOut(references=paper.references),
    }
    def _writer_complete(prompt, *, schema, **kw):
        return _section_payloads.get(schema, paper)
    _patch("math_agent.nodes.writer.complete", side_effect=_writer_complete)
    _patch("math_agent.nodes.paper_critic.complete",
           side_effect=itertools.cycle([CriticReport(target="paper", score=9, approved=True)]))
    _patch("math_agent.nodes.evaluation.complete",
           side_effect=itertools.cycle([evaluation]))

    _patch("math_agent.nodes.latex_node.compile_latex",
           return_value=type("R", (object,), {
               "success": True, "pdf_path": "", "log": "", "error_kind": "",
           })())


def _good_evaluation() -> EvaluationReport:
    return EvaluationReport(
        assumption_reasonableness=8, modeling_creativity=8,
        result_correctness=8, writing_clarity=8, extra_depth=8, overall=8.0,
    )


def _bad_evaluation() -> EvaluationReport:
    return EvaluationReport(
        assumption_reasonableness=2, modeling_creativity=3,
        result_correctness=3, writing_clarity=3, extra_depth=2, overall=3.0,
    )


@pytest.fixture
def install_bench_mocks():
    """默认 fixture：paper 必含所有 expectations keywords，evaluation 高分 → 全 PASS。"""
    with ExitStack() as stack:
        _setup_mocks(stack, paper=_make_paper(_ALL_KEYWORDS),
                     evaluation=_good_evaluation())
        yield


@pytest.fixture
def install_bench_mocks_missing_keyword():
    """故意 paper 不含 keywords，验证 runner 能识别 FAIL。"""
    with ExitStack() as stack:
        _setup_mocks(stack, paper=_make_paper(["（缺失关键词的 paper）"]),
                     evaluation=_good_evaluation())
        yield


@pytest.fixture
def install_bench_mocks_low_overall():
    """让 evaluation overall=3.0，验证 runner 能识别 overall < min_overall 的失败路径。"""
    with ExitStack() as stack:
        _setup_mocks(stack, paper=_make_paper(_ALL_KEYWORDS),
                     evaluation=_bad_evaluation())
        yield
