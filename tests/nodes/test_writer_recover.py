"""writer 子流程可恢复性测试。

验证：section 崩溃后，recover 从队列断点续跑，已完成节不重调 LLM。
"""
from langgraph.checkpoint.memory import MemorySaver

from math_agent.graph import build_graph
from math_agent.prompts.writer_section import (
    WriterOutline, _AbstractProblemOut, _AssumptionsNotationOut,
    _ModelOut, _SolutionOut, _SensitivityOut, _ConclusionOut, _ReferencesOut,
)
from math_agent.state import (
    Assumption, ModelVersion, CodeArtifact, CriticReport,
    EvaluationReport, FigureArtifact, HumanDecision, SensitivityRun,
    ProblemBlueprint, ModelCodeConsistencyReport,
)
from math_agent.nodes.analyst import AnalystOutput
from math_agent.nodes.coder import CoderDraft
from math_agent.nodes.sensitivity import SensitivityPlan, SensitivityCode, Interpretations
from math_agent.nodes.figure_pipeline import FigureCriticOut, FigureAnalysisOut
from math_agent.prompts.modeler_derivation import ConsistencyCheck
from math_agent.state import DerivationStep


_PAYLOADS = {
    WriterOutline: WriterOutline(),
    _AbstractProblemOut: _AbstractProblemOut(
        abstract="a"*200, problem_restatement="x"*200, keywords="k"),
    _AssumptionsNotationOut: _AssumptionsNotationOut(assumptions="x"*200, notation="x"*200),
    _ModelOut: _ModelOut(model_section="x"*200),
    _SolutionOut: _SolutionOut(solution="x"*200),
    _SensitivityOut: _SensitivityOut(sensitivity="x"*200),
    _ConclusionOut: _ConclusionOut(conclusion="x"*200),
    _ReferencesOut: _ReferencesOut(references="-"),
}


def _setup_upstream_mocks(mocker, workdir):
    """mock writer 之前的所有节点，让流程能跑到 writer。"""
    mocker.patch("math_agent.nodes.analyst.complete",
                 return_value=ProblemBlueprint(
                     core_task="test task",
                     assumptions=[
                         Assumption(statement="A", rationale="r", sensitivity_relevant=True)],
                     problem_domains=["optimization"],
                 ))
    # blueprint_critic 审查通过
    mocker.patch("math_agent.nodes.blueprint_critic.complete",
                 return_value=CriticReport(target="analyst", score=9, approved=True))
    stage_iter = iter(["basic", "improved", "final"])

    def _modeler_complete(prompt, *, schema, **kw):
        if schema is ModelVersion:
            return ModelVersion(stage=next(stage_iter), description="d" * 200)
        if schema is DerivationStep:
            return DerivationStep(title="step", motivation="m", statement="s", result="r")
        if schema is ConsistencyCheck:
            return ConsistencyCheck(coherent=True, issues=[])
        return ModelVersion(stage="basic", description="d" * 200)

    mocker.patch("math_agent.nodes.modeler.complete", side_effect=_modeler_complete)
    mocker.patch("math_agent.nodes.model_critic.complete",
                 return_value=CriticReport(target="modeler", score=9, approved=True))
    mocker.patch("math_agent.nodes.coder.complete",
                 return_value=CoderDraft(purpose="主结果", code="print('coder done')"))
    # model_code_consistency 审查通过
    mocker.patch("math_agent.nodes.model_code_consistency.complete",
                 return_value=ModelCodeConsistencyReport(score=9, approved=True))
    sens_plan = SensitivityPlan(runs=[{"parameter": "lambda", "values": [1,2,3,4,5], "metric": "y", "rationale": "r"}])
    sens_code = SensitivityCode(code=(
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "v=[1,2,3,4,5]; r=[x*2 for x in v]\n"
        "plt.plot(v,r); plt.savefig('lambda.png')\n"
        "print(f'RESULT: parameter=lambda values={v} results={r}')\n"
    ))
    sens_interp = Interpretations(interpretations=["lambda 越大 y 线性增长，敏感度中等。"])
    mocker.patch("math_agent.nodes.sensitivity.complete",
                 side_effect=[sens_plan, sens_code, sens_interp])
    fc = FigureCriticOut(score=9, approved=True)
    fa = FigureAnalysisOut(analysis="趋势单调，敏感度中等。")
    mocker.patch("math_agent.nodes.figure_pipeline.complete",
                 side_effect=[fc, fa])
    # writer 下游
    mocker.patch("math_agent.nodes.paper_critic.complete",
                 return_value=CriticReport(target="paper", score=9, approved=True))
    mocker.patch("math_agent.nodes.evaluation.complete",
                 return_value=EvaluationReport(
                     assumption_reasonableness=8, modeling_creativity=8,
                     result_correctness=8, writing_clarity=8, extra_depth=8, overall=8.0,
                 ))
    mocker.patch("math_agent.nodes.latex.compile_latex",
                 return_value=type("R",(object,),{"success": True, "pdf_path":"", "log":""})())


def test_writer_section_crash_then_recover(mocker, workdir):
    """section 3 崩溃后 recover，section 1-2 不重调 LLM，3-7 续跑完成。"""
    _setup_upstream_mocks(mocker, workdir)

    call_log: list[str] = []

    def _writer_complete(prompt, *, schema, **kw):
        # 用 schema 标识当前节；WriterOutline=prep，其余=section。
        # ponytail: 用 schema 类型名记录调用顺序，足够测试断言。
        name = schema.__name__ if hasattr(schema, "__name__") else type(schema).__name__
        call_log.append(name)
        if schema is _SolutionOut and len([c for c in call_log if c == "_SolutionOut"]) == 1:
            # 第 3 节（solution）首次调用时崩溃
            from math_agent.errors import LLMTransportError
            raise LLMTransportError("模拟 router 半挂")
        return _PAYLOADS[schema]

    mocker.patch("math_agent.nodes.writer.complete", side_effect=_writer_complete)

    saver = MemorySaver()
    g = build_graph(checkpointer=saver, interrupt_before=["human_review"])
    config = {"configurable": {"thread_id": "t1"}}
    initial = {
        "problem": "p", "stage_target": "basic", "iteration": 0,
        "output_dir": str(workdir),
    }

    # 第一次 invoke：跑到 writer section 3（solution）崩溃
    from math_agent.errors import LLMTransportError
    try:
        g.invoke(initial, config=config)
    except LLMTransportError:
        pass  # 预期崩溃

    # section 1-2（abstract_problem, assumptions_notation）已完成，3 崩溃
    assert "_AbstractProblemOut" in call_log
    assert "_AssumptionsNotationOut" in call_log

    # recover：从 checkpoint 续跑（与 resume 同结构，但不注入 human_decision）
    g.invoke(None, config=config)

    # 续跑后 section 3-7 都跑完
    # section 1-2 不应重复调用（_AbstractProblemOut 只出现 1 次）
    assert call_log.count("_AbstractProblemOut") == 1
    assert call_log.count("_AssumptionsNotationOut") == 1
    # section 3（_SolutionOut）在 recover 后重试成功
    assert call_log.count("_SolutionOut") == 2  # 1 次崩溃 + 1 次成功

    # 最终 paper 写完：通过 graph StateSnapshot 读取
    state = g.get_state(config)
    assert state is not None and state.values is not None
    paper = state.values.get("paper")
    assert paper is not None
    assert paper.abstract.startswith("a")
    assert paper.references == "-"
