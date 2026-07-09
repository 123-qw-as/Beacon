"""端到端 smoke：mock 掉所有 LLM 与 LaTeX 编译，验证：
1) 节点按顺序被命中；
2) interrupt_before=['human_review'] 时 graph 暂停；
3) 注入 human_decision 后恢复，能产出 paper.tex + paper.md。
"""
from pathlib import Path

from PIL import Image

from math_agent.graph import build_graph
from math_agent.state import (
    Assumption, ModelVersion, CriticReport, PaperSections,
    EvaluationReport, HumanDecision, DerivationStep, ProblemBlueprint,
    ModelCodeConsistencyReport,
)
from math_agent.nodes.analyst import AnalystOutput
from math_agent.nodes.coder import CoderDraft
from math_agent.nodes.sensitivity import SensitivityPlan, SensitivityCode, Interpretations
from math_agent.nodes.figure_pipeline import FigureCriticOut, FigureAnalysisOut
from math_agent.prompts.modeler_derivation import ConsistencyCheck


def _setup_all_mocks(mocker, workdir):
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
        # final 阶段会额外调用 derivation steps + consistency gate，
        # 需按请求的 schema 返回正确类型，否则解析出错。
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

    # coder 仅输出 print（生成 png 由 sensitivity step 完成）
    mocker.patch("math_agent.nodes.coder.complete",
                 return_value=CoderDraft(purpose="主结果", code="print('coder done')"))

    # model_code_consistency 审查通过
    mocker.patch("math_agent.nodes.model_code_consistency.complete",
                 return_value=ModelCodeConsistencyReport(score=9, approved=True))

    # sensitivity 三段
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

    # figure pipeline：sensitivity 图 = 1 张
    fc = FigureCriticOut(score=9, approved=True)
    fa = FigureAnalysisOut(analysis="趋势单调，敏感度中等。")
    mocker.patch("math_agent.nodes.figure_pipeline.complete",
                 side_effect=[fc, fa])

    # writer prep：大纲（WriterOutline schema）
    # writer section：7 节，每节返回对应精简 schema。
    # ponytail: 一个 side_effect 处理两种 schema，按 schema 类型分发。
    from math_agent.prompts.writer_section import (
        WriterOutline, _AbstractProblemOut, _AssumptionsNotationOut,
        _ModelOut, _SolutionOut, _SensitivityOut, _ConclusionOut, _ReferencesOut,
    )
    _section_payloads = {
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
    def _writer_complete(prompt, *, schema, **kw):
        return _section_payloads.get(schema, _section_payloads[WriterOutline])
    mocker.patch("math_agent.nodes.writer.complete", side_effect=_writer_complete)
    mocker.patch("math_agent.nodes.paper_critic.complete",
                 return_value=CriticReport(target="paper", score=9, approved=True))
    mocker.patch("math_agent.nodes.evaluation.complete",
                 return_value=EvaluationReport(
                     assumption_reasonableness=8, modeling_creativity=8,
                     result_correctness=8, writing_clarity=8, extra_depth=8, overall=8.0,
                 ))

    # latex compile —— 不依赖系统 xelatex
    mocker.patch("math_agent.nodes.latex.compile_latex",
                 return_value=type("R",(object,),{"success": True, "pdf_path":"", "log":""})())


def test_full_pipeline_with_hitl_interrupt_and_resume(mocker, workdir):
    from langgraph.checkpoint.memory import MemorySaver  # 测试用内存 checkpointer
    _setup_all_mocks(mocker, workdir)

    saver = MemorySaver()
    g = build_graph(checkpointer=saver, interrupt_before=["human_review"])

    config = {"configurable": {"thread_id": "t1"}}
    initial = {
        "problem": "p", "stage_target": "basic", "iteration": 0,
        "output_dir": str(workdir),
    }
    # 第一次 invoke：流到 human_review 之前应当被中断
    g.invoke(initial, config=config)
    snap = saver.get(config)
    assert snap is not None
    # 此时论文 latex 还没生成
    assert not (workdir / "paper.tex").exists()

    # 注入决定，再恢复
    g.update_state(config, {"human_decision": HumanDecision(approved=True, notes="ok")})
    g.invoke(None, config=config)

    assert (workdir / "paper.tex").exists()
    assert (workdir / "paper.md").exists()
