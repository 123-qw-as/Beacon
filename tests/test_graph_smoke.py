from PIL import Image
from pathlib import Path

from math_agent.graph import build_graph
from math_agent.state import (
    Assumption, ModelVersion, CriticReport, CriticIssue, PaperSections,
    EvaluationReport, HumanDecision, DerivationStep,
)
from math_agent.nodes.analyst import AnalystOutput
from math_agent.nodes.coder import CoderDraft
from math_agent.nodes.sensitivity import SensitivityPlan, SensitivityCode, Interpretations
from math_agent.nodes.figure_pipeline import FigureCriticOut, FigureAnalysisOut
from math_agent.prompts.modeler_derivation import ConsistencyCheck


def _png(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 240), "white").save(p)
    return str(p)


def _full_mocks(mocker, workdir, *, stages=("basic", "improved", "final"), critics=None):
    """给所有 LLM 节点装上桩，保证 graph 端到端能跑通。"""
    mocker.patch("math_agent.nodes.analyst.complete",
                 return_value=AnalystOutput(assumptions=[
                     Assumption(statement="A", rationale="r", sensitivity_relevant=True)]))

    stage_iter = iter(stages)

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

    crit_iter = iter(critics) if critics else None
    if crit_iter is not None:
        mocker.patch("math_agent.nodes.model_critic.complete",
                     side_effect=lambda *a, **k: next(crit_iter))
    else:
        mocker.patch("math_agent.nodes.model_critic.complete",
                     return_value=CriticReport(target="modeler", score=9, approved=True))

    mocker.patch("math_agent.nodes.coder.complete",
                 return_value=CoderDraft(purpose="ok", code="print('done')"))

    sens_plan = SensitivityPlan(runs=[{"parameter": "lambda", "values": [1, 2, 3, 4, 5],
                                       "metric": "y", "rationale": "r"}])
    sens_code = SensitivityCode(code=(
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "v=[1,2,3,4,5]; r=[x*2 for x in v]\n"
        "plt.plot(v,r); plt.savefig('lambda.png')\n"
        "print(f'RESULT: parameter=lambda values={v} results={r}')\n"
    ))
    mocker.patch("math_agent.nodes.sensitivity.complete",
                 side_effect=[sens_plan, sens_code,
                              Interpretations(interpretations=["lambda 越大 y 线性增长。"])])

    mocker.patch("math_agent.nodes.figure_pipeline.complete",
                 side_effect=[FigureCriticOut(score=9, approved=True),
                              FigureAnalysisOut(analysis="趋势单调，敏感度中等。")])

    mocker.patch("math_agent.nodes.writer.complete",
                 return_value=PaperSections(
                     abstract="x"*200, problem_restatement="x"*200, assumptions="x"*200,
                     notation="x"*200, model_section="x"*200, solution="x"*200,
                     sensitivity="x"*200, conclusion="x"*200, references="-",
                 ))
    mocker.patch("math_agent.nodes.paper_critic.complete",
                 return_value=CriticReport(target="paper", score=9, approved=True))
    mocker.patch("math_agent.nodes.evaluation.complete",
                 return_value=EvaluationReport(
                     assumption_reasonableness=8, modeling_creativity=8,
                     result_correctness=8, writing_clarity=8, extra_depth=8, overall=8.0,
                 ))
    mocker.patch("math_agent.nodes.latex.compile_latex",
                 return_value=type("R", (), {"success": True, "pdf_path": "", "log": ""})())


def test_graph_runs_full_modeling_loop(mocker, workdir):
    _full_mocks(mocker, workdir)
    g = build_graph()
    final = g.invoke({
        "problem": "p", "stage_target": "basic", "iteration": 0,
        "output_dir": str(workdir),
        "human_decision": HumanDecision(approved=True),
    })
    stages = [m.stage for m in final["model_versions"]]
    assert stages == ["basic", "improved", "final"]


def test_graph_retries_modeler_on_low_score(mocker, workdir):
    """basic 阶段前两轮 critic 不通过、第三轮通过：modeler 应在 basic 阶段被调 3 次。"""
    critics = [
        CriticReport(target="modeler", score=4, approved=False),
        CriticReport(target="modeler", score=5, approved=False),
        CriticReport(target="modeler", score=9, approved=True),
        CriticReport(target="modeler", score=9, approved=True),
        CriticReport(target="modeler", score=9, approved=True),
    ]
    _full_mocks(mocker, workdir,
                stages=("basic", "basic", "basic", "improved", "final"),
                critics=critics)
    g = build_graph()
    final = g.invoke({
        "problem": "p", "stage_target": "basic", "iteration": 0,
        "output_dir": str(workdir),
        "human_decision": HumanDecision(approved=True),
    })
    basic_versions = [m for m in final["model_versions"] if m.stage == "basic"]
    basic_critics = [c for c in final["critic_reports"] if c.stage == "basic"]
    assert len(basic_versions) == 3
    assert len(basic_critics) == 3
    assert basic_critics[-1].approved is True
    assert any(m.stage == "final" for m in final["model_versions"])


def test_graph_writes_paper_md(mocker, workdir):
    _full_mocks(mocker, workdir)
    g = build_graph()
    g.invoke({
        "problem": "single bike", "stage_target": "basic", "iteration": 0,
        "output_dir": str(workdir),
        "human_decision": HumanDecision(approved=True),
    })
    assert (workdir / "paper.md").exists()
    assert (workdir / "paper.tex").exists()
    md = (workdir / "paper.md").read_text(encoding="utf-8")
    assert "## 摘要" in md
    assert "## 6. 敏感性分析" in md


def test_writer_paper_critic_loop_isolated(mocker):
    """隔离测试 writer↔paper_critic 闭环。第一次 critic 拒，第二次通过 → writer 调 2 次。

    Plan D Phase 2：writer_node 改为大纲(1)+分章(7) 多调用。
    首轮 8 次，重试轮（section=general → 全部分组）7 次，共 15 次 complete。
    用 side_effect 按调用顺序返回：1 outline + 7 v1 分章 + 7 v2 分章。
    """
    from langgraph.graph import StateGraph, END
    from math_agent.state import MathModelingState as _S
    from math_agent.nodes.writer import writer_node
    from math_agent.nodes.paper_critic import paper_critic_node
    from math_agent.routing import after_paper_critic
    from math_agent.prompts.writer_section import (
        WriterOutline, writer_sections, schema_for_group,
    )

    def _paper(mk):
        return PaperSections(abstract=mk * 100, problem_restatement="x" * 150,
                             assumptions="x" * 150, notation="x" * 150,
                             model_section="x" * 400, solution="x" * 200,
                             sensitivity="x" * 150, conclusion="x" * 150, references="-")

    outline = WriterOutline(abstract="thesis")
    # 首轮：1 大纲 + 7 分章（v1）
    # 重试轮：7 分章（v2）
    seq = [outline] + [_paper("v1")] * 7 + [_paper("v2")] * 7
    mocker.patch("math_agent.nodes.writer.complete", side_effect=seq)
    mocker.patch("math_agent.nodes.paper_critic.complete", side_effect=[
        CriticReport(target="paper", score=4, approved=False,
                     issues=[CriticIssue(problem="编数字")], suggestions=["改定性"]),
        CriticReport(target="paper", score=9, approved=True),
    ])

    g = StateGraph(_S)
    g.add_node("writer", writer_node)
    g.add_node("paper_critic", paper_critic_node)
    g.set_entry_point("writer")
    g.add_edge("writer", "paper_critic")
    g.add_conditional_edges("paper_critic", after_paper_critic,
                            {"retry": "writer", "advance": END})
    compiled = g.compile()

    final = compiled.invoke({"problem": "p"})
    assert final["writer_iteration"] == 2
    assert final["paper"].abstract.startswith("v2")
    paper_critics = [r for r in final["critic_reports"] if r.target == "paper"]
    assert len(paper_critics) == 2
    assert paper_critics[-1].approved is True
