from PIL import Image
from pathlib import Path

from math_agent.graph import build_graph
from math_agent.state import (
    Assumption, ModelVersion, CriticReport, PaperSections,
    SensitivityRun, FigureArtifact, EvaluationReport, HumanDecision,
)
from math_agent.nodes.analyst import AnalystOutput
from math_agent.nodes.coder import CoderDraft
from math_agent.nodes.sensitivity import SensitivityPlan, SensitivityCode, Interpretations
from math_agent.nodes.figure_pipeline import FigureCriticOut, FigureAnalysisOut


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
    mocker.patch("math_agent.nodes.modeler.complete",
                 side_effect=lambda *a, **k: ModelVersion(stage=next(stage_iter), description="d"*200))

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
