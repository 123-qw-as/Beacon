from math_agent.graph import build_graph
from math_agent.state import Assumption, ModelVersion, CriticReport, PaperSections
from math_agent.nodes.analyst import AnalystOutput
from math_agent.nodes.coder import CoderDraft


def _mock_coder(mocker):
    mocker.patch(
        "math_agent.nodes.coder.complete",
        return_value=CoderDraft(purpose="ok", code="print('done')"),
    )


def _mock_writer(mocker):
    mocker.patch(
        "math_agent.nodes.writer.complete",
        return_value=PaperSections(
            abstract="x"*200, problem_restatement="x"*200, assumptions="x"*200,
            notation="x"*200, model_section="x"*200, solution="x"*200,
            conclusion="x"*200, references="-",
        ),
    )


def test_graph_runs_full_modeling_loop(mocker, workdir):
    mocker.patch(
        "math_agent.nodes.analyst.complete",
        return_value=AnalystOutput(
            assumptions=[Assumption(statement="A", rationale="r")]
        ),
    )

    stage_iter = iter(["basic", "improved", "final"])

    def fake_modeler(prompt, **kw):
        return ModelVersion(stage=next(stage_iter), description="d"*200, equations=["x=1"])

    def fake_critic(prompt, **kw):
        return CriticReport(target="modeler", score=9, approved=True)

    mocker.patch("math_agent.nodes.modeler.complete", side_effect=fake_modeler)
    mocker.patch("math_agent.nodes.model_critic.complete", side_effect=fake_critic)
    _mock_coder(mocker)
    _mock_writer(mocker)

    g = build_graph()
    final = g.invoke({"problem": "p", "stage_target": "basic", "iteration": 0, "output_dir": str(workdir)})
    stages = [m.stage for m in final["model_versions"]]
    assert stages == ["basic", "improved", "final"]


def test_graph_retries_modeler_on_low_score(mocker, workdir):
    """basic 阶段前两轮 critic 不通过、第三轮通过：modeler 应在 basic 阶段被调 3 次；
    随后 improved/final 各一次性通过，覆盖 routing 的 `retry` 分支与 `latest_critic_for_stage` 过滤。"""
    mocker.patch(
        "math_agent.nodes.analyst.complete",
        return_value=AnalystOutput(assumptions=[Assumption(statement="A", rationale="r")]),
    )

    stage_iter = iter(["basic", "basic", "basic", "improved", "final"])
    mocker.patch(
        "math_agent.nodes.modeler.complete",
        side_effect=lambda *a, **k: ModelVersion(stage=next(stage_iter), description="d"*200),
    )

    critic_iter = iter([
        CriticReport(target="modeler", score=4, approved=False),
        CriticReport(target="modeler", score=5, approved=False),
        CriticReport(target="modeler", score=9, approved=True),
        CriticReport(target="modeler", score=9, approved=True),
        CriticReport(target="modeler", score=9, approved=True),
    ])
    mocker.patch(
        "math_agent.nodes.model_critic.complete",
        side_effect=lambda *a, **k: next(critic_iter),
    )
    _mock_coder(mocker)
    _mock_writer(mocker)

    g = build_graph()
    final = g.invoke({"problem": "p", "stage_target": "basic", "iteration": 0, "output_dir": str(workdir)})

    basic_versions = [m for m in final["model_versions"] if m.stage == "basic"]
    basic_critics = [c for c in final["critic_reports"] if c.stage == "basic"]
    assert len(basic_versions) == 3
    assert len(basic_critics) == 3
    assert basic_critics[-1].approved is True
    assert any(m.stage == "final" for m in final["model_versions"])


def test_graph_includes_coder(mocker, workdir):
    mocker.patch(
        "math_agent.nodes.analyst.complete",
        return_value=AnalystOutput(assumptions=[Assumption(statement="A", rationale="r")]),
    )
    stage_iter = iter(["basic", "improved", "final"])
    mocker.patch(
        "math_agent.nodes.modeler.complete",
        side_effect=lambda *a, **k: ModelVersion(stage=next(stage_iter), description="d"*200),
    )
    mocker.patch(
        "math_agent.nodes.model_critic.complete",
        return_value=CriticReport(target="modeler", score=9, approved=True),
    )
    _mock_coder(mocker)
    _mock_writer(mocker)

    g = build_graph()
    final = g.invoke({
        "problem": "p", "stage_target": "basic", "iteration": 0,
        "output_dir": str(workdir),
    })
    assert final["code_artifacts"][-1].success


def test_graph_writes_paper_md(mocker, workdir):
    mocker.patch(
        "math_agent.nodes.analyst.complete",
        return_value=AnalystOutput(assumptions=[Assumption(statement="A", rationale="r")]),
    )
    stage_iter = iter(["basic", "improved", "final"])
    mocker.patch(
        "math_agent.nodes.modeler.complete",
        side_effect=lambda *a, **k: ModelVersion(stage=next(stage_iter), description="d"*200),
    )
    mocker.patch(
        "math_agent.nodes.model_critic.complete",
        return_value=CriticReport(target="modeler", score=9, approved=True),
    )
    _mock_coder(mocker)
    _mock_writer(mocker)
    g = build_graph()
    g.invoke({
        "problem": "single bike", "stage_target": "basic", "iteration": 0,
        "output_dir": str(workdir),
    })
    assert (workdir / "paper.md").exists()
    assert "## 摘要" in (workdir / "paper.md").read_text(encoding="utf-8")
