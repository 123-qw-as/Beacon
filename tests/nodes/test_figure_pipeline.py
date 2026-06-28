from pathlib import Path
from PIL import Image

from math_agent.state import (
    MathModelingState, CodeArtifact, SensitivityRun,
)
from math_agent.nodes.figure_pipeline import (
    figure_pipeline_node, FigureCriticOut, FigureAnalysisOut,
)


def _png(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), "white").save(p, dpi=(150, 150))
    return str(p)


def test_pipeline_collects_pngs_from_code_artifacts_and_sensitivity(mocker, workdir):
    p1 = _png(workdir / "code" / "fig_a.png")
    p2 = _png(workdir / "sensitivity" / "lambda.png")

    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.code_artifacts.append(CodeArtifact(
        purpose="主结果", code="...", success=True,
        artifact_paths=[p1, "ignore.txt"],
    ))
    s.sensitivity_runs.append(SensitivityRun(
        parameter="lambda", values=[1, 2], metric="y", results=[1, 2],
        figure_path=p2,
    ))

    critic = FigureCriticOut(score=9, issues=[], suggestions=[], approved=True)
    analysis = FigureAnalysisOut(analysis="图显示 lambda 越大 y 越大，敏感度高。")
    mocker.patch("math_agent.nodes.figure_pipeline.complete",
                 side_effect=[critic, analysis, critic, analysis])

    delta = figure_pipeline_node(s)
    assert len(delta["figures"]) == 2
    paths = {f.path for f in delta["figures"]}
    assert paths == {p1, p2}
    assert all(f.quality_score == 9 for f in delta["figures"])
    assert all("lambda" in f.analysis or "敏感度" in f.analysis for f in delta["figures"])


def test_pipeline_skips_non_png_artifacts(mocker, workdir):
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.code_artifacts.append(CodeArtifact(
        purpose="x", code="...", success=True, artifact_paths=["a.csv", "b.txt"],
    ))
    mocker.patch("math_agent.nodes.figure_pipeline.complete")
    delta = figure_pipeline_node(s)
    assert delta.get("figures", []) == []


def test_pipeline_records_issue_for_low_quality_after_retry(mocker, workdir):
    p1 = _png(workdir / "code" / "x.png")
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.code_artifacts.append(CodeArtifact(
        purpose="x", code="...", success=True, artifact_paths=[p1],
    ))
    bad = FigureCriticOut(score=4, issues=["缺图例"], suggestions=["加图例"], approved=False)
    analysis = FigureAnalysisOut(analysis="尽管质量一般，趋势仍可读出。")
    mocker.patch("math_agent.nodes.figure_pipeline.complete",
                 side_effect=[bad, bad, analysis])
    delta = figure_pipeline_node(s)
    fig = delta["figures"][0]
    assert fig.quality_score == 4
    assert "缺图例" in fig.quality_issues
