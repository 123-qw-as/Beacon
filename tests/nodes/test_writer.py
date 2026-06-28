from math_agent.state import MathModelingState, ModelVersion, CodeArtifact, PaperSections
from math_agent.nodes.writer import writer_node, render_markdown


def test_writer_fills_paper(mocker):
    fake = PaperSections(
        abstract="a"*200, problem_restatement="b"*200, assumptions="c"*200,
        notation="d"*200, model_section="e"*200, solution="f"*200,
        conclusion="g"*200, references="h",
    )
    mocker.patch("math_agent.nodes.writer.complete", return_value=fake)
    s = MathModelingState(problem="p")
    s.model_versions.append(ModelVersion(stage="final", description="d"))
    s.code_artifacts.append(CodeArtifact(purpose="x", code="c", success=True, stdout="42"))
    delta = writer_node(s)
    assert isinstance(delta["paper"], PaperSections)
    assert delta["paper"].abstract.startswith("a")


def test_render_markdown_contains_sections():
    s = MathModelingState(problem="P")
    s.paper = PaperSections(abstract="A", problem_restatement="B", assumptions="C",
                            notation="D", model_section="E", solution="F",
                            conclusion="H", references="I")
    s.code_artifacts.append(CodeArtifact(purpose="x", code="print(1)", success=True, stdout="1"))
    md = render_markdown(s)
    assert "## 摘要" in md and "## 6. 模型评价" in md
    assert "print(1)" in md
