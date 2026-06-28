from math_agent.state import MathModelingState, ModelVersion
from math_agent.nodes.coder import coder_node, CoderDraft


def test_coder_runs_code_and_records_artifact(mocker, workdir):
    mocker.patch(
        "math_agent.nodes.coder.complete",
        return_value=CoderDraft(purpose="solve", code="print('hello')"),
    )
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.model_versions.append(ModelVersion(stage="final", description="d", equations=["x=1"]))
    delta = coder_node(s)
    assert delta["code_artifacts"][0].success
    assert "hello" in delta["code_artifacts"][0].stdout


def test_coder_retries_once_on_failure(mocker, workdir):
    drafts = [
        CoderDraft(purpose="solve", code="raise RuntimeError('x')"),
        CoderDraft(purpose="solve", code="print('ok')"),
    ]
    mocker.patch("math_agent.nodes.coder.complete", side_effect=drafts)
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.model_versions.append(ModelVersion(stage="final", description="d"))
    delta = coder_node(s)
    # 应当保留两个 artifact：第一次失败、第二次成功
    arts = delta["code_artifacts"]
    assert len(arts) == 2
    assert arts[0].success is False
    assert arts[1].success is True
