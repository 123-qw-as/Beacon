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
    # 应当保留两个 figure artifact：第一次失败、第二次成功
    # （Phase 2 起主方案成功后还会追加 baseline 对照方案，按 category 过滤）
    arts = [a for a in delta["code_artifacts"] if a.category == "figure"]
    assert len(arts) == 2
    assert arts[0].success is False
    assert arts[1].success is True


def test_coder_records_error_when_all_retries_fail(mocker, workdir):
    """所有尝试都失败时，应在 state.errors 中显式记录。

    用 itertools.cycle 而非固定列表：若未来 MAX_CODE_RETRIES 上调，
    测试不会因 mock 耗尽而 StopIteration 掩盖真正问题。
    """
    from itertools import cycle
    mocker.patch(
        "math_agent.nodes.coder.complete",
        side_effect=cycle([CoderDraft(purpose="solve", code="raise RuntimeError('boom')")]),
    )
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.model_versions.append(ModelVersion(stage="final", description="d"))

    delta = coder_node(s)

    assert len(delta["code_artifacts"]) >= 1
    assert all(a.success is False for a in delta["code_artifacts"])
    assert "errors" in delta and delta["errors"]
    assert delta["errors"][0].startswith("coder:")


def test_coder_prompt_on_timeout_asks_to_shrink_scale(mocker, workdir):
    """attempt_0 超时时，attempt_1 的 prompt 应命中"缩小规模"提示而不是喂 stderr 修 bug。"""
    from math_agent.tools.runner import RunResult
    from unittest.mock import call

    drafts = [
        CoderDraft(purpose="s", code="import time; time.sleep(999)"),
        CoderDraft(purpose="s", code="print('ok')"),
    ]
    spy_complete = mocker.patch("math_agent.nodes.coder.complete", side_effect=drafts)
    # 第一次跑成 timeout，第二次跑成成功——不让真 subprocess 跑
    mocker.patch(
        "math_agent.nodes.coder.run_python",
        side_effect=[
            RunResult(success=False, stderr="timeout after 300s", error_kind="timeout"),
            RunResult(success=True, stdout="ok", error_kind=""),
        ],
    )

    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.model_versions.append(ModelVersion(stage="final", description="d"))
    coder_node(s)

    # 第 2 次 complete 的 prompt 应含"缩小"关键词，不含"stderr 节选"
    second_prompt = spy_complete.call_args_list[1].args[0]
    assert "缩小" in second_prompt
    assert "stderr 节选" not in second_prompt


def test_coder_prompt_on_runtime_asks_to_fix_via_stderr(mocker, workdir):
    """attempt_0 runtime 失败时，attempt_1 的 prompt 应喂 stderr 让 LLM 修 bug。"""
    from math_agent.tools.runner import RunResult

    drafts = [
        CoderDraft(purpose="s", code="raise ValueError('boom')"),
        CoderDraft(purpose="s", code="print('ok')"),
    ]
    spy_complete = mocker.patch("math_agent.nodes.coder.complete", side_effect=drafts)
    mocker.patch(
        "math_agent.nodes.coder.run_python",
        side_effect=[
            RunResult(success=False, stderr="Traceback ... ValueError: boom",
                      error_kind="runtime"),
            RunResult(success=True, stdout="ok", error_kind=""),
        ],
    )

    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.model_versions.append(ModelVersion(stage="final", description="d"))
    coder_node(s)

    second_prompt = spy_complete.call_args_list[1].args[0]
    assert "stderr 节选" in second_prompt
    assert "ValueError: boom" in second_prompt
    assert "缩小" not in second_prompt
