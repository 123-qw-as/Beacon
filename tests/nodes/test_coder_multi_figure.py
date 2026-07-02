"""Plan D Phase 3：coder_node 按 modeler.figure_purposes 拆成 N 个单图调用。

覆盖：
- 多 figure_purposes → 每个图一次 complete 调用
- 无 figure_purposes → 退化为单次调用（向后兼容）
- 每张图独立重试
- 每张图独立 workdir 子目录 fig_{i}_attempt_{j}
"""
from math_agent.state import MathModelingState, ModelVersion
from math_agent.nodes.coder import coder_node, CoderDraft


def test_coder_calls_once_per_figure_purpose(mocker, workdir):
    """modeler 给 3 个 figure_purposes → coder 发 3 次 complete（每次首跑即成功）。"""
    drafts = [
        CoderDraft(purpose="时序图", code="print('fig1')"),
        CoderDraft(purpose="路径图", code="print('fig2')"),
        CoderDraft(purpose="饼图", code="print('fig3')"),
    ]
    spy = mocker.patch("math_agent.nodes.coder.complete", side_effect=drafts)
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.model_versions.append(ModelVersion(
        stage="final", description="d",
        figure_purposes=["需求时序图", "调度路径图", "成本构成饼图"],
    ))
    delta = coder_node(s)

    assert len(delta["code_artifacts"]) == 3
    assert spy.call_count == 3  # 每个图 1 次调用，全部首跑成功
    assert all(a.success for a in delta["code_artifacts"])


def test_coder_falls_back_to_single_call_without_figure_purposes(mocker, workdir):
    """无 figure_purposes → 用 model.description 当 purpose，单次调用（向后兼容）。"""
    mocker.patch(
        "math_agent.nodes.coder.complete",
        return_value=CoderDraft(purpose="solve", code="print('ok')"),
    )
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.model_versions.append(ModelVersion(stage="final", description="the model"))
    delta = coder_node(s)

    assert len(delta["code_artifacts"]) == 1
    assert delta["code_artifacts"][0].success


def test_coder_retries_per_figure_independently(mocker, workdir):
    """fig0 先失败后成功，fig1 首跑成功 → 共 3 次 complete、3 个 artifact。"""
    drafts = [
        CoderDraft(purpose="fig0a", code="raise RuntimeError('x')"),
        CoderDraft(purpose="fig0b", code="print('fig0 ok')"),
        CoderDraft(purpose="fig1", code="print('fig1 ok')"),
    ]
    mocker.patch("math_agent.nodes.coder.complete", side_effect=drafts)
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.model_versions.append(ModelVersion(
        stage="final", description="d",
        figure_purposes=["图0", "图1"],
    ))
    delta = coder_node(s)

    arts = delta["code_artifacts"]
    assert len(arts) == 3  # fig0 失败 + fig0 成功 + fig1 成功
    assert arts[0].success is False
    assert arts[1].success is True
    assert arts[2].success is True


def test_coder_workdir_uses_fig_index(mocker, workdir):
    """每张图拿到独立子目录 fig_{i}_attempt_{j}。"""
    from math_agent.tools.runner import RunResult

    mocker.patch(
        "math_agent.nodes.coder.complete",
        return_value=CoderDraft(purpose="p", code="print('ok')"),
    )
    spy_run = mocker.patch(
        "math_agent.nodes.coder.run_python",
        return_value=RunResult(success=True, stdout="ok", error_kind=""),
    )
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.model_versions.append(ModelVersion(
        stage="final", description="d",
        figure_purposes=["图0", "图1"],
    ))
    coder_node(s)

    # 2 张图各 1 次尝试 → 2 次 run_python；workdir 是关键字参数
    workdir_args = [str(c.kwargs["workdir"]) for c in spy_run.call_args_list]
    assert any("fig_0" in w for w in workdir_args)
    assert any("fig_1" in w for w in workdir_args)


def test_coder_records_error_when_all_figures_fail(mocker, workdir):
    """所有图所有 attempts 都失败 → delta.errors 记录，且计数含全部尝试。"""
    from itertools import cycle
    mocker.patch(
        "math_agent.nodes.coder.complete",
        side_effect=cycle([CoderDraft(purpose="p", code="raise RuntimeError('boom')")]),
    )
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.model_versions.append(ModelVersion(
        stage="final", description="d",
        figure_purposes=["图0", "图1"],
    ))
    delta = coder_node(s)

    assert "errors" in delta and delta["errors"]
    assert delta["errors"][0].startswith("coder:")
    # 2 张图 × (1 + 1 retry) = 4 次尝试全失败
    assert len(delta["code_artifacts"]) == 4
    assert all(a.success is False for a in delta["code_artifacts"])


def test_coder_figure_one_prompt_contains_purpose(mocker, workdir):
    """单图 prompt 应把当前 purpose 写进去，让 LLM 知道画哪张图。"""
    spy = mocker.patch(
        "math_agent.nodes.coder.complete",
        return_value=CoderDraft(purpose="p", code="print('ok')"),
    )
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.model_versions.append(ModelVersion(
        stage="final", description="d",
        figure_purposes=["需求时序图"],
    ))
    coder_node(s)

    prompt = spy.call_args.args[0]
    assert "需求时序图" in prompt
    assert "当前绘图任务" in prompt
