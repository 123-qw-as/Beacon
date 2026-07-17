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
        CoderDraft(purpose="时序图", code="print('RESULT: baseline=ours total_cost=10 service_rate=0.9')"),
        CoderDraft(purpose="路径图", code="print('RESULT: baseline=ours total_cost=10 service_rate=0.9')"),
        CoderDraft(purpose="饼图", code="print('RESULT: baseline=ours total_cost=10 service_rate=0.9')"),
    ]
    spy = mocker.patch("math_agent.nodes.coder.complete", side_effect=drafts)
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.model_versions.append(ModelVersion(
        stage="final", description="d",
        figure_purposes=["需求时序图", "调度路径图", "成本构成饼图"],
    ))
    delta = coder_node(s)

    # Phase 2 起主方案成功后还会追加 baseline 对照方案，按 figure prompt 过滤调用计数
    figure_calls = [c for c in spy.call_args_list if "当前绘图任务" in c.args[0]]
    figure_arts = [a for a in delta["code_artifacts"] if a.category == "figure"]
    assert len(figure_arts) == 3
    assert len(figure_calls) == 3  # 每个图 1 次调用，全部首跑成功
    assert all(a.success for a in figure_arts)
    assert [a.evidence_role for a in figure_arts] == ["primary", "supporting", "supporting"]
    assert "唯一主方案证据" in figure_calls[1].args[0]
    assert all(call.kwargs["profile"] == "code" for call in figure_calls)


def test_coder_falls_back_to_single_call_without_figure_purposes(mocker, workdir):
    """无 figure_purposes → 用 model.description 当 purpose，单次调用（向后兼容）。"""
    mocker.patch(
        "math_agent.nodes.coder.complete",
        return_value=CoderDraft(
            purpose="solve",
            code="print('RESULT: baseline=ours total_cost=10 service_rate=0.9')",
        ),
    )
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.model_versions.append(ModelVersion(stage="final", description="the model"))
    delta = coder_node(s)

    # Phase 2 起会追加 baseline 对照方案；主方案 figure 仍应只有 1 个成功 artifact
    figure_arts = [a for a in delta["code_artifacts"] if a.category == "figure"]
    assert len(figure_arts) == 1
    assert figure_arts[0].success


def test_coder_retries_per_figure_independently(mocker, workdir):
    """fig0 先失败后成功，fig1 首跑成功 → 共 3 次 complete、3 个 artifact。"""
    drafts = [
        CoderDraft(purpose="fig0a", code="raise RuntimeError('x')"),
        CoderDraft(purpose="fig0b", code="print('RESULT: baseline=ours total_cost=10 service_rate=0.9')"),
        CoderDraft(purpose="fig1", code="print('RESULT: baseline=ours total_cost=10 service_rate=0.9')"),
    ]
    mocker.patch("math_agent.nodes.coder.complete", side_effect=drafts)
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.model_versions.append(ModelVersion(
        stage="final", description="d",
        figure_purposes=["图0", "图1"],
    ))
    delta = coder_node(s)

    # Phase 2 起会追加 baseline 对照方案；按 figure category 过滤主方案 artifact
    arts = [a for a in delta["code_artifacts"] if a.category == "figure"]
    assert len(arts) == 3  # fig0 失败 + fig0 成功 + fig1 成功
    assert arts[0].success is False
    assert arts[1].success is True
    assert arts[2].success is True


def test_coder_workdir_uses_fig_index(mocker, workdir):
    """每张图拿到独立子目录 fig_{i}_attempt_{j}。"""
    from math_agent.tools.runner import RunResult

    mocker.patch(
        "math_agent.nodes.coder.complete",
        return_value=CoderDraft(
            purpose="p",
            code="print('RESULT: baseline=ours total_cost=10 service_rate=0.9')",
        ),
    )
    spy_run = mocker.patch(
        "math_agent.nodes.coder.run_python",
        return_value=RunResult(
            success=True,
            stdout="RESULT: baseline=ours total_cost=10 service_rate=0.9",
            error_kind="",
        ),
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
    from math_agent.config import MAX_CODE_RETRIES
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
    # 主求解耗尽重试后立即结束本批次，补充图不得顶替主证据。
    assert len(delta["code_artifacts"]) == MAX_CODE_RETRIES + 1
    assert all(a.success is False for a in delta["code_artifacts"])


def test_coder_figure_one_prompt_contains_purpose(mocker, workdir):
    """单图 prompt 应把当前 purpose 写进去，让 LLM 知道画哪张图。"""
    spy = mocker.patch(
        "math_agent.nodes.coder.complete",
        return_value=CoderDraft(
            purpose="p",
            code="print('RESULT: baseline=ours total_cost=10 service_rate=0.9')",
        ),
    )
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.model_versions.append(ModelVersion(
        stage="final", description="d",
        figure_purposes=["需求时序图"],
    ))
    coder_node(s)

    # Phase 2 起主方案后会追加 baseline 调用；figure prompt 仍是第一个调用
    prompt = spy.call_args_list[0].args[0]
    assert "需求时序图" in prompt
    assert "当前绘图任务" in prompt
