"""Plan D Phase 2：writer 分章节多调用架构测试。

Phase 1.5 改造后：writer_node 是 prep（大纲 + 填充队列），writer_section_node 逐节写。
测试通过 prep + 循环 section 模拟完整写作流程。
"""
from math_agent.state import (
    MathModelingState,
    ModelVersion,
    CodeArtifact,
    PaperSections,
    Assumption,
    SensitivityRun,
    FigureArtifact,
    CriticReport,
    CriticIssue,
)
from math_agent.nodes.writer import writer_node, writer_section_node
from math_agent.prompts.writer_section import (
    WriterOutline,
    _sections_to_rewrite,
    schema_for_group,
    writer_sections,
)


def _rich_state() -> MathModelingState:
    s = MathModelingState(problem="共享单车调度优化")
    s.assumptions.extend([
        Assumption(statement="需求服从 Poisson 分布", rationale="日志拟合 KS=0.92"),
    ])
    s.model_versions.append(ModelVersion(
        stage="final", description="带容量约束的排队网络",
        equations=[r"\lambda_i(t) = \alpha_i"],
        variables={"lambda": "到达率"},
    ))
    s.code_artifacts.append(CodeArtifact(
        purpose="求解最优调度", code="...", success=True, stdout="FINAL_RESULT=42.7",
    ))
    return s


def _make_complete_side_effect():
    """返回一个 side_effect，按 schema 类型返回对应实例。

    - WriterOutline → 含 thesis 的 outline
    - 其余分组 schema → 该 schema 的实例，字段填入可识别的标记值
    记录每次调用的 schema 以便断言调用序列。
    """
    calls = []

    def _side(prompt, *, schema=None, **kwargs):
        calls.append(schema)
        if schema is WriterOutline:
            return WriterOutline(
                abstract="摘要论点", problem_restatement="问题论点",
                assumptions="假设论点", notation="符号论点",
                model_section="模型论点", solution="求解论点",
                sensitivity="敏感性论点", conclusion="结论论点",
                references="参考文献论点",
            )
        # 分组 schema：按字段名填标记值
        fields = schema.model_fields
        return schema(**{f: f"VAL_{f}" + "甲" * 7000 for f in fields})

    return _side, calls


def _run_writer_full(s):
    """跑完整 writer 子流程：prep + 逐节 drain 队列，返回最终 paper。"""
    delta = writer_node(s)
    s.writer_iteration = delta["writer_iteration"]
    s.writer_outline_dump = delta["writer_outline_dump"]
    s.writer_retrieved_context = delta["writer_retrieved_context"]
    s.writer_section_queue = delta["writer_section_queue"]
    while s.writer_section_queue:
        step = writer_section_node(s)
        s.paper = step["paper"]
        s.writer_section_queue = step["writer_section_queue"]
    return s.paper


# ---------------------------------------------------------------------------
# Pass 1 + Pass 2 调用结构
# ---------------------------------------------------------------------------

def test_writer_makes_outline_then_per_section_calls(mocker):
    """首轮：大纲之后，七个分组都按本题证据调用 writer。"""
    side, calls = _make_complete_side_effect()
    mocker.patch("math_agent.nodes.writer.complete", side_effect=side)
    s = _rich_state()

    paper = _run_writer_full(s)

    assert len(calls) == 8
    # 第一次是大纲
    assert calls[0] is WriterOutline
    expected_schemas = [schema_for_group(g.name) for g in writer_sections()]
    assert calls[1:] == expected_schemas
    # 最终 paper 字段被填充
    assert isinstance(paper, PaperSections)
    assert paper.abstract.startswith("VAL_abstract")
    assert paper.keywords.startswith("VAL_keywords")
    assert paper.model_section.startswith("VAL_model_section")
    assert paper.sensitivity.startswith("VAL_sensitivity")


def test_writer_first_iter_runs_all_seven_sections(mocker):
    """首轮跑全部分组；通用题不复用其他题目的固定事实稿。"""
    side, calls = _make_complete_side_effect()
    mocker.patch("math_agent.nodes.writer.complete", side_effect=side)
    _run_writer_full(_rich_state())
    assert len(calls) == 8


# ---------------------------------------------------------------------------
# 重试轮：只重跑被标记的分组
# ---------------------------------------------------------------------------

def test_writer_second_iter_only_rewrites_flagged_sections(mocker):
    """writer_iteration=1 + critic 标记 solution → 只跑 solution 分组（1 次，无大纲）。"""
    side, calls = _make_complete_side_effect()
    mocker.patch("math_agent.nodes.writer.complete", side_effect=side)

    s = _rich_state()
    s.writer_iteration = 1
    # 上一轮已产出的 paper，非重写字段须保留
    s.paper = PaperSections(
        abstract="OLD_ABSTRACT", problem_restatement="OLD_PR",
        assumptions="OLD_A", notation="OLD_N",
        model_section="OLD_M", solution="OLD_S",
        sensitivity="OLD_SE", conclusion="OLD_C", references="OLD_R",
        keywords="OLD_KW",
    )
    s.critic_reports.append(CriticReport(
        target="paper", score=4, approved=False,
        issues=[CriticIssue(section="solution", problem="solution 数字未追溯")],
    ))

    paper = _run_writer_full(s)

    # 只 1 次 section 调用（solution 分组），无大纲
    assert len(calls) == 1
    assert calls[0] is schema_for_group("solution")
    # solution 被更新
    assert paper.solution.startswith("VAL_solution")
    # 其余字段保留旧值
    assert paper.abstract == "OLD_ABSTRACT"
    assert paper.model_section == "OLD_M"
    assert paper.keywords == "OLD_KW"
    assert paper.references == "OLD_R"


def test_writer_second_iter_multiple_flagged_groups(mocker):
    """critic 标记 abstract + notation → 跑 abstract_problem 与 assumptions_notation 两组。"""
    side, calls = _make_complete_side_effect()
    mocker.patch("math_agent.nodes.writer.complete", side_effect=side)

    s = _rich_state()
    s.writer_iteration = 1
    s.paper = PaperSections(
        abstract="OA", problem_restatement="OP", assumptions="OAS",
        notation="ON", model_section="OM", solution="OS",
        sensitivity="OSE", conclusion="OC", references="OR", keywords="OKW",
    )
    s.critic_reports.append(CriticReport(
        target="paper", score=5, approved=False,
        issues=[
            CriticIssue(section="abstract", problem="摘要缺数字"),
            CriticIssue(section="notation", problem="符号表不全"),
        ],
    ))

    paper = _run_writer_full(s)

    # 2 次分组调用，无大纲；顺序遵循 writer_sections
    assert len(calls) == 2
    assert calls[0] is schema_for_group("abstract_problem")
    assert calls[1] is schema_for_group("assumptions_notation")
    # 被更新
    assert paper.abstract.startswith("VAL_abstract")
    assert paper.keywords.startswith("VAL_keywords")
    assert paper.notation.startswith("VAL_notation")
    # 未触及的保留
    assert paper.solution == "OS"
    assert paper.model_section == "OM"


def test_writer_general_issue_rewrites_all_sections(mocker):
    """CriticIssue(section='general') → 全部 7 分组重跑（仍无大纲）。"""
    side, calls = _make_complete_side_effect()
    mocker.patch("math_agent.nodes.writer.complete", side_effect=side)

    s = _rich_state()
    s.writer_iteration = 1
    s.paper = PaperSections(abstract="OA", solution="OS")
    s.critic_reports.append(CriticReport(
        target="paper", score=3, approved=False,
        issues=[CriticIssue(section="general", problem="整体结构松散")],
    ))

    _run_writer_full(s)

    # 无大纲；全部分组都按当前题目证据重写
    assert len(calls) == 7
    expected_schemas = [schema_for_group(g.name) for g in writer_sections()]
    assert calls == expected_schemas


def test_writer_outline_skipped_on_retry(mocker):
    """writer_iteration=1 → 不调用大纲（calls 中无 WriterOutline）。"""
    side, calls = _make_complete_side_effect()
    mocker.patch("math_agent.nodes.writer.complete", side_effect=side)

    s = _rich_state()
    s.writer_iteration = 1
    s.paper = PaperSections(solution="OLD")
    s.critic_reports.append(CriticReport(
        target="paper", score=4, approved=False,
        issues=[CriticIssue(section="conclusion", problem="结论套话")],
    ))

    _run_writer_full(s)

    assert WriterOutline not in calls
    # 只跑了 conclusion 分组
    assert calls == [schema_for_group("conclusion")]


# ---------------------------------------------------------------------------
# 迭代计数
# ---------------------------------------------------------------------------

def test_writer_increments_iteration_multi_call(mocker):
    side, calls = _make_complete_side_effect()
    mocker.patch("math_agent.nodes.writer.complete", side_effect=side)

    s = _rich_state()
    _run_writer_full(s)
    assert s.writer_iteration == 1

    s2 = _rich_state()
    s2.writer_iteration = 1
    s2.paper = PaperSections(solution="x")
    s2.critic_reports.append(CriticReport(
        target="paper", score=5, approved=False,
        issues=[CriticIssue(section="solution", problem="fix")],
    ))
    _run_writer_full(s2)
    assert s2.writer_iteration == 2


# ---------------------------------------------------------------------------
# _sections_to_rewrite 单元测试
# ---------------------------------------------------------------------------

def test_sections_to_rewrite_general_returns_all():
    res = _sections_to_rewrite([CriticIssue(section="general", problem="x")])
    assert res == [g.name for g in writer_sections()]


def test_sections_to_rewrite_single_field():
    res = _sections_to_rewrite([CriticIssue(section="model_section", problem="x")])
    assert res == ["model"]


def test_sections_to_rewrite_dedup_same_group():
    # abstract 与 problem_restatement 同属 abstract_problem → 去重为一组
    res = _sections_to_rewrite([
        CriticIssue(section="abstract", problem="a"),
        CriticIssue(section="problem_restatement", problem="p"),
    ])
    assert res == ["abstract_problem"]


def test_sections_to_rewrite_preserves_order():
    res = _sections_to_rewrite([
        CriticIssue(section="references", problem="r"),
        CriticIssue(section="abstract", problem="a"),
    ])
    # 按 writer_sections 原顺序：abstract_problem 在 references 之前
    assert res == ["abstract_problem", "references"]


def test_sections_to_rewrite_empty_returns_all():
    res = _sections_to_rewrite([])
    assert res == [g.name for g in writer_sections()]


# ---------------------------------------------------------------------------
# 分组 schema 完整性
# ---------------------------------------------------------------------------

def test_every_group_has_schema_and_template():
    for g in writer_sections():
        sch = schema_for_group(g.name)
        # schema 字段必须与 group.fields 一致
        assert set(sch.model_fields.keys()) == set(g.fields)


def test_section_prompt_contains_iron_rules(mocker):
    """每个分章 prompt 都包含 IRON RULES（来自共享 partial）。"""
    side, calls = _make_complete_side_effect()
    # 捕获传给 complete 的 prompt 文本
    prompts = []
    orig_side = side

    def capture(prompt, *, schema=None, **kwargs):
        prompts.append(prompt)
        return orig_side(prompt, schema=schema, **kwargs)

    mocker.patch("math_agent.nodes.writer.complete", side_effect=capture)
    _run_writer_full(_rich_state())

    # 1 次大纲 + 7 个分组
    assert len(prompts) == 8
    for p in prompts:
        assert "IRON RULES" in p
        assert "禁编造数据" in p


def test_writer_retries_once_when_section_is_below_content_budget(mocker):
    state = _rich_state()
    state.writer_iteration = 1
    state.writer_outline_dump = WriterOutline(solution="求解锚点").model_dump()
    state.writer_section_queue = ["solution"]
    calls = []

    def complete_side_effect(prompt, *, schema=None, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            return schema(solution="过短")
        return schema(solution="VAL_solution" + "甲" * 3200)

    mocker.patch("math_agent.nodes.writer.complete", side_effect=complete_side_effect)

    delta = writer_section_node(state)

    assert len(calls) == 2
    assert "篇幅质量门禁" in calls[1]
    assert delta["paper"].solution.startswith("VAL_solution")


def test_writer_retry_reuses_first_outline(mocker):
    side, calls = _make_complete_side_effect()
    mocker.patch("math_agent.nodes.writer.complete", side_effect=side)
    s = _rich_state()
    s.writer_iteration = 1
    s.writer_outline_dump = WriterOutline(solution="保留的求解锚点").model_dump()
    s.critic_reports.append(CriticReport(
        target="paper", score=4, approved=False,
        issues=[CriticIssue(section="solution", problem="需重写")],
    ))

    delta = writer_node(s)

    assert delta["writer_outline_dump"]["solution"] == "保留的求解锚点"
    assert WriterOutline not in calls


def test_available_numbers_ignores_non_result_debug_values():
    from math_agent.prompts.writer_section import _extract_available_numbers
    from math_agent.state import CodeArtifact
    s = _rich_state()
    s.code_artifacts = [CodeArtifact(
        purpose="实验", code="", success=True,
        stdout="epoch=99 loss=0.1\nRESULT: baseline=ours cost=12.5 service=0.9",
    )]
    numbers = _extract_available_numbers(s)
    assert "RESULT:" in numbers
    assert "cost=12.5" in numbers
    assert "epoch=99" not in numbers


def test_available_numbers_ignores_old_coder_batches():
    from math_agent.prompts.writer_section import _extract_available_numbers
    from math_agent.state import CodeArtifact
    state = _rich_state()
    state.code_artifacts = [
        CodeArtifact(
            purpose="old", code="", success=True, batch=1,
            stdout="RESULT: baseline=old cost=999",
        ),
        CodeArtifact(
            purpose="new", code="", success=True, batch=2,
            stdout="RESULT: baseline=ours cost=100",
        ),
    ]
    numbers = _extract_available_numbers(state)
    assert "999" not in numbers
    assert "100" in numbers


def test_writer_evidence_keeps_verified_depth_lines_for_primary_solver():
    from math_agent.prompts.writer_section import _compact_code_artifacts, _extract_available_numbers
    from math_agent.state import CodeArtifact

    state = _rich_state()
    state.code_artifacts = [CodeArtifact(
        purpose="主方案",
        code="# BEACON_GREEN_LOGISTICS_SAFE_SOLVER",
        success=True,
        evidence_role="primary",
        stdout=(
            "RESULT: baseline=ours total_cost=100 vehicles=2 service_rate=1\n"
            "ALGORITHM_SEARCH: initial_score=120 final_score=100 improvement_rate=.1667\n"
            "ROBUSTNESS: scenarios=200 timewin_p05=.81 cost_p95=121\n"
            "SERVICE_DIAGNOSTICS: late_tasks=14 max_late_min=42\n"
            "DYNAMIC_EVENTS: scenarios=50 new_order_success_rate=.72 fallback_rate=.32\n"
        ),
    )]

    compact = _compact_code_artifacts(state)
    numbers = _extract_available_numbers(state)

    assert "ROBUSTNESS: scenarios=200" in compact[0].stdout
    assert "DYNAMIC_EVENTS: scenarios=50" in compact[0].stdout
    assert "cost_p95=121" in numbers
    assert "max_late_min=42" in numbers


def test_writer_evidence_rejects_exit_zero_errors_and_impossible_counts():
    from math_agent.prompts.writer_section import _compact_code_artifacts, _extract_available_numbers
    from math_agent.state import CodeArtifact, DataFileInfo

    state = _rich_state()
    state.data_files = [DataFileInfo(
        filename="订单.xlsx", file_type="xlsx", path="订单.xlsx",
        summary={"rows": 2170},
    )]
    state.code_artifacts = [
        CodeArtifact(
            purpose="吞异常", code="", success=True,
            stdout="Error during execution: '纬度'\nRESULT: baseline=ours total_cost=0 service_rate=0",
        ),
        CodeArtifact(
            purpose="越界", code="", success=True,
            stdout="RESULT: baseline=ours total_cost=4812127.99 veh_count=19011 service_rate=0.92",
        ),
        CodeArtifact(
            purpose="有效", code="", success=True,
            stdout="RESULT: baseline=ours total_cost=2470.93 vehicles=7 service_rate=1.0",
        ),
    ]

    numbers = _extract_available_numbers(state)
    compact = _compact_code_artifacts(state)

    assert "4812127.99" not in numbers
    assert "2470.93" in numbers
    assert [item.purpose for item in compact] == ["有效"]


def test_writer_ignores_supporting_figure_metrics():
    from math_agent.prompts.writer_section import _extract_available_numbers
    from math_agent.state import CodeArtifact

    state = _rich_state()
    state.code_artifacts = [
        CodeArtifact(
            purpose="主求解", code="", success=True, evidence_role="primary",
            stdout="RESULT: baseline=ours total_cost=100 service_rate=0.95",
        ),
        CodeArtifact(
            purpose="补充图", code="", success=True, evidence_role="supporting",
            stdout="RESULT: baseline=ours total_cost=999 service_rate=0.80",
        ),
    ]

    numbers = _extract_available_numbers(state)

    assert "total_cost=100" in numbers
    assert "total_cost=999" not in numbers
