"""Plan D Phase 2：writer 分章节多调用架构测试。"""
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
from math_agent.nodes.writer import writer_node
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
        return schema(**{f: f"VAL_{f}" for f in fields})

    return _side, calls


# ---------------------------------------------------------------------------
# Pass 1 + Pass 2 调用结构
# ---------------------------------------------------------------------------

def test_writer_makes_outline_then_per_section_calls(mocker):
    """首轮：1 次大纲 + 7 次分章 = 8 次 complete 调用。"""
    side, calls = _make_complete_side_effect()
    mocker.patch("math_agent.nodes.writer.complete", side_effect=side)
    s = _rich_state()

    delta = writer_node(s)

    assert len(calls) == 8
    # 第一次是大纲
    assert calls[0] is WriterOutline
    # 后 7 次是分组 schema（顺序与 writer_sections 一致）
    expected_schemas = [schema_for_group(g.name) for g in writer_sections()]
    assert calls[1:] == expected_schemas
    # 返回的 paper 是 PaperSections 且字段被填充
    assert isinstance(delta["paper"], PaperSections)
    assert delta["paper"].abstract == "VAL_abstract"
    assert delta["paper"].keywords == "VAL_keywords"
    assert delta["paper"].model_section == "VAL_model_section"


def test_writer_first_iter_runs_all_seven_sections(mocker):
    """首轮无 critic 时跑全部 7 分组。"""
    side, calls = _make_complete_side_effect()
    mocker.patch("math_agent.nodes.writer.complete", side_effect=side)
    writer_node(_rich_state())
    # 8 = 1 outline + 7 sections
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

    delta = writer_node(s)

    # 只 1 次调用（solution 分组），无大纲
    assert len(calls) == 1
    assert calls[0] is schema_for_group("solution")
    # solution 被更新
    assert delta["paper"].solution == "VAL_solution"
    # 其余字段保留旧值
    assert delta["paper"].abstract == "OLD_ABSTRACT"
    assert delta["paper"].model_section == "OLD_M"
    assert delta["paper"].keywords == "OLD_KW"
    assert delta["paper"].references == "OLD_R"


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

    delta = writer_node(s)

    # 2 次分组调用，无大纲；顺序遵循 writer_sections
    assert len(calls) == 2
    assert calls[0] is schema_for_group("abstract_problem")
    assert calls[1] is schema_for_group("assumptions_notation")
    # 被更新
    assert delta["paper"].abstract == "VAL_abstract"
    assert delta["paper"].keywords == "VAL_keywords"
    assert delta["paper"].notation == "VAL_notation"
    # 未触及的保留
    assert delta["paper"].solution == "OS"
    assert delta["paper"].model_section == "OM"


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

    delta = writer_node(s)

    # 无大纲 + 7 分组
    assert len(calls) == 7
    expected_schemas = [schema_for_group(g.name) for g in writer_sections()]
    assert calls == expected_schemas


def test_writer_outline_skipped_on_retry(mocker):
    """writer_iteration=1 → 不调用大纲（calls[0] 不是 WriterOutline）。"""
    side, calls = _make_complete_side_effect()
    mocker.patch("math_agent.nodes.writer.complete", side_effect=side)

    s = _rich_state()
    s.writer_iteration = 1
    s.paper = PaperSections(solution="OLD")
    s.critic_reports.append(CriticReport(
        target="paper", score=4, approved=False,
        issues=[CriticIssue(section="conclusion", problem="结论套话")],
    ))

    writer_node(s)

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
    delta = writer_node(s)
    assert delta["writer_iteration"] == 1

    s2 = _rich_state()
    s2.writer_iteration = 1
    s2.paper = PaperSections(solution="x")
    s2.critic_reports.append(CriticReport(
        target="paper", score=5, approved=False,
        issues=[CriticIssue(section="solution", problem="fix")],
    ))
    delta2 = writer_node(s2)
    assert delta2["writer_iteration"] == 2


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
    writer_node(_rich_state())

    # prompts[0] 是大纲（不含 IRON RULES 的完整 6 条？大纲模板也 include 了 partial）
    # 1 大纲 + 7 分章，每个都 include writer_iron_rules.md.j2
    assert len(prompts) == 8
    for p in prompts:
        assert "IRON RULES" in p
        assert "禁编造数据" in p
