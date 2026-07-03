"""table_assembler 单元测试：禁用词清洗 + 表格生成。"""
from math_agent.nodes.table_assembler import _clean_forbidden_words


def test_clean_replaces_papercritic():
    text = "本文 PaperCritic 评分较高"
    cleaned, warnings = _clean_forbidden_words(text, "model_section")
    assert "PaperCritic" not in cleaned
    assert "[内部评审]" in cleaned
    assert len(warnings) == 1


def test_clean_replaces_claim_evidence_reasoning():
    text = "Claim: 成本下降。Evidence: 代码输出。Reasoning: 优化有效。"
    cleaned, warnings = _clean_forbidden_words(text, "solution")
    assert "Claim" not in cleaned
    assert "结论" in cleaned
    assert "依据" in cleaned
    assert "推理" in cleaned
    assert len(warnings) == 3


def test_clean_replaces_code_number():
    text = "见代码1和代码[2]的输出"
    cleaned, warnings = _clean_forbidden_words(text, "solution")
    assert "代码1" not in cleaned
    assert "代码[2]" not in cleaned
    assert "代码" in cleaned


def test_clean_replaces_placeholder_names():
    text = "队员李华和张三、王五参与"
    cleaned, warnings = _clean_forbidden_words(text, "conclusion")
    assert "李华" not in cleaned
    assert "张三" not in cleaned
    assert "王五" not in cleaned
    assert "队员A" in cleaned


def test_clean_replaces_timeout_and_placeholder():
    text = "代码超时，结果为占位"
    cleaned, warnings = _clean_forbidden_words(text, "solution")
    assert "超时" not in cleaned
    assert "占位" not in cleaned


def test_clean_preserves_clean_text():
    text = "本文建立了一个混合整数规划模型，求解得到最优成本 1245.3。"
    cleaned, warnings = _clean_forbidden_words(text, "model_section")
    assert cleaned == text
    assert warnings == []


def test_clean_handles_empty_string():
    cleaned, warnings = _clean_forbidden_words("", "abstract")
    assert cleaned == ""
    assert warnings == []


from math_agent.nodes.table_assembler import _generate_variable_table


def test_variable_table_basic():
    variables = {"x_i": "决策变量，第i个时段的调度量", "d_i": "需求量(件)", "alpha": "学习率"}
    table = _generate_variable_table(variables)
    assert "| 符号 | 含义 | 单位 |" in table
    assert "|---|---|---|" in table
    assert "x\_i" in table  # 下划线被 LaTeX 转义
    assert "决策变量，第i个时段的调度量" in table
    assert "件" in table          # 从 "需求量(件)" 拆出单位
    assert "—" in table           # alpha 无单位 → 填 —


def test_variable_table_empty():
    table = _generate_variable_table({})
    assert table == ""


def test_variable_table_no_unit_in_parens():
    variables = {"s": "存量"}
    table = _generate_variable_table({"s": "存量"})
    lines = table.strip().split("\n")
    # 表头 + 分隔行 + 1 数据行 = 3 行
    assert len(lines) == 3
    assert "—" in lines[2]        # 无单位


from math_agent.state import SensitivityRun
from math_agent.nodes.table_assembler import _generate_sensitivity_table


def test_sensitivity_table_basic():
    runs = [
        SensitivityRun(parameter="alpha", values=[0.1, 0.5, 1.0],
                       metric="MAE", results=[10.0, 20.0, 30.0]),
    ]
    table = _generate_sensitivity_table(runs)
    assert "| 参数 | 取值范围 | 指标 | 指标变化范围 | 敏感性评级 |" in table
    assert "alpha" in table
    assert "MAE" in table
    assert "高" in table          # (30-10)/20 = 100% > 30% → 高


def test_sensitivity_table_medium_rating():
    runs = [
        SensitivityRun(parameter="beta", values=[1, 2, 3],
                       metric="cost", results=[100.0, 115.0, 108.0]),
    ]
    table = _generate_sensitivity_table(runs)
    # (115-100)/107.67 ≈ 13.9% → 中
    assert "中" in table


def test_sensitivity_table_low_rating():
    runs = [
        SensitivityRun(parameter="gamma", values=[1, 2, 3],
                       metric="rate", results=[0.90, 0.91, 0.905]),
    ]
    table = _generate_sensitivity_table(runs)
    # (0.91-0.90)/0.905 ≈ 1.1% → 低
    assert "低" in table


def test_sensitivity_table_empty():
    table = _generate_sensitivity_table([])
    assert table == ""


from math_agent.nodes.table_assembler import _inject_table


def test_inject_table_appends_when_absent():
    text = "这是原有内容。"
    table = "| A | B |\n|---|---|\n| 1 | 2 |"
    result = _inject_table(text, "参数表", table)
    assert "## 参数表" in result
    assert "这是原有内容。" in result
    assert "| A | B |" in result


def test_inject_table_skips_when_already_present():
    text = "原有内容。\n\n## 参数表\n\n已有表格"
    table = "| A | B |\n|---|---|\n| 1 | 2 |"
    result = _inject_table(text, "参数表", table)
    # 不重复注入
    assert result.count("## 参数表") == 1
    assert "| A | B |" not in result


def test_inject_table_empty_table_returns_unchanged():
    text = "原有内容。"
    result = _inject_table(text, "参数表", "")
    assert result == text


from math_agent.state import MathModelingState, ModelVersion, PaperSections, SensitivityRun
from math_agent.nodes.table_assembler import table_assembler_node


def _state_for_assembler():
    s = MathModelingState(problem="test")
    s.model_versions.append(ModelVersion(
        stage="final",
        description="final model",
        variables={"x_i": "调度量(件)", "d_i": "需求量", "alpha": "学习率"},
    ))
    s.sensitivity_runs.append(SensitivityRun(
        parameter="alpha", values=[0.1, 0.5, 1.0],
        metric="MAE", results=[10.0, 20.0, 30.0],
    ))
    s.paper = PaperSections(
        model_section="基础预测模型：使用 XGBoost。PaperCritic 给了好评。",
        notation="原有符号表",
        sensitivity="敏感性分析正文",
        solution="求解过程见代码1。Claim: 最优。",
        conclusion="模型优点多。",
    )
    return s


def test_table_assembler_node_injects_variable_table():
    s = _state_for_assembler()
    result = table_assembler_node(s)
    new_paper = result["paper"]
    assert "| 符号 | 含义 | 单位 |" in new_paper.notation
    assert "x\_i" in new_paper.notation  # 下划线被 LaTeX 转义


def test_table_assembler_node_injects_sensitivity_table():
    s = _state_for_assembler()
    result = table_assembler_node(s)
    new_paper = result["paper"]
    assert "| 参数 | 取值范围 |" in new_paper.sensitivity
    assert "alpha" in new_paper.sensitivity


def test_table_assembler_node_cleans_forbidden_words():
    s = _state_for_assembler()
    result = table_assembler_node(s)
    new_paper = result["paper"]
    assert "PaperCritic" not in new_paper.model_section
    assert "Claim" not in new_paper.solution
    assert "代码1" not in new_paper.solution
    assert len(result["table_warnings"]) >= 3


def test_table_assembler_node_handles_empty_state():
    s = MathModelingState(problem="empty")
    s.paper = PaperSections()
    result = table_assembler_node(s)
    # 不崩，warnings 可能为空
    assert "paper" in result
    assert "table_warnings" in result


def test_references_section_not_cleaned():
    """references 含真实英文文献标题，不应被禁用词清洗破坏。"""
    from math_agent.nodes.table_assembler import _SECTION_FIELDS
    assert "references" not in _SECTION_FIELDS

    # 直接测试：含 Evidence/Issue 的参考文献文本应原样保留
    from math_agent.nodes.table_assembler import table_assembler_node
    from math_agent.state import MathModelingState, PaperSections
    s = MathModelingState(problem="test")
    s.paper = PaperSections(
        references="[1] Smith J. Evidence-Based Optimization. Journal of Math, Issue 12, 2023."
    )
    result = table_assembler_node(s)
    assert "Evidence" in result["paper"].references
    assert "Issue" in result["paper"].references


from math_agent.nodes.table_assembler import _generate_comparison_table
from math_agent.state import CodeArtifact


def test_comparison_table_from_baselines():
    artifacts = [
        CodeArtifact(
            purpose="无调度对照", code="", success=True,
            stdout="RESULT: baseline=no_schedule total_cost=1245.3 service_rate=0.82",
            category="baseline:no_schedule",
        ),
        CodeArtifact(
            purpose="贪婪对照", code="", success=True,
            stdout="RESULT: baseline=greedy total_cost=980.0 service_rate=0.91",
            category="baseline:greedy",
        ),
        CodeArtifact(
            purpose="主方案", code="", success=True,
            stdout="RESULT: baseline=ours total_cost=750.5 service_rate=0.95",
            category="figure",
        ),
    ]
    table = _generate_comparison_table(artifacts)
    assert "| 方案 |" in table
    assert "无调度" in table
    assert "1245.3" in table
    assert "980.0" in table
    assert "750.5" in table


def test_comparison_table_empty_when_no_baselines():
    artifacts = [
        CodeArtifact(purpose="main", code="", success=True, stdout="", category="figure"),
    ]
    table = _generate_comparison_table(artifacts)
    assert table == ""


def test_comparison_table_handles_failed_baselines():
    artifacts = [
        CodeArtifact(
            purpose="无调度对照", code="", success=False,
            stdout="", stderr="error", category="baseline:no_schedule",
        ),
        CodeArtifact(
            purpose="贪婪对照", code="", success=True,
            stdout="RESULT: baseline=greedy total_cost=980.0",
            category="baseline:greedy",
        ),
    ]
    table = _generate_comparison_table(artifacts)
    assert "运行失败" in table
    assert "980.0" in table


def test_table_assembler_node_injects_comparison_table():
    """table_assembler 应把对比表注入 paper.solution。"""
    from math_agent.state import MathModelingState, ModelVersion, PaperSections, CodeArtifact
    s = MathModelingState(problem="test")
    s.model_versions.append(ModelVersion(
        stage="final", description="m", variables={"x": "v"},
    ))
    s.code_artifacts = [
        CodeArtifact(
            purpose="无调度", code="", success=True,
            stdout="RESULT: baseline=no_schedule total_cost=1245.3 service_rate=0.82",
            category="baseline:no_schedule",
        ),
        CodeArtifact(
            purpose="贪婪", code="", success=True,
            stdout="RESULT: baseline=greedy total_cost=980.0 service_rate=0.91",
            category="baseline:greedy",
        ),
    ]
    s.paper = PaperSections(solution="## 求解算法与流程\n求解过程。")
    result = table_assembler_node(s)
    assert "| 方案 |" in result["paper"].solution
    assert "无调度" in result["paper"].solution

