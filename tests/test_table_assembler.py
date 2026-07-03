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
    assert "x_i" in table
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

