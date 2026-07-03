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
