from math_agent.rag.chunking import chunk_text, Chunk


def test_chunk_text_returns_chunks_with_overlap():
    text = "abcdefghij" * 50  # 500 字符
    chunks = chunk_text(text, max_chars=200, overlap=50, source="x.md")
    assert all(isinstance(c, Chunk) for c in chunks)
    # 长度大致符合（除末尾）
    assert all(len(c.text) <= 200 for c in chunks)
    # 相邻块有 overlap
    assert chunks[0].text[-50:] == chunks[1].text[:50]
    # 源信息透传
    assert all(c.source == "x.md" for c in chunks)


def test_chunk_text_short_input_one_chunk():
    chunks = chunk_text("hello", max_chars=200, overlap=50, source="s")
    assert len(chunks) == 1
    assert chunks[0].text == "hello"


def test_chunk_text_respects_paragraph_boundary_when_possible():
    para = "段落一" * 50
    text = para + "\n\n" + "段落二" * 50
    chunks = chunk_text(text, max_chars=180, overlap=20, source="s")
    assert any("段落二" in c.text for c in chunks)


def test_chunk_text_rejects_bad_overlap():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("x", max_chars=10, overlap=10, source="s")


def test_chunk_text_empty_returns_empty():
    assert chunk_text("", max_chars=100, overlap=10, source="s") == []


def test_chunk_text_splits_at_headings():
    text = "## 模型建立\n内容A" + "甲" * 20 + "\n\n## 求解\n内容B" + "乙" * 20
    chunks = chunk_text(text, max_chars=200, overlap=10, source="s")
    assert len(chunks) >= 2
    # 两节各自成块，不混入对方标题
    assert any("模型建立" in c.text or c.section == "模型建立" for c in chunks)
    assert any("求解" in c.text or c.section == "求解" for c in chunks)
    # 任一块不同时含两个标题的正文
    assert not any("内容A" in c.text and "内容B" in c.text for c in chunks)


def test_chunk_text_records_section_metadata():
    text = "## 模型建立\n内容A\n\n## 求解\n内容B"
    chunks = chunk_text(text, max_chars=200, overlap=10, source="s")
    sections = {c.section for c in chunks}
    assert "模型建立" in sections
    assert "求解" in sections


def test_chunk_text_no_headings_keeps_empty_section():
    chunks = chunk_text("纯文本无标题", max_chars=200, overlap=10, source="s")
    assert len(chunks) == 1
    assert chunks[0].section == ""


def test_chunk_text_section_over_max_chars_windows_within_section():
    body = "甲" * 300   # 单节正文超 max_chars
    text = f"## 模型建立\n{body}"
    chunks = chunk_text(text, max_chars=100, overlap=20, source="s")
    assert len(chunks) > 1
    # 同一节切出的多块 section 一致
    assert all(c.section == "模型建立" for c in chunks)


def test_chunk_text_source_type_propagated():
    chunks = chunk_text("x", max_chars=100, overlap=10, source="s", source_type="paper")
    assert chunks[0].source_type == "paper"
