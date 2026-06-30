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
