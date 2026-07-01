from math_agent.rag.retrieve import search, format_snippets, Snippet


def test_search_returns_empty_when_db_missing(workdir):
    out = search("hello", db_path=workdir / "nonexistent.db", k=3)
    assert out == []


def test_search_returns_snippets(mocker, workdir):
    from math_agent.rag.store import VectorStore
    from math_agent.rag.chunking import Chunk

    store = VectorStore.open(workdir / "vec.db", dim=3)
    store.add(
        chunks=[Chunk(text="alpha", source="s", index=0)],
        embeddings=[[1.0, 0.0, 0.0]],
    )
    store.close()

    mocker.patch(
        "math_agent.rag.retrieve.embed_texts",
        return_value=[[1.0, 0.0, 0.0]],
    )

    out = search("query", db_path=workdir / "vec.db", k=1,
                 embedding_model="m", dim=3)
    assert len(out) == 1
    assert isinstance(out[0], Snippet)
    assert out[0].text == "alpha"


def test_format_snippets_empty_returns_empty():
    assert format_snippets([]) == ""


def test_format_snippets_truncates_when_over_max():
    snips = [Snippet(text="x" * 1000, source="s", score=0.0)]
    out = format_snippets(snips, max_chars=200)
    assert len(out) <= 200 + len("\n...（已截断）") + 10
    assert "已截断" in out
