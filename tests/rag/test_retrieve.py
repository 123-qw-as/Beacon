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


def test_format_snippets_truncates_at_block_boundary():
    """max_chars 只够 1 条：输出含完整第 1 块，不含第 2 块的 ## [2] 标记。"""
    snips = [
        Snippet(text="甲" * 80, source="src1.md", score=0.1),
        Snippet(text="乙" * 80, source="src2.md", score=0.2),
        Snippet(text="丙" * 80, source="src3.md", score=0.3),
    ]
    out = format_snippets(snips, max_chars=200)
    assert len(out) <= 200 + 20   # 预留截断标记
    assert "已截断" in out
    assert "src1.md" in out          # 第 1 块完整保留
    assert "## [2]" not in out       # 第 2 块未被加入
    assert "显示 1 条" in out


def test_format_snippets_keeps_all_when_under_max():
    """短 snippet + 宽松 max_chars：全部保留，无截断标记。"""
    snips = [Snippet(text="短内容", source="s.md", score=0.1)]
    out = format_snippets(snips, max_chars=500)
    assert "已截断" not in out
    assert "短内容" in out


def test_format_snippets_single_block_over_max_truncates_block():
    """单条极长 + 小 max_chars：含 header、含截断标记、长度受控。"""
    snips = [Snippet(text="巨" * 500, source="big.md", score=0.0)]
    out = format_snippets(snips, max_chars=120)
    assert len(out) <= 120 + 10
    assert "已截断" in out
    assert "big.md" in out   # 来源仍可见


def test_search_filters_by_source_type(mocker, workdir):
    """库里 paper + model_lib，source_type=paper 只回 paper。"""
    from math_agent.rag.store import VectorStore
    from math_agent.rag.chunking import Chunk

    store = VectorStore.open(workdir / "vec.db", dim=3)
    store.add(
        chunks=[Chunk(text="paper 内容", source="p.md", index=0, source_type="paper")],
        embeddings=[[1.0, 0.0, 0.0]],
    )
    store.add(
        chunks=[Chunk(text="model 内容", source="m.md", index=0, source_type="model_lib")],
        embeddings=[[0.0, 1.0, 0.0]],
    )
    store.close()

    mocker.patch("math_agent.rag.retrieve.embed_texts",
                 return_value=[[1.0, 0.0, 0.0]])
    out = search("q", db_path=workdir / "vec.db", k=10, dim=3,
                 embedding_model="m", source_type="paper")
    assert len(out) == 1
    assert out[0].source_type == "paper"


def test_search_falls_back_when_filter_empty(mocker, workdir):
    """库里只有 model_lib，source_type=paper 过滤无结果 → fallback 回全库。"""
    from math_agent.rag.store import VectorStore
    from math_agent.rag.chunking import Chunk

    store = VectorStore.open(workdir / "vec.db", dim=3)
    store.add(
        chunks=[Chunk(text="model 内容", source="m.md", index=0, source_type="model_lib")],
        embeddings=[[1.0, 0.0, 0.0]],
    )
    store.close()

    mocker.patch("math_agent.rag.retrieve.embed_texts",
                 return_value=[[1.0, 0.0, 0.0]])
    out = search("q", db_path=workdir / "vec.db", k=10, dim=3,
                 embedding_model="m", source_type="paper")
    assert len(out) == 1   # fallback 回 model_lib
    assert out[0].source_type == "model_lib"


def test_search_no_filter_works_as_before(mocker, workdir):
    """不传 source_type → 全返回（向后兼容）。"""
    from math_agent.rag.store import VectorStore
    from math_agent.rag.chunking import Chunk

    store = VectorStore.open(workdir / "vec.db", dim=3)
    store.add(
        chunks=[Chunk(text="a", source="p.md", index=0, source_type="paper")],
        embeddings=[[1.0, 0.0, 0.0]],
    )
    store.add(
        chunks=[Chunk(text="b", source="m.md", index=0, source_type="model_lib")],
        embeddings=[[0.0, 1.0, 0.0]],
    )
    store.close()

    mocker.patch("math_agent.rag.retrieve.embed_texts",
                 return_value=[[1.0, 0.0, 0.0]])
    out = search("q", db_path=workdir / "vec.db", k=10, dim=3,
                 embedding_model="m")
    assert len(out) == 2
