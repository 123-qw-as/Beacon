import pytest

from math_agent.rag.store import VectorStore, StoredChunk
from math_agent.rag.chunking import Chunk


def test_store_round_trip(workdir):
    db = workdir / "vec.db"
    store = VectorStore.open(db, dim=3)
    store.add(
        chunks=[Chunk(text="alpha", source="a.md", index=0),
                Chunk(text="beta", source="a.md", index=1)],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    hits = store.search([1.0, 0.0, 0.0], k=2)
    assert isinstance(hits[0], StoredChunk)
    assert hits[0].text == "alpha"
    assert len(hits) == 2
    store.close()


def test_store_persists_across_open(workdir):
    db = workdir / "vec.db"
    s1 = VectorStore.open(db, dim=3)
    s1.add(
        chunks=[Chunk(text="x", source="s", index=0)],
        embeddings=[[1.0, 0.0, 0.0]],
    )
    s1.close()

    s2 = VectorStore.open(db, dim=3)
    hits = s2.search([1.0, 0.0, 0.0], k=1)
    assert hits[0].text == "x"
    s2.close()


def test_store_rejects_dim_mismatch(workdir):
    store = VectorStore.open(workdir / "vec.db", dim=3)
    with pytest.raises(ValueError):
        store.add(
            chunks=[Chunk(text="x", source="s", index=0)],
            embeddings=[[1.0, 0.0]],
        )
    store.close()


def test_store_add_is_idempotent_on_duplicate_source_hash(workdir):
    """同一 (source, content) 调两次 add：第二次不应重复入 vec_chunks。"""
    store = VectorStore.open(workdir / "vec.db", dim=3)
    chunk = Chunk(text="alpha", source="a.md", index=0)
    emb = [1.0, 0.0, 0.0]
    n1 = store.add(chunks=[chunk], embeddings=[emb])
    n2 = store.add(chunks=[chunk], embeddings=[emb])
    assert n1 == 1   # 首次新增
    assert n2 == 0   # 重复，跳过
    hits = store.search(emb, k=10)
    assert len(hits) == 1   # 不翻倍
    store.close()


def test_store_open_rejects_dim_mismatch(workdir):
    """库以 dim=3 建立后，用 dim=4 重新 open 应报错（防止静默 footgun）。"""
    db = workdir / "vec.db"
    s1 = VectorStore.open(db, dim=3)
    s1.add(chunks=[Chunk(text="x", source="s", index=0)],
           embeddings=[[1.0, 0.0, 0.0]])
    s1.close()
    with pytest.raises(ValueError, match="dim"):
        VectorStore.open(db, dim=4)


def test_store_open_persists_dim_across_reopen(workdir):
    """用相同 dim 重新 open 应正常工作（meta 命中，不报错）。"""
    db = workdir / "vec.db"
    s1 = VectorStore.open(db, dim=3)
    s1.add(chunks=[Chunk(text="x", source="s", index=0)],
           embeddings=[[1.0, 0.0, 0.0]])
    s1.close()
    s2 = VectorStore.open(db, dim=3)   # 不抛
    hits = s2.search([1.0, 0.0, 0.0], k=1)
    assert hits[0].text == "x"
    s2.close()


def test_store_search_filters_by_source_type(workdir):
    """source_type 过滤：只返回匹配类型的 chunk。"""
    store = VectorStore.open(workdir / "vec.db", dim=3)
    store.add(
        chunks=[Chunk(text="paper 片段", source="p.md", index=0, source_type="paper")],
        embeddings=[[1.0, 0.0, 0.0]],
    )
    store.add(
        chunks=[Chunk(text="model 片段", source="m.md", index=0, source_type="model_lib")],
        embeddings=[[0.0, 1.0, 0.0]],
    )
    hits = store.search([1.0, 0.0, 0.0], k=10, source_type="paper")
    assert len(hits) == 1
    assert hits[0].source_type == "paper"
    store.close()


def test_store_search_no_filter_returns_all(workdir):
    """不传 source_type → 全返回（向后兼容）。"""
    store = VectorStore.open(workdir / "vec.db", dim=3)
    store.add(
        chunks=[Chunk(text="a", source="p.md", index=0, source_type="paper")],
        embeddings=[[1.0, 0.0, 0.0]],
    )
    store.add(
        chunks=[Chunk(text="b", source="m.md", index=0, source_type="model_lib")],
        embeddings=[[0.0, 1.0, 0.0]],
    )
    hits = store.search([1.0, 0.0, 0.0], k=10)
    assert len(hits) == 2
    store.close()


def test_store_add_persists_metadata(workdir):
    """add 后 search，StoredChunk.source_type/section 正确。"""
    store = VectorStore.open(workdir / "vec.db", dim=3)
    store.add(
        chunks=[Chunk(text="x", source="s", index=0,
                      source_type="model_lib", section="适用场景")],
        embeddings=[[1.0, 0.0, 0.0]],
    )
    hits = store.search([1.0, 0.0, 0.0], k=1)
    assert hits[0].source_type == "model_lib"
    assert hits[0].section == "适用场景"
    store.close()
