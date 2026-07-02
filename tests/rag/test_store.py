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
