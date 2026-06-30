"""统一检索入口；节点只 import 这一个函数。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from math_agent.rag.embeddings import embed_texts
from math_agent.rag.store import VectorStore


@dataclass
class Snippet:
    text: str
    source: str
    score: float


def search(
    query: str,
    *,
    db_path: str | Path,
    k: int = 5,
    embedding_model: str = "text-embedding-3-small",
    dim: int = 1536,
) -> list[Snippet]:
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    vec = embed_texts([query], model=embedding_model)[0]
    store = VectorStore.open(db_path, dim=dim)
    try:
        rows = store.search(vec, k=k)
    finally:
        store.close()
    return [Snippet(text=r.text, source=r.source, score=r.score) for r in rows]


def format_snippets(snippets: list[Snippet], *, max_chars: int | None = None) -> str:
    """供 prompt 拼接的统一格式。

    max_chars: 若给出则把整段输出截到该长度（含 header），避免推爆上下文。
    """
    if not snippets:
        return ""
    parts = ["# 检索到的参考资料（仅供启发，不可照抄）"]
    for i, s in enumerate(snippets, 1):
        parts.append(f"## [{i}] 来源：{s.source}\n{s.text}")
    out = "\n\n".join(parts)
    if max_chars is not None and len(out) > max_chars:
        out = out[:max_chars] + "\n...（已截断）"
    return out
