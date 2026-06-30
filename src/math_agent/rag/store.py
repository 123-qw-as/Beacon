"""sqlite-vec 向量索引。

设计：
- 单进程使用，单文件 sqlite。
- chunks 和向量存两张表，主键 id 关联（避免把文本塞进 vec 表）。
"""
from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from math_agent.rag.chunking import Chunk


@dataclass
class StoredChunk:
    id: int
    text: str
    source: str
    index: int
    score: float


def _to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class VectorStore:
    def __init__(self, conn: sqlite3.Connection, dim: int):
        self._conn = conn
        self._dim = dim

    @classmethod
    def open(cls, path: str | Path, *, dim: int) -> "VectorStore":
        conn = sqlite3.connect(str(path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "text TEXT NOT NULL, source TEXT NOT NULL, idx INTEGER NOT NULL)"
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"id INTEGER PRIMARY KEY, embedding float[{dim}])"
        )
        conn.commit()
        return cls(conn, dim)

    def add(self, *, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        for e in embeddings:
            if len(e) != self._dim:
                raise ValueError(f"embedding dim {len(e)} != store dim {self._dim}")
        cur = self._conn.cursor()
        for c, e in zip(chunks, embeddings):
            cur.execute(
                "INSERT INTO chunks(text, source, idx) VALUES (?, ?, ?)",
                (c.text, c.source, c.index),
            )
            rowid = cur.lastrowid
            cur.execute("INSERT INTO vec_chunks(id, embedding) VALUES (?, ?)",
                        (rowid, _to_blob(e)))
        self._conn.commit()

    def search(self, query: list[float], *, k: int = 5) -> list[StoredChunk]:
        if len(query) != self._dim:
            raise ValueError(f"query dim {len(query)} != store dim {self._dim}")
        # sqlite-vec 0.1.x KNN 查询要求显式 `k = ?` 约束（LIMIT 单独不够）。
        cur = self._conn.execute(
            "SELECT v.id, c.text, c.source, c.idx, v.distance "
            "FROM vec_chunks v JOIN chunks c ON c.id = v.id "
            "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
            (_to_blob(query), k),
        )
        return [
            StoredChunk(id=row[0], text=row[1], source=row[2], index=row[3], score=row[4])
            for row in cur
        ]

    def close(self) -> None:
        self._conn.close()
