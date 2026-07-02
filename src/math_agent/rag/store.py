"""sqlite-vec 向量索引。

设计：
- 单进程使用，单文件 sqlite。
- chunks 和向量存两张表，主键 id 关联（避免把文本塞进 vec 表）。
- chunks 表带 (source, content_hash) 唯一约束：re-ingest 同一文件时按内容去重，
  避免重复入库污染检索、浪费 embedding API 费用。
"""
from __future__ import annotations

import hashlib
import sqlite3
import struct
import warnings
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
    source_type: str = ""
    section: str = ""


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    return column in cols


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
        # chunks 表：旧库可能没有 content_hash 列，迁移加列但不补唯一约束（旧数据
        # 空串会冲突），新库才建完整约束。旧库不享受去重，靠 warning 提示删库重建。
        old_db = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone() is not None
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "text TEXT NOT NULL, source TEXT NOT NULL, idx INTEGER NOT NULL, "
            "content_hash TEXT NOT NULL, "
            "source_type TEXT NOT NULL DEFAULT '', "
            "section TEXT NOT NULL DEFAULT '', "
            "UNIQUE(source, content_hash))"
        )
        if old_db and not _has_column(conn, "chunks", "content_hash"):
            # 旧库迁移：加列（空串默认值），不补唯一约束
            conn.execute("ALTER TABLE chunks ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''")
            warnings.warn(
                f"旧 RAG 库 {path} 缺 content_hash 字段，无法去重；"
                "建议删除该库后重新 ingest 以启用去重。",
                stacklevel=2,
            )
        # ponytail: 同形态迁移 source_type/section。旧库这两列为空，
        # writer 过滤 source_type="paper" 会跳过它们 → retrieve fallback 兜回全库。
        # 想精确过滤需重新 ingest。短章节不合并，min-size 视实测再加。
        for col in ("source_type", "section"):
            if old_db and not _has_column(conn, "chunks", col):
                conn.execute(
                    f"ALTER TABLE chunks ADD COLUMN {col} TEXT NOT NULL DEFAULT ''"
                )
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"id INTEGER PRIMARY KEY, embedding float[{dim}])"
        )
        # meta 表持久化 embedding 维度：open 时校验调用方 dim 与库内一致，
        # 不等即报错。防止 vec0 表 CREATE IF NOT EXISTS 静默忽略 dim 差异
        # 导致后续 add/search 维度校验全错的 footgun。
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        row = conn.execute("SELECT value FROM meta WHERE key='dim'").fetchone()
        if row is None:
            # ponytail: 旧库（有 vec_chunks 但无 meta）首次 open 时把调用方 dim
            # 写入并锁定。若首次传错 dim，后续 add/search 会立即报维度不匹配，
            # 用户第一时间发现——可接受的取舍。
            conn.execute("INSERT INTO meta(key, value) VALUES ('dim', ?)", (str(dim),))
        elif int(row[0]) != dim:
            conn.close()
            raise ValueError(
                f"store dim {row[0]} != requested {dim}; "
                "检查 --dim 或环境变量 MATH_AGENT_RAG_DIM，"
                "或删除该库用正确维度重建。"
            )
        conn.commit()
        return cls(conn, dim)

    def add(self, *, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """插入 chunks；返回实际新增条数（命中去重的跳过）。

        去重键：(source, content_hash)。已存在的 chunk 不重插 vec_chunks。
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        for e in embeddings:
            if len(e) != self._dim:
                raise ValueError(f"embedding dim {len(e)} != store dim {self._dim}")
        cur = self._conn.cursor()
        new_count = 0
        for c, e in zip(chunks, embeddings):
            ch = _content_hash(c.text)
            # ponytail: INSERT...RETURNING 抛 IntegrityError on 唯一约束冲突，
            # fallback SELECT 拿既有 id。比 RETURNING+no-op-update 的 rowcount
            # 歧义（rowcount 在 RETURNING 模式下恒为 0）更稳。
            try:
                cur.execute(
                    "INSERT INTO chunks(text, source, idx, content_hash, source_type, section) "
                    "VALUES (?, ?, ?, ?, ?, ?) RETURNING id",
                    (c.text, c.source, c.index, ch, c.source_type, c.section),
                )
                rowid = cur.fetchone()[0]
                new_count += 1
                cur.execute("INSERT INTO vec_chunks(id, embedding) VALUES (?, ?)",
                            (rowid, _to_blob(e)))
            except sqlite3.IntegrityError:
                # 已存在的 chunk：其 vec_chunks 行也已就位，整体跳过
                continue
        self._conn.commit()
        return new_count

    def search(self, query: list[float], *, k: int = 5,
               source_type: str | None = None) -> list[StoredChunk]:
        if len(query) != self._dim:
            raise ValueError(f"query dim {len(query)} != store dim {self._dim}")
        # sqlite-vec 0.1.x KNN 查询要求显式 `k = ?` 约束（LIMIT 单独不够）。
        where = "WHERE v.embedding MATCH ? AND k = ?"
        params: list = [_to_blob(query), k]
        if source_type is not None:
            where += " AND c.source_type = ?"
            params.append(source_type)
        cur = self._conn.execute(
            "SELECT v.id, c.text, c.source, c.idx, v.distance, c.source_type, c.section "
            "FROM vec_chunks v JOIN chunks c ON c.id = v.id "
            f"{where} ORDER BY v.distance",
            params,
        )
        return [
            StoredChunk(id=row[0], text=row[1], source=row[2], index=row[3],
                        score=row[4], source_type=row[5], section=row[6])
            for row in cur
        ]

    def close(self) -> None:
        self._conn.close()
