"""离线 ingest：扫描语料目录 → 切块 → 嵌入 → 入库。

支持后缀：.md, .txt, .pdf
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from math_agent.rag.chunking import chunk_text
from math_agent.rag.embeddings import embed_texts
from math_agent.rag.store import VectorStore


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


@dataclass
class IngestReport:
    files_processed: int
    chunks_added: int
    skipped: list[str]


def _extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _extract_pdf_text(path)
    return path.read_text(encoding="utf-8")


def ingest_directory(
    *,
    src_dir: str | Path,
    db_path: str | Path,
    embedding_model: str,
    dim: int,
    max_chars: int = 1200,
    overlap: int = 200,
) -> IngestReport:
    src_dir = Path(src_dir)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    store = VectorStore.open(db_path, dim=dim)
    files_processed = 0
    chunks_added = 0
    skipped: list[str] = []
    try:
        for p in sorted(src_dir.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            try:
                text = _read_file(p)
            except Exception as e:
                skipped.append(f"{p}: {e}")
                continue
            chunks = chunk_text(text, max_chars=max_chars, overlap=overlap, source=str(p))
            if not chunks:
                continue
            embeddings = embed_texts([c.text for c in chunks], model=embedding_model)
            store.add(chunks=chunks, embeddings=embeddings)
            files_processed += 1
            chunks_added += len(chunks)
    finally:
        store.close()
    return IngestReport(files_processed=files_processed,
                        chunks_added=chunks_added, skipped=skipped)
