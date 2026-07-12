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


def _sanitize_text(s: str) -> str:
    """剥掉 lone-surrogate / NUL 等 utf-8 不能编码的字符。

    pypdf 在某些 CID-encoded 中文 PDF 上偶尔产出 lone surrogate（U+D800-U+DFFF），
    后续 httpx encode_request 会抛 UnicodeEncodeError 把整个 ingest 挂掉。
    """
    if not s:
        return s
    # encode 一次 utf-8 with errors=ignore，等价于丢掉所有不可编码字符
    return s.encode("utf-8", errors="ignore").decode("utf-8")


def _normalize_math_text(text: str) -> str:
    """规范化 LaTeX PDF 提取的数学文本。

    - LaTeX \\neq 输出为组合短斜线 U+0338 + =，合并为 ≠ (U+2260)
    - LaTeX \\mu 输出为 micro sign U+00B5，统一为希腊 μ (U+03BC)
    """
    text = text.replace("̸=", "≠")     # \neq
    text = text.replace("̸<", "≮")     # \nless
    text = text.replace("̸>", "≯")     # \ngtr
    text = text.replace("̸≤", "≰")     # \nleq
    text = text.replace("̸≥", "≱")     # \ngeq
    text = text.replace("̸∼", "≄")     # \nsim
    text = text.replace("̸≈", "≇")     # \napprox
    text = text.replace("\u00b5", "\u03bc")  # micro sign -> greek mu
    return text


# SymbolMT 等 PUA 字体的字符映射（Word/WPS 生成的 PDF 常见）
_SYMBOL_FONT_MAPS = {
    "SymbolMT": {
        "\uf03d": "=", "\uf02d": "-", "\uf02b": "+", "\uf02f": "/",
        "\uf020": " ", "\uf0d7": "\u00d7", "\uf0b4": "\u2032",
        "\uf0b0": "\u00b0", "\uf070": "\u03c0", "\uf061": "\u03b1",
        "\uf062": "\u03b2", "\uf067": "\u03b3", "\uf071": "\u03b8",
        "\uf06c": "\u03bb", "\uf06d": "\u03bc", "\uf073": "\u03c3",
        "\uf045": "\u2208", "\uf05e": "\u222b", "\uf0d5": "\u221e",
        "\uf0d6": "\u2211",
    },
}
_SYMBOL_FONT_MAPS["Symbol"] = _SYMBOL_FONT_MAPS["SymbolMT"]


def _map_symbol_font(text: str, font: str) -> str:
    """映射 SymbolMT 等 PUA 字体的字符到标准 Unicode。"""
    for font_prefix, char_map in _SYMBOL_FONT_MAPS.items():
        if font.startswith(font_prefix):
            return "".join(char_map.get(c, c) for c in text)
    return text


def _extract_page_with_superscripts(page) -> str:
    """提取页面文本，处理上标和公式排序。

    - 按 x 坐标排序 span（公式中 span 顺序可能乱）
    - SymbolMT 等 PUA 字体字符映射到标准 Unicode
    - 字号明显小于基线的 span 包裹为 ^{...}（上标还原）
    """
    blocks = page.get_text("dict")["blocks"]
    lines_out = []
    for block in blocks:
        all_spans = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t = span["text"]
                if not t.strip():
                    continue
                font = span.get("font", "")
                size = round(span["size"], 1)
                x0 = span["bbox"][0]
                t = _map_symbol_font(t, font)
                all_spans.append({"text": t, "font": font, "size": size, "x0": x0})
        if not all_spans:
            continue

        formula_spans = [s for s in all_spans if "Symbol" in s["font"] or "Italic" in s["font"]]
        is_formula = len(formula_spans) > 0 and len(all_spans) > 3

        if is_formula:
            formula_only = [s for s in all_spans if "SimSun" not in s["font"] and "SimHei" not in s["font"]]
            if formula_only:
                baseline_size = max(set(s["size"] for s in formula_only), key=lambda sz: sum(1 for s in formula_only if s["size"] == sz))
            else:
                baseline_size = max(set(s["size"] for s in all_spans), key=lambda sz: sum(1 for s in all_spans if s["size"] == sz))
            all_spans.sort(key=lambda s: s["x0"])
            parts = []
            for s in all_spans:
                if s["size"] < baseline_size - 1.5:
                    parts.append(f"^{{{s['text']}}}")
                else:
                    parts.append(s["text"])
            lines_out.append("".join(parts))
        else:
            for line in block.get("lines", []):
                spans = [s for s in line.get("spans", []) if s["text"].strip()]
                if not spans:
                    continue
                sizes = [round(s["size"], 1) for s in spans]
                baseline_size = max(set(sizes), key=sizes.count)
                parts = []
                for span in spans:
                    text = _map_symbol_font(span["text"], span.get("font", ""))
                    size = round(span["size"], 1)
                    if size < baseline_size - 1.5:
                        parts.append(f"^{{{text}}}")
                    else:
                        parts.append(text)
                lines_out.append("".join(parts))
    return "\n".join(lines_out)


def _extract_pdf_text(path: Path) -> str:
    # 优先使用 PyMuPDF (fitz)，其对数学符号/CID 字体的解码远优于 pypdf
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(path))
        raw = "\n\n".join(_extract_page_with_superscripts(page) for page in doc)
        doc.close()
    except ImportError:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        raw = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return _normalize_math_text(_sanitize_text(raw))


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _extract_pdf_text(path)
    return _sanitize_text(path.read_text(encoding="utf-8", errors="ignore"))


# 路径含 papers/论文 → paper（写作风格）；其余 → model_lib（建模参考）。
_PAPER_DIR_HINTS = {"papers", "论文"}


def _derive_source_type(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    return "paper" if (parts & _PAPER_DIR_HINTS) else "model_lib"


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
            chunks = chunk_text(text, max_chars=max_chars, overlap=overlap,
                                source=str(p), source_type=_derive_source_type(p))
            if not chunks:
                continue
            new_chunks = store.missing_chunks(chunks)
            if new_chunks:
                embeddings = embed_texts([c.text for c in new_chunks], model=embedding_model)
                added = store.add(chunks=new_chunks, embeddings=embeddings)
            else:
                added = 0
            files_processed += 1
            chunks_added += added
    finally:
        store.close()
    return IngestReport(files_processed=files_processed,
                        chunks_added=chunks_added, skipped=skipped)
