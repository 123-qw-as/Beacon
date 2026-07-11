#!/usr/bin/env python
"""提取上传文件的摘要 JSON，供前端展示和 analyst prompt 注入。

用法：python scripts/extract_file_meta.py <file_path>
输出：stdout 一行 JSON
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def _meta_xlsx(path: Path) -> dict:
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheets = []
    for i, name in enumerate(wb.sheetnames):
        if i >= 5:
            sheets.append({"name": name, "rows": 0, "cols": 0, "columns": [], "preview": []})
            continue
        ws = wb[name]
        rows = list(ws.iter_rows(max_row=6, values_only=True))
        if not rows:
            sheets.append({"name": name, "rows": 0, "cols": 0, "columns": [], "preview": []})
            continue
        columns = [str(c) if c is not None else "" for c in rows[0]]
        preview = [[str(c) if c is not None else "" for c in row] for row in rows[1:6]]
        # ponytail: max_row in read_only 模式不可靠，用 iter_rows 扫一遍计数
        total_rows = sum(1 for _ in ws.iter_rows(values_only=True))
        sheets.append({
            "name": name, "rows": total_rows, "cols": len(columns),
            "columns": columns, "preview": preview,
        })
    wb.close()
    return {"sheets": sheets}


def _meta_csv(path: Path) -> dict:
    import csv
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return {"sheets": [{"name": path.name, "rows": 0, "cols": 0, "columns": [], "preview": []}]}
    columns = [str(c) for c in rows[0]]
    preview = [[str(c) for c in row] for row in rows[1:6]]
    return {"sheets": [{"name": path.name, "rows": len(rows), "cols": len(columns),
                         "columns": columns, "preview": preview}]}


def _meta_pdf(path: Path) -> dict:
    # 优先使用 PyMuPDF (fitz)，其对数学符号/CID 字体的解码远优于 pypdf
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(path))
        raw = "\n\n".join(page.get_text() or "" for page in doc)
        total_pages = len(doc)
        doc.close()
    except ImportError:
        # 降级到 pypdf（数学符号可能显示为方框）
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        raw = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        total_pages = len(reader.pages)
    # lone-surrogate 清洗
    text = raw.encode("utf-8", errors="ignore").decode("utf-8")
    return {"text_excerpt": text[:5000], "total_pages": total_pages}


def _meta_docx(path: Path) -> dict:
    from docx import Document
    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    return {"text_excerpt": text[:3000], "paragraphs": len(paragraphs), "tables": len(doc.tables)}


def _meta_text(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {"text_excerpt": text[:3000], "lines": text.count("\n") + 1}


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: extract_file_meta.py <file_path>"}))
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.is_file():
        print(json.dumps({"error": f"file not found: {path}"}))
        sys.exit(1)

    suffix = path.suffix.lower()
    type_map = {
        ".xlsx": ("xlsx", _meta_xlsx),
        ".xls": ("xlsx", _meta_xlsx),
        ".csv": ("csv", _meta_csv),
        ".pdf": ("pdf", _meta_pdf),
        ".docx": ("docx", _meta_docx),
        ".txt": ("txt", _meta_text),
        ".md": ("txt", _meta_text),
    }
    if suffix not in type_map:
        print(json.dumps({"error": f"unsupported file type: {suffix}"}))
        sys.exit(1)

    file_type, extractor = type_map[suffix]
    try:
        summary = extractor(path)
    except Exception as e:
        print(json.dumps({"error": f"extraction failed: {e}"}))
        sys.exit(1)

    print(json.dumps({
        "file_type": file_type,
        "filename": path.name,
        "summary": summary,
    }, ensure_ascii=False))


if __name__ == "__main__":
    # UTF-8 输出，避免 Windows GBK 控制台截断中文
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    main()
