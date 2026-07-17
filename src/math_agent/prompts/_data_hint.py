"""供 coder/sensitivity prompt 共用的数据文件路径提示。"""
from __future__ import annotations
from functools import lru_cache
import json
import os
from pathlib import Path


@lru_cache(maxsize=64)
def _profile_file(path_text: str, file_type: str, mtime_ns: int) -> str:
    """读取真实表头、dtype 和两行样例；mtime 参与缓存键，文件变化后自动失效。"""
    del mtime_ns  # 仅用于缓存失效
    try:
        import pandas as pd

        path = Path(path_text)
        if file_type in ("xlsx", "xls"):
            frame = pd.read_excel(path, nrows=3)
        elif file_type == "csv":
            frame = pd.read_csv(path, nrows=3)
        else:
            return ""
        selected = list(frame.columns[:12])
        dtypes = {str(column): str(frame[column].dtype) for column in selected}
        sample_columns = selected[:8]
        samples = []
        for record in frame.loc[:, sample_columns].head(2).to_dict(orient="records"):
            samples.append({str(key): str(value)[:80] for key, value in record.items()})
        return (
            "\n  真实读取契约: columns="
            + json.dumps([str(column) for column in selected], ensure_ascii=False)
            + "; dtypes=" + json.dumps(dtypes, ensure_ascii=False)
            + "; 前2行样例=" + json.dumps(samples, ensure_ascii=False)
        )
    except Exception as exc:
        return f"\n  运行时数据画像不可用: {type(exc).__name__}"


def build_data_hint(data_dir: str | None, data_files: list) -> str:
    """构造数据文件路径提示文本。

    data_files: list[DataFileInfo]，需要有 .filename / .file_type / .path 属性。
    """
    if not data_dir or not data_files:
        return ""
    safe_data_dir = Path(data_dir).as_posix()
    lines = [f"数据目录（可直接交给 pathlib.Path）: {safe_data_dir}"]
    for df in data_files:
        fp = os.path.join(data_dir, df.path) if not os.path.isabs(df.path) else df.path
        fp = Path(fp).as_posix()
        summary = df.summary or {}
        columns = summary.get("columns", [])
        if isinstance(columns, list) and columns:
            schema = f"；实际列名: {', '.join(map(str, columns[:20]))}"
        else:
            schema = ""
        rows = summary.get("rows")
        if isinstance(rows, (int, float)):
            schema += f"；约 {int(rows)} 行"
        if df.file_type in ("xlsx", "xls"):
            lines.append(f"- {fp} (Excel, 用 pd.read_excel 读取{schema})")
        elif df.file_type == "csv":
            lines.append(f"- {fp} (CSV, 用 pd.read_csv 读取{schema})")
        elif df.file_type == "pdf":
            lines.append(f'- {fp} (PDF, 需用 pymupdf 提取文本: import fitz; doc=fitz.open(r"{fp}"); text="\\n".join(p.get_text() for p in doc))')
        elif df.file_type == "docx":
            lines.append(f"- {fp} (Word, 需用 python-docx 读取)")
        else:
            lines.append(f"- {fp} (文本文件)")
        local_path = Path(fp)
        if local_path.is_file() and df.file_type in ("xlsx", "xls", "csv"):
            lines.append(_profile_file(
                str(local_path), df.file_type, local_path.stat().st_mtime_ns,
            ))
    return (
        "\n# 可用数据文件\n" + "\n".join(lines) + "\n"
        "请优先读取这些真实数据进行计算，不要编造 mock 数据。\n"
        "必须按上面列出的实际列名读取，不得猜测‘经度/纬度’等不存在的字段。\n"
        "跨表主键必须显式映射：先根据各表真实列名统一重命名为内部 customer_id；"
        "不得假设所有表使用同一个原始列名。\n"
        "时间窗若为 HH:MM 字符串，必须先转换为从 0:00 起的分钟或小时数，再与到达时间比较。\n"
        "Windows 路径统一使用上面的正斜杠形式或 pathlib.Path；"
        "不要创建以单个反斜杠结尾的 raw string（会导致 SyntaxError）。\n"
    )


def build_data_summary_hint(data_files: list) -> str:
    """构造数据摘要提示文本（供 analyst 用，不含绝对路径）。"""
    if not data_files:
        return ""
    lines = []
    for df in data_files:
        line = f"- {df.filename} ({df.file_type})"
        summary = df.summary or {}
        if "sheets" in summary:
            for s in summary["sheets"][:5]:
                cols = ", ".join(s.get("columns", [])[:8])
                lines.append(f"  └ {s['name']}: {s.get('rows',0)}行×{s.get('cols',0)}列 [{cols}]")
        elif "text_excerpt" in summary:
            excerpt = summary["text_excerpt"][:200].replace("\n", " ")
            lines.append(f"  └ 文本摘录: {excerpt}...")
        lines.append(line)
    return (
        "\n# 附件数据概况\n已有以下数据文件可用：\n" + "\n".join(lines) + "\n"
        "请在 data_requirements 中将对应字段标注为 given，并在建模路线中考虑如何使用这些真实数据。\n"
    )
