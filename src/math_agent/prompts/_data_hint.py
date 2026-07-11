"""供 coder/sensitivity prompt 共用的数据文件路径提示。"""
from __future__ import annotations
import os


def build_data_hint(data_dir: str | None, data_files: list) -> str:
    """构造数据文件路径提示文本。

    data_files: list[DataFileInfo]，需要有 .filename / .file_type / .path 属性。
    """
    if not data_dir or not data_files:
        return ""
    lines = [f"数据目录: {data_dir}"]
    for df in data_files:
        fp = os.path.join(data_dir, df.path) if not os.path.isabs(df.path) else df.path
        if df.file_type in ("xlsx", "xls"):
            lines.append(f"- {fp} (Excel, 用 pd.read_excel 读取)")
        elif df.file_type == "csv":
            lines.append(f"- {fp} (CSV, 用 pd.read_csv 读取)")
        elif df.file_type == "pdf":
            lines.append(f"- {fp} (PDF, 需用 pypdf 提取文本)")
        elif df.file_type == "docx":
            lines.append(f"- {fp} (Word, 需用 python-docx 读取)")
        else:
            lines.append(f"- {fp} (文本文件)")
    return (
        "\n# 可用数据文件\n" + "\n".join(lines) + "\n"
        "请优先读取这些真实数据进行计算，不要编造 mock 数据。\n"
        "路径中含中文/空格时用 r-string：pd.read_excel(r\"...\")\n"
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
