"""table_assembler 节点：在 writer/critic 循环结束后，
1) 从结构化 state 生成 markdown 表格注入 PaperSections；
2) 对所有 section 做确定性禁用词清洗。

纯代码，不调用 LLM。表格数据来自 model_versions/sensitivity_runs 等结构化字段。
"""
from __future__ import annotations

import re

# 禁用词 → 替换词。顺序敏感：先替换单数 issue 再处理其他。
# ponytail: 用 list 而非 dict，因为同一模式可能需要不同替换上下文。
_FORBIDDEN_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)papercritic"), "[内部评审]"),
    (re.compile(r"(?i)\bclaim\b"), "结论"),
    (re.compile(r"(?i)\bevidence\b"), "依据"),
    (re.compile(r"(?i)\breasoning\b"), "推理"),
    (re.compile(r"代码\s*\[\s*\d+\s*\]"), "代码"),
    (re.compile(r"代码\s*\d+"), "代码"),
    (re.compile(r"(?i)\bissue\b(?!s)"), "问题"),       # 单数 issue，保留复数 issues
    (re.compile(r"回应\s*[:：]"), "处理:"),
    (re.compile(r"回应"), "处理"),
    (re.compile(r"超时"), "运行"),
    (re.compile(r"占位"), "--"),
    (re.compile(r"李华"), "队员A"),
    (re.compile(r"张三"), "队员A"),
    (re.compile(r"王五"), "队员B"),
]


def _clean_forbidden_words(text: str, section: str) -> tuple[str, list[str]]:
    """对单个 section 文本做确定性禁用词清洗。

    返回 (清洗后文本, 警告列表)。警告格式: "替换: <old> → <new>"。
    """
    if not text:
        return text, []
    warnings: list[str] = []
    for pattern, replacement in _FORBIDDEN_PATTERNS:
        if pattern.search(text):
            text = pattern.sub(replacement, text)
            warnings.append(f"[{section}] {pattern.pattern} → {replacement}")
    return text, warnings


import re as _re  # 已有 import re，但 _UNIT_RE 需要独立引用

_UNIT_RE = _re.compile(r"^(.*?)\s*[（(]([^()（）]+)[)）]\s*$")


def _generate_variable_table(variables: dict[str, str]) -> str:
    """从 model_versions[-1].variables 生成符号说明 markdown 表。

    description 含括号单位则拆分（"需求量(件)" → 含义"需求量" / 单位"件"）。
    返回空字符串如果 variables 为空。
    """
    if not variables:
        return ""
    lines = ["| 符号 | 含义 | 单位 |", "|---|---|---|"]
    for name, desc in variables.items():
        m = _UNIT_RE.match(desc)
        if m:
            meaning, unit = m.group(1).strip(), m.group(2).strip()
        else:
            meaning, unit = desc.strip(), "—"
        lines.append(f"| {name} | {meaning} | {unit} |")
    return "\n".join(lines)


def _sensitivity_rating(results: list[float]) -> str:
    """(max-min)/|mean| → 高/中/低。"""
    if not results or len(results) < 2:
        return "—"
    mean = sum(results) / len(results)
    if mean == 0:
        return "—"
    ratio = (max(results) - min(results)) / abs(mean)
    if ratio > 0.30:
        return "高"
    if ratio > 0.10:
        return "中"
    return "低"


def _generate_sensitivity_table(runs: list) -> str:
    """从 SensitivityRun 列表生成敏感性结果汇总 markdown 表。"""
    if not runs:
        return ""
    lines = ["| 参数 | 取值范围 | 指标 | 指标变化范围 | 敏感性评级 |",
             "|---|---|---|---|---|"]
    for r in runs:
        vals = f"[{r.values[0]}, {r.values[-1]}]" if r.values else "—"
        res = f"[{min(r.results):.4g}, {max(r.results):.4g}]" if r.results else "—"
        rating = _sensitivity_rating(r.results)
        lines.append(f"| {r.parameter} | {vals} | {r.metric} | {res} | {rating} |")
    return "\n".join(lines)
