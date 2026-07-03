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
