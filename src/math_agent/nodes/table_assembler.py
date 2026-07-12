"""table_assembler 节点：在 writer/critic 循环结束后，
1) 从结构化 state 生成 markdown 表格注入 PaperSections；
2) 对所有 section 做确定性禁用词清洗。

纯代码，不调用 LLM。表格数据来自 model_versions/sensitivity_runs 等结构化字段。
"""
from __future__ import annotations

import re

from math_agent.tools.runner import extract_numeric_results

# 禁用词 → 替换词。顺序敏感：先替换单数 issue 再处理其他。
# ponytail: 用 list 而非 dict，因为同一模式可能需要不同替换上下文。
# Claim/Evidence/Reasoning/issue 只在中文上下文中替换（前后有中文字符），
# 避免破坏纯英文段落（如 abstract 里的英文引用句）。
# 注意：(?i) 不能在 lookbehind 内，用 re.IGNORECASE flag 代替。
_CJK = r"\u4e00-\u9fff"
_FORBIDDEN_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"papercritic", re.IGNORECASE), "[内部评审]"),
    # Claim/Evidence/Reasoning：前面或后面有中文字符才替换（CER 框架泄露场景）
    # 注意：CJK 和 ASCII 之间无 \b（Python \w 含 unicode），用 lookaround CJK 代替
    (re.compile(rf"(?<=[{_CJK}])claim", re.IGNORECASE), "结论"),
    (re.compile(rf"claim(?=[{_CJK}])", re.IGNORECASE), "结论"),
    (re.compile(rf"(?<=[{_CJK}])evidence", re.IGNORECASE), "依据"),
    (re.compile(rf"evidence(?=[{_CJK}])", re.IGNORECASE), "依据"),
    (re.compile(rf"(?<=[{_CJK}])reasoning", re.IGNORECASE), "推理"),
    (re.compile(rf"reasoning(?=[{_CJK}])", re.IGNORECASE), "推理"),
    # 代码[数字] → 代码（只匹配方括号形式，不误吃"代码 45 行"）
    (re.compile(r"代码\s*\[\s*\d+\s*\]"), "代码"),
    # issue 单数只在中文上下文替换
    (re.compile(rf"(?<=[{_CJK}])issue(?!s)", re.IGNORECASE), "问题"),
    (re.compile(rf"issue(?=[{_CJK}])", re.IGNORECASE), "问题"),
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


_UNIT_RE = re.compile(r"^(.*?)\s*[（(]([^()（）]+)[)）]\s*$")


def _sanitize_table_cell(text: str) -> str:
    """清理表格 cell 里的 LaTeX 命令，让它们当纯文本渲染。

    变量名里可能有 \\mathbf{h}、$F_{i,t}$ 等——在 tabularx 里裸用会崩编译。
    ponytail: 不用 \\textbackslash{} 转义（会被 _prepare_section 二次处理拆坏），
    直接删掉反斜杠和 $，保留字母——变量表里不需要渲染数学公式，纯文本够了。
    """
    if not text:
        return text
    # 删掉反斜杠（\mathbf → mathbf，\beta → beta）
    text = text.replace("\\", "")
    # 删掉 $（$F_{i,t}$ → F_{i,t}）
    text = text.replace("$", "")
    # 转义剩余的特殊字符
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("#", r"\#")
    text = text.replace("_", r"\_")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")
    return text


def _generate_variable_table(variables: dict[str, str]) -> str:
    """从 model_versions[-1].variables 生成符号说明 markdown 表。

    description 含括号单位则拆分（"需求量(件)" → 含义"需求量" / 单位"件"）。
    返回空字符串如果 variables 为空。
    cell 内容做 LaTeX 转义，避免变量名里的 \\mathbf{} $F_{i,t}$ 崩 tabularx。
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
        lines.append(f"| {_sanitize_table_cell(name)} | {_sanitize_table_cell(meaning)} | {_sanitize_table_cell(unit)} |")
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


# baseline category → 中文显示名
_BASELINE_NAMES = {
    "no_schedule": "无调度",
    "simple_pred": "简单平均预测",
    "greedy": "贪婪启发式",
    "ours": "本文方案",
}


def _generate_comparison_table(artifacts: list) -> str:
    """从 code_artifacts 中提取 baseline 对照结果生成对比表。

    主方案（category='figure'）的 stdout 如果也含 RESULT: baseline=ours 也纳入。
    注意：指标列顺序取决于 artifact 顺序——若各方案输出不同指标，缺失列填 —。
    无 baseline artifacts 或无 RESULT 行时返回空字符串。
    """
    rows: list[dict[str, str]] = []
    for a in artifacts:
        results = extract_numeric_results(a.stdout) if a.stdout else {}
        if not results:
            if a.category.startswith("baseline:"):
                cat_key = a.category.split(":", 1)[1]
                name = _BASELINE_NAMES.get(cat_key, cat_key)
                rows.append({"方案": name, "状态": "运行失败"})
            continue
        for identifier, metrics in results.items():
            name = _BASELINE_NAMES.get(identifier, identifier)
            row = {"方案": name}
            row.update({k: str(v) for k, v in metrics.items()})
            rows.append(row)

    if not rows:
        return ""

    all_metrics: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen and k != "方案":
                seen.add(k)
                all_metrics.append(k)

    if not all_metrics:
        all_metrics = ["状态"]

    header = "| 方案 | " + " | ".join(all_metrics) + " |"
    sep = "|---|" + "|".join(["---" for _ in all_metrics]) + "|"
    lines = [header, sep]
    for r in rows:
        cells = [r.get("方案", "—")]
        for m in all_metrics:
            cells.append(r.get(m, "—"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _inject_table(section_text: str, title: str, table_md: str) -> str:
    """把表格注入 section 文本末尾。若已含同名 ## title 则跳过（去重）。

    table_md 为空则原样返回（表格生成器无数据时）。
    """
    if not table_md:
        return section_text
    heading = f"## {title}"
    if heading in section_text:
        return section_text  # 已存在，不重复注入
    if section_text and not section_text.endswith("\n"):
        section_text += "\n"
    return f"{section_text}\n{heading}\n\n{table_md}\n"


from math_agent.state import MathModelingState, PaperSections


# 要清洗的 section 字段名。
# 注意：references 不在此列表中——参考文献含真实英文文献标题/期刊名，
# 其中 Evidence/Issue/Claim/Reasoning 等是合法英文单词，清洗会破坏引用。
# 禁用词是内部流程产物（PaperCritic、CER 框架术语泄漏到中文正文），
# 不会出现在格式规范的参考文献条目中。
_SECTION_FIELDS = [
    "abstract", "problem_restatement", "assumptions", "notation",
    "model_section", "solution", "sensitivity", "conclusion",
]


def table_assembler_node(state: MathModelingState) -> dict:
    """writer/critic 循环后的后处理：注入表格 + 清洗禁用词。

    返回增量 dict: {"paper": PaperSections, "table_warnings": list[str]}。
    """
    paper = state.paper.model_copy(deep=True)
    warnings: list[str] = []

    # 1) 生成并注入表格
    final_model = next((m for m in reversed(state.model_versions) if m.stage == "final"),
                       state.model_versions[-1] if state.model_versions else None)
    if final_model and final_model.variables:
        var_table = _generate_variable_table(final_model.variables)
        paper.notation = _inject_table(paper.notation, "模型变量表", var_table)

    sens_table = _generate_sensitivity_table(state.sensitivity_runs)
    paper.sensitivity = _inject_table(paper.sensitivity, "敏感性结果汇总表", sens_table)

    # 对比表（从 baseline artifacts 提取）
    comp_table = _generate_comparison_table(state.latest_code_artifacts())
    paper.solution = _inject_table(paper.solution, "各方案结果对比表", comp_table)

    # 2) 禁用词清洗（所有 section）
    for field in _SECTION_FIELDS:
        text = getattr(paper, field, "")
        if text:
            cleaned, w = _clean_forbidden_words(text, field)
            setattr(paper, field, cleaned)
            warnings.extend(w)

    return {"paper": paper, "table_warnings": warnings}
