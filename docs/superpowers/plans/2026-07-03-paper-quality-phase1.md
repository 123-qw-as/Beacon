# Paper Quality Improvement — Phase 1 Implementation Plan

> **历史方案记录（非现行事实源）**：路径、状态、命令和待办保留当时上下文；当前实现与限制见 [`../../README.md`](../../README.md)。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `table_assembler` node that injects structured tables + cleans forbidden words after the writer/critic loop, rewrite paper templates to an 8-section judge-friendly structure, and curate the appendix — all without new code-execution runs.

**Architecture:** Insert `table_assembler` between `paper_critic` and `evaluation` in the graph. It reads structured state (variables, sensitivity_runs), generates 2 markdown tables, injects them into `PaperSections` fields, then regex-cleans forbidden words. Templates + writer prompts are rewritten so `model_section`/`solution` use `##` subsection headings. Appendix truncation is one-line Python in `latex_node`, no Jinja2 filters.

**Tech Stack:** Python 3.11, LangGraph, Pydantic, Jinja2, pytest.

**Spec:** `docs/superpowers/specs/2026-07-03-paper-quality-improvement-design.md`

**Scope:** Phase 1 only. Phase 2 (baseline runs, numeric extraction) is a separate plan after Phase 1 validates.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/math_agent/state.py` | Modify | Add `table_warnings: list[str]` field to state |
| `src/math_agent/nodes/table_assembler.py` | Create | Table generation + forbidden-word cleaning (~150 lines) |
| `src/math_agent/graph.py` | Modify | Register `table_assembler` node, redirect `advance` edge |
| `src/math_agent/templates/writer_iron_rules.md.j2` | Modify | Relax RULE 4 heading ban; add forbidden-word rule |
| `src/math_agent/templates/writer_section_model.md.j2` | Modify | 8-section structure: 3 subsections with `##` |
| `src/math_agent/templates/writer_section_solution.md.j2` | Modify | 2 subsections with `##` (algorithm + experiment) |
| `src/math_agent/templates/gmcm.tex.j2` | Modify | Section headings + curated appendix |
| `src/math_agent/templates/paper.tex.j2` | Modify | Section headings + add appendix block |
| `src/math_agent/templates/paper.md.j2` | Modify | Section headings + curated appendix |
| `src/math_agent/nodes/latex.py` | Modify | Appendix truncation in `latex_node` |
| `tests/test_table_assembler.py` | Create | Unit tests for table gen + cleaning |
| `scripts/e2e_table_assembler.py` | Create | Integration test: mock state → run node → verify output |

---

## Task 1: Add `table_warnings` field to state

**Files:**
- Modify: `src/math_agent/state.py:158`
- Test: `tests/test_state.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_state.py` (append at end of file):

```python
def test_state_has_table_warnings_field():
    """table_assembler writes cleaning warnings here; evaluation/latex can read them."""
    s = MathModelingState(problem="x")
    assert hasattr(s, "table_warnings")
    assert s.table_warnings == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_state.py::test_state_has_table_warnings_field -v`
Expected: FAIL with `AttributeError: 'MathModelingState' object has no attribute 'table_warnings'`

- [ ] **Step 3: Add the field**

In `src/math_agent/state.py`, find the `errors` field (L158):

```python
    errors: Annotated[list[str], add] = Field(default_factory=list)
```

Add immediately after it:

```python
    # table_assembler 产出的清洗/注入警告（覆盖语义；每次 table_assembler 运行整体替换）
    table_warnings: list[str] = Field(default_factory=list)
```

Note: `table_warnings` uses **overwrite** semantics (not `Annotated[..., add]`) because each `table_assembler` run produces a fresh complete list. If the node ever runs multiple times (e.g. via checkpoint resume), stale warnings should not accumulate.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_state.py::test_state_has_table_warnings_field -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/math_agent/state.py tests/test_state.py
git commit -m "feat(state): add table_warnings field for table_assembler node"
```

---

## Task 2: Create `_clean_forbidden_words` function with tests

**Files:**
- Create: `src/math_agent/nodes/table_assembler.py`
- Test: `tests/test_table_assembler.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_table_assembler.py`:

```python
"""table_assembler 单元测试：禁用词清洗 + 表格生成。"""
from math_agent.nodes.table_assembler import _clean_forbidden_words


def test_clean_replaces_papercritic():
    text = "本文 PaperCritic 评分较高"
    cleaned, warnings = _clean_forbidden_words(text, "model_section")
    assert "PaperCritic" not in cleaned
    assert "[内部评审]" in cleaned
    assert len(warnings) == 1


def test_clean_replaces_claim_evidence_reasoning():
    text = "Claim: 成本下降。Evidence: 代码输出。Reasoning: 优化有效。"
    cleaned, warnings = _clean_forbidden_words(text, "solution")
    assert "Claim" not in cleaned
    assert "结论" in cleaned
    assert "依据" in cleaned
    assert "推理" in cleaned
    assert len(warnings) == 3


def test_clean_replaces_code_number():
    text = "见代码1和代码[2]的输出"
    cleaned, warnings = _clean_forbidden_words(text, "solution")
    assert "代码1" not in cleaned
    assert "代码[2]" not in cleaned
    assert "代码" in cleaned


def test_clean_replaces_placeholder_names():
    text = "队员李华和张三、王五参与"
    cleaned, warnings = _clean_forbidden_words(text, "conclusion")
    assert "李华" not in cleaned
    assert "张三" not in cleaned
    assert "王五" not in cleaned
    assert "队员A" in cleaned


def test_clean_replaces_timeout_and_placeholder():
    text = "代码超时，结果为占位"
    cleaned, warnings = _clean_forbidden_words(text, "solution")
    assert "超时" not in cleaned
    assert "占位" not in cleaned


def test_clean_preserves_clean_text():
    text = "本文建立了一个混合整数规划模型，求解得到最优成本 1245.3。"
    cleaned, warnings = _clean_forbidden_words(text, "model_section")
    assert cleaned == text
    assert warnings == []


def test_clean_handles_empty_string():
    cleaned, warnings = _clean_forbidden_words("", "abstract")
    assert cleaned == ""
    assert warnings == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_table_assembler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'math_agent.nodes.table_assembler'`

- [ ] **Step 3: Write minimal implementation**

Create `src/math_agent/nodes/table_assembler.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_table_assembler.py -v`
Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/math_agent/nodes/table_assembler.py tests/test_table_assembler.py
git commit -m "feat(table_assembler): add forbidden-word cleaner with tests"
```

---

## Task 3: Create `_generate_variable_table` function with tests

**Files:**
- Modify: `src/math_agent/nodes/table_assembler.py`
- Test: `tests/test_table_assembler.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_table_assembler.py`:

```python
from math_agent.nodes.table_assembler import _generate_variable_table


def test_variable_table_basic():
    variables = {"x_i": "决策变量，第i个时段的调度量", "d_i": "需求量(件)", "alpha": "学习率"}
    table = _generate_variable_table(variables)
    assert "| 符号 | 含义 | 单位 |" in table
    assert "|---|---|---|" in table
    assert "x_i" in table
    assert "决策变量，第i个时段的调度量" in table
    assert "件" in table          # 从 "需求量(件)" 拆出单位
    assert "—" in table           # alpha 无单位 → 填 —


def test_variable_table_empty():
    table = _generate_variable_table({})
    assert table == ""


def test_variable_table_no_unit_in_parens():
    variables = {"s": "存量"}
    table = _generate_variable_table({"s": "存量"})
    lines = table.strip().split("\n")
    # 表头 + 分隔行 + 1 数据行 = 3 行
    assert len(lines) == 3
    assert "—" in lines[2]        # 无单位
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_table_assembler.py::test_variable_table_basic -v`
Expected: FAIL with `ImportError: cannot import name '_generate_variable_table'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/math_agent/nodes/table_assembler.py` (after `_clean_forbidden_words`):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_table_assembler.py -v`
Expected: all tests PASS (7 + 3 = 10)

- [ ] **Step 5: Commit**

```bash
git add src/math_agent/nodes/table_assembler.py tests/test_table_assembler.py
git commit -m "feat(table_assembler): add variable table generator with tests"
```

---

## Task 4: Create `_generate_sensitivity_table` function with tests

**Files:**
- Modify: `src/math_agent/nodes/table_assembler.py`
- Test: `tests/test_table_assembler.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_table_assembler.py`:

```python
from math_agent.state import SensitivityRun
from math_agent.nodes.table_assembler import _generate_sensitivity_table


def test_sensitivity_table_basic():
    runs = [
        SensitivityRun(parameter="alpha", values=[0.1, 0.5, 1.0],
                       metric="MAE", results=[10.0, 20.0, 30.0]),
    ]
    table = _generate_sensitivity_table(runs)
    assert "| 参数 | 取值范围 | 指标 | 指标变化范围 | 敏感性评级 |" in table
    assert "alpha" in table
    assert "MAE" in table
    assert "高" in table          # (30-10)/20 = 100% > 30% → 高


def test_sensitivity_table_medium_rating():
    runs = [
        SensitivityRun(parameter="beta", values=[1, 2, 3],
                       metric="cost", results=[100.0, 115.0, 108.0]),
    ]
    table = _generate_sensitivity_table(runs)
    # (115-100)/107.67 ≈ 13.9% → 中
    assert "中" in table


def test_sensitivity_table_low_rating():
    runs = [
        SensitivityRun(parameter="gamma", values=[1, 2, 3],
                       metric="rate", results=[0.90, 0.91, 0.905]),
    ]
    table = _generate_sensitivity_table(runs)
    # (0.91-0.90)/0.905 ≈ 1.1% → 低
    assert "低" in table


def test_sensitivity_table_empty():
    table = _generate_sensitivity_table([])
    assert table == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_table_assembler.py::test_sensitivity_table_basic -v`
Expected: FAIL with `ImportError: cannot import name '_generate_sensitivity_table'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/math_agent/nodes/table_assembler.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_table_assembler.py -v`
Expected: all 14 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/math_agent/nodes/table_assembler.py tests/test_table_assembler.py
git commit -m "feat(table_assembler): add sensitivity table generator with tests"
```

---

## Task 5: Create `_inject_table` function with tests

**Files:**
- Modify: `src/math_agent/nodes/table_assembler.py`
- Test: `tests/test_table_assembler.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_table_assembler.py`:

```python
from math_agent.nodes.table_assembler import _inject_table


def test_inject_table_appends_when_absent():
    text = "这是原有内容。"
    table = "| A | B |\n|---|---|\n| 1 | 2 |"
    result = _inject_table(text, "参数表", table)
    assert "## 参数表" in result
    assert "这是原有内容。" in result
    assert "| A | B |" in result


def test_inject_table_skips_when_already_present():
    text = "原有内容。\n\n## 参数表\n\n已有表格"
    table = "| A | B |\n|---|---|\n| 1 | 2 |"
    result = _inject_table(text, "参数表", table)
    # 不重复注入
    assert result.count("## 参数表") == 1
    assert "| A | B |" not in result


def test_inject_table_empty_table_returns_unchanged():
    text = "原有内容。"
    result = _inject_table(text, "参数表", "")
    assert result == text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_table_assembler.py::test_inject_table_appends_when_absent -v`
Expected: FAIL with `ImportError: cannot import name '_inject_table'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/math_agent/nodes/table_assembler.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_table_assembler.py -v`
Expected: all 17 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/math_agent/nodes/table_assembler.py tests/test_table_assembler.py
git commit -m "feat(table_assembler): add _inject_table with dedup + tests"
```

---

## Task 6: Create `table_assembler_node` — wire it all together

**Files:**
- Modify: `src/math_agent/nodes/table_assembler.py`
- Test: `tests/test_table_assembler.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_table_assembler.py`:

```python
from math_agent.state import MathModelingState, ModelVersion, PaperSections, SensitivityRun
from math_agent.nodes.table_assembler import table_assembler_node


def _state_for_assembler():
    s = MathModelingState(problem="test")
    s.model_versions.append(ModelVersion(
        stage="final",
        description="final model",
        variables={"x_i": "调度量(件)", "d_i": "需求量", "alpha": "学习率"},
    ))
    s.sensitivity_runs.append(SensitivityRun(
        parameter="alpha", values=[0.1, 0.5, 1.0],
        metric="MAE", results=[10.0, 20.0, 30.0],
    ))
    s.paper = PaperSections(
        model_section="基础预测模型：使用 XGBoost。PaperCritic 给了好评。",
        notation="原有符号表",
        sensitivity="敏感性分析正文",
        solution="求解过程见代码1。Claim: 最优。",
        conclusion="模型优点多。",
    )
    return s


def test_table_assembler_node_injects_variable_table():
    s = _state_for_assembler()
    result = table_assembler_node(s)
    new_paper = result["paper"]
    assert "| 符号 | 含义 | 单位 |" in new_paper.notation
    assert "x_i" in new_paper.notation


def test_table_assembler_node_injects_sensitivity_table():
    s = _state_for_assembler()
    result = table_assembler_node(s)
    new_paper = result["paper"]
    assert "| 参数 | 取值范围 |" in new_paper.sensitivity
    assert "alpha" in new_paper.sensitivity


def test_table_assembler_node_cleans_forbidden_words():
    s = _state_for_assembler()
    result = table_assembler_node(s)
    new_paper = result["paper"]
    assert "PaperCritic" not in new_paper.model_section
    assert "Claim" not in new_paper.solution
    assert "代码1" not in new_paper.solution
    assert len(result["table_warnings"]) >= 3


def test_table_assembler_node_handles_empty_state():
    s = MathModelingState(problem="empty")
    s.paper = PaperSections()
    result = table_assembler_node(s)
    # 不崩，warnings 可能为空
    assert "paper" in result
    assert "table_warnings" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_table_assembler.py::test_table_assembler_node_injects_variable_table -v`
Expected: FAIL with `ImportError: cannot import name 'table_assembler_node'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/math_agent/nodes/table_assembler.py`:

```python
from math_agent.state import MathModelingState, PaperSections


# 要清洗的 section 字段名
_SECTION_FIELDS = [
    "abstract", "problem_restatement", "assumptions", "notation",
    "model_section", "solution", "sensitivity", "conclusion", "references",
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

    # 2) 禁用词清洗（所有 section）
    for field in _SECTION_FIELDS:
        text = getattr(paper, field, "")
        if text:
            cleaned, w = _clean_forbidden_words(text, field)
            setattr(paper, field, cleaned)
            warnings.extend(w)

    return {"paper": paper, "table_warnings": warnings}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_table_assembler.py -v`
Expected: all 21 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/math_agent/nodes/table_assembler.py tests/test_table_assembler.py
git commit -m "feat(table_assembler): wire node — inject tables + clean words"
```

---

## Task 7: Register `table_assembler` in graph

**Files:**
- Modify: `src/math_agent/graph.py:43-86`
- Test: `tests/test_graph_smoke.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_graph_smoke.py`:

```python
def test_graph_has_table_assembler_node():
    """table_assembler 必须在 paper_critic 和 evaluation 之间。"""
    from math_agent.graph import build_graph
    g = build_graph()
    # 构建后的 graph 节点名可通过 .nodes 访问
    assert "table_assembler" in g.nodes
    assert "evaluation" in g.nodes
    assert "paper_critic" in g.nodes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph_smoke.py::test_graph_has_table_assembler_node -v`
Expected: FAIL with `AssertionError: assert 'table_assembler' in {...}`

- [ ] **Step 3: Modify graph.py**

In `src/math_agent/graph.py`:

Add import after L17 (`from math_agent.nodes.latex import latex_node`):

```python
from math_agent.nodes.table_assembler import table_assembler_node
```

After L60 (`g.add_node("latex", _wrap(latex_node, "latex"))`), add:

```python
    g.add_node("table_assembler", _wrap(table_assembler_node, "table_assembler"))
```

Change L78 — the conditional edges mapping. Current:

```python
    g.add_conditional_edges(
        "paper_critic",
        after_paper_critic,
        {"retry": "writer", "advance": "evaluation"},
    )
```

Change `"advance": "evaluation"` to `"advance": "table_assembler"`:

```python
    g.add_conditional_edges(
        "paper_critic",
        after_paper_critic,
        {"retry": "writer", "advance": "table_assembler"},
    )
```

After L80 (`g.add_edge("evaluation", "human_review")`), the flow needs `table_assembler → evaluation`. Add before it:

```python
    g.add_edge("table_assembler", "evaluation")
```

The final edge sequence (L73-82) becomes:

```python
    g.add_edge("figure_pipeline", "writer")
    g.add_edge("writer", "paper_critic")
    g.add_conditional_edges(
        "paper_critic",
        after_paper_critic,
        {"retry": "writer", "advance": "table_assembler"},
    )
    g.add_edge("table_assembler", "evaluation")
    g.add_edge("evaluation", "human_review")
    g.add_edge("human_review", "latex")
    g.add_edge("latex", END)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph_smoke.py::test_graph_has_table_assembler_node -v`
Expected: PASS

- [ ] **Step 5: Run full graph smoke test to check no regression**

Run: `python -m pytest tests/test_graph_smoke.py tests/test_routing.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/math_agent/graph.py tests/test_graph_smoke.py
git commit -m "feat(graph): register table_assembler between paper_critic and evaluation"
```

---

## Task 8: Relax IRON RULE 4 — allow `##`/`###` headings

**Files:**
- Modify: `src/math_agent/templates/writer_iron_rules.md.j2:9-10`
- Test: manual (template change, verified in Task 9-10)

- [ ] **Step 1: Read current rule**

`writer_iron_rules.md.j2` L9-10 currently:

```
   - **不要用 markdown 标题**（`#`、`##`、`###`）。需要小节就用一段加粗文字开头，如 `**basic 阶段**：...`。
```

- [ ] **Step 2: Replace with relaxed rule**

Replace L10 with:

```
   - **章节标题**：`model_section` 和 `solution` 内的子节用 `## xxx` 标题；更深层子节用 `### xxx`。**禁止顶层 `#`**——那会生成与模板 `\section{}` 冲突的 `\section{}`。其他章节（abstract / assumptions / notation / sensitivity / conclusion）不用标题，用加粗段落开头。
```

- [ ] **Step 3: Add forbidden-word rule (RULE 7) after RULE 6**

After the closing of RULE 6 (L22 `章节内**每个子论点**独占一段，段间空行。**绝对不要把 5 条假设挤成一坨连写**。`), add:

```

7. **禁泄露**：正文中不得出现 PaperCritic、Claim、Evidence、Reasoning、issue（作为英文词）、回应、代码[数字]、超时、占位、李华、张三、王五。这些是内部流程痕迹或占位符，评委看到会扣分。
```

- [ ] **Step 4: Verify template renders without Jinja error**

Run:
```bash
python -c "
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
env = Environment(loader=FileSystemLoader('src/math_agent/templates'), autoescape=select_autoescape([]))
t = env.get_template('writer_iron_rules.md.j2')
print(t.render()[:200])
print('OK')
"
```
Expected: prints first 200 chars + `OK`

- [ ] **Step 5: Commit**

```bash
git add src/math_agent/templates/writer_iron_rules.md.j2
git commit -m "feat(writer): relax IRON RULE 4 — allow ##/### headings; add RULE 7 forbidden words"
```

---

## Task 9: Rewrite `writer_section_model.md.j2` for 8-section structure

**Files:**
- Modify: `src/math_agent/templates/writer_section_model.md.j2:67-73`

- [ ] **Step 1: Replace the word-budget table row**

Current L67-73:

```
## 字段字数预算与结构要求

| 字段 | 字数区间 | 强制结构 |
|---|---|---|
| `model_section` | 3000–5000 | 按 basic→improved→final 三小节，每小节 ≥800 字。每小节首句 = "相对上版引入了 X"。每小节须包含：(1) 模型动机与形式化定义，(2) 核心方程推导（含每步动机），(3) 参数估计方法，(4) 与上一版的差异分析。若 model_versions 提供了推导链，须在正文中**完整展开**其动机→陈述→结果的论证脉络，不得省略中间步骤。 |

每个子论点独占一段，段间空行。若 model_versions 提供了推导链，须在正文中体现其动机→陈述→结果的论证脉络。
```

Replace with:

```
## 字段字数预算与结构要求

| 字段 | 字数区间 | 强制结构 |
|---|---|---|
| `model_section` | 3000–5000 | 必须按以下三段组织，每段 ≥800 字，用 `##` 标题分隔。把 model_versions[final] 的推导链按主题分配到三段中，不要按 basic/improved/final 迭代阶段平铺。 |

### 三段结构（用 `##` 标题）

**## 基础预测模型**
- 预测问题的形式化定义（输入/输出/目标函数）
- 特征工程与变量选择依据
- 预测模型选择（XGBoost / 时序模型 / 回归等）与数学定义
- 从 model_versions[final].derivation_steps 取相关推导步骤，完整展开动机→陈述→结果

**## 优化调度模型**
- 从预测结果到调度决策的映射关系
- 目标函数形式化（min 成本 / max 服务率 / 多目标加权）
- 约束条件逐一推导（从题目条件 + variables）
- 模型类型声明（MILP / MINLP / 随机规划）

**## 动态与不确定性扩展**
- 从静态到动态的扩展逻辑（滚动窗口 / 反馈修正）
- 不确定性来源分析（需求波动 / 预测误差）
- 鲁棒性/随机性处理方法（CVaR / 机会约束 / 场景树）
- 扩展模型的数学形式

每段首句点明该段相对于更简单方案的增量价值。每个子论点独占一段，段间空行。若题目无调度需求，将"优化调度"段改为"模型优化与求解策略"；若无不确定性需求，将"动态扩展"段改为"模型泛化性讨论"。
```

- [ ] **Step 2: Verify template renders**

Run:
```bash
python -c "
from jinja2 import Environment, FileSystemLoader, select_autoescape
env = Environment(loader=FileSystemLoader('src/math_agent/templates'), autoescape=select_autoescape([]))
t = env.get_template('writer_section_model.md.j2')
from math_agent.state import MathModelingState, ModelVersion
s = MathModelingState(problem='test')
s.model_versions.append(ModelVersion(stage='final', description='m', variables={'x':'v'}))
print(t.render(problem=s.problem, assumptions=[], model_versions=s.model_versions, code_artifacts=[], prior_critic=None, outline_model_section='anchor', retrieved_context='').count('基础预测模型'))
print('OK')
"
```
Expected: prints `1` (the heading appears once in instructions) + `OK`

- [ ] **Step 3: Commit**

```bash
git add src/math_agent/templates/writer_section_model.md.j2
git commit -m "feat(writer): rewrite model_section for 8-section judge-friendly structure"
```

---

## Task 10: Rewrite `writer_section_solution.md.j2` for 8-section structure

**Files:**
- Modify: `src/math_agent/templates/writer_section_solution.md.j2:50-56`

- [ ] **Step 1: Replace the word-budget table row and CER instruction**

Current L50-56:

```
## 字段字数预算与结构要求

| 字段 | 字数区间 | 强制结构 |
|---|---|---|
| `solution` | 1500–2500 | 须包含：(1) 求解算法总述（算法选择动机、复杂度分析），(2) 分步求解过程（每步含公式代入与数值计算），(3) 结果展示与验证（采用 CER：Claim→Evidence→Reasoning，数字须可追溯），(4) 结果分析与讨论。每个子论点独占一段。 |

**IRON RULE 1（禁编造）铁律**：solution 中的所有数字必须可在上方「代码运行结果」中追溯。无来源的数字一律改为定性表述。
```

Replace with:

```
## 字段字数预算与结构要求

| 字段 | 字数区间 | 强制结构 |
|---|---|---|
| `solution` | 1500–2500 | 必须按以下两段组织，用 `##` 标题分隔。 |

### 两段结构（用 `##` 标题）

**## 求解算法与流程**
- 算法选择动机（对比候选：精确求解器 / 启发式 / 元启发式）
- 算法描述与复杂度分析（用文字描述流程，不输出代码块）
- 关键步骤的公式代入（把通用方程填入题目具体参数）
- 收敛性/最优性保证（若有）

**## 实验设计与对比分析**
- 实验环境（数据规模 / 时间窗口）
- 至少 3 个对照方案的描述与设置理由（无调度 / 简单预测 / 贪婪启发式等）
- 各方案定量结果（从上方「代码运行结果」提取，必须与 stdout 一致）
- 结果分析：每个结论用「结论→依据→分析」结构，依据来自代码 stdout

**IRON RULE 1（禁编造）铁律**：solution 中的所有数字必须可在上方「代码运行结果」中追溯。无来源的数字一律改为定性表述。
```

Note: replaced "CER：Claim→Evidence→Reasoning" with "结论→依据→分析" — this also prevents the forbidden English words from leaking via the prompt.

- [ ] **Step 2: Verify template renders**

Run:
```bash
python -c "
from jinja2 import Environment, FileSystemLoader, select_autoescape
env = Environment(loader=FileSystemLoader('src/math_agent/templates'), autoescape=select_autoescape([]))
t = env.get_template('writer_section_solution.md.j2')
print(t.render(problem='p', model_versions=[], code_artifacts=[], prior_critic=None, outline_solution='a', retrieved_context='').count('求解算法与流程'))
print('OK')
"
```
Expected: `1` + `OK`

- [ ] **Step 3: Commit**

```bash
git add src/math_agent/templates/writer_section_solution.md.j2
git commit -m "feat(writer): rewrite solution for 8-section structure; replace CER with Chinese terms"
```

---

## Task 11: Update `gmcm.tex.j2` section headings + curated appendix

**Files:**
- Modify: `src/math_agent/templates/gmcm.tex.j2:37-41, 76-94`

- [ ] **Step 1: Update section headings**

Current L37-41:

```latex
\section{模型的建立与求解}
{{ paper.model_section }}

\subsection{求解过程与结果}
{{ paper.solution }}
```

The `model_section` now contains `## 基础预测模型` etc. which `_md_headings_to_latex` converts to `\subsection{}`. The hardcoded `\subsection{求解过程与结果}` is redundant — `solution` now has its own `##` headings. Replace L37-41 with:

```latex
\section{模型的建立}
{{ paper.model_section }}

\section{模型的求解与实验}
{{ paper.solution }}
```

- [ ] **Step 2: Update appendix to curated form**

Current L76-94:

```latex
{% if code_artifacts %}
\newpage
\appendix
\section{程序代码}
{% for art in code_artifacts %}
\subsection[代码 {{ loop.index }}]{ {{- art.purpose -}} }
\begin{lstlisting}[language=Python]
{{ art.code }}
\end{lstlisting}

{% if art.stdout %}
\subsubsection*{运行输出}
\begin{verbatim}
{{ art.stdout }}
\end{verbatim}
{% endif %}

{% endfor %}
{% endif %}
```

Replace with curated version (truncated code + key output only):

```latex
{% if code_artifacts %}
\newpage
\appendix
\section{关键算法代码与说明}
{% for art in code_artifacts %}
\subsection{ {{- art.purpose -}} }
\begin{lstlisting}[language=Python]
{{ art.curated_code | default(art.code) }}
\end{lstlisting}

{% if art.curated_stdout %}
\subsubsection*{关键输出摘要}
\begin{verbatim}
{{ art.curated_stdout }}
\end{verbatim}
{% endif %}

{% endfor %}
{% endif %}
```

Note: `curated_code`/`curated_stdout` are prepared in `latex_node` (Task 13). The `| default(art.code)` fallback ensures the template works even before Task 13 lands.

- [ ] **Step 3: Verify template renders**

Run:
```bash
python -c "
from jinja2 import Environment, FileSystemLoader, select_autoescape
env = Environment(loader=FileSystemLoader('src/math_agent/templates'), autoescape=select_autoescape([]))
t = env.get_template('gmcm.tex.j2')
from math_agent.state import PaperSections
out = t.render(problem='test', paper=PaperSections(model_section='## 基础预测模型\n内容'), figures=[], sensitivity_runs=[], code_artifacts=[], keywords='kw', team_id='1', school='S', member_a='A', member_b='B', member_c='C')
assert '关键算法代码' in out
assert '模型的建立' in out
print('OK')
"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/math_agent/templates/gmcm.tex.j2
git commit -m "feat(gmcm): 8-section headings + curated appendix template"
```

---

## Task 12: Update `paper.md.j2` and `paper.tex.j2`

**Files:**
- Modify: `src/math_agent/templates/paper.md.j2:15-29, 36-51`
- Modify: `src/math_agent/templates/paper.tex.j2:31-35`

- [ ] **Step 1: Update paper.md.j2 section headings**

Current L15-29:

```
## 4. 模型的建立与演化
{{ paper.model_section }}

## 5. 模型的求解
{{ paper.solution }}

## 6. 敏感性分析
{{ paper.sensitivity }}
{% for r in sensitivity_runs %}
### {{ r.parameter }}
{{ r.interpretation }}
{% endfor %}

## 7. 模型评价与结论
{{ paper.conclusion }}
```

Replace L15-16 with:

```
## 4. 模型的建立
{{ paper.model_section }}

## 5. 模型的求解与实验
{{ paper.solution }}
```

- [ ] **Step 2: Update paper.md.j2 appendix to curated form**

Current L36-51:

```
## 附录 A. 代码与运行输出
{% for a in code_artifacts %}
### A.{{ loop.index }} {{ a.purpose }}（success={{ a.success }}）
```python
{{ a.code }}
```
**stdout**：
```
{{ a.stdout }}
```
{% if a.stderr %}**stderr**：
```
{{ a.stderr }}
```
{% endif %}
{% endfor %}
```

Replace with:

```
## 附录 A. 关键算法代码
{% for a in code_artifacts %}
{% if a.success %}
### A.{{ loop.index }} {{ a.purpose }}

```python
{{ a.curated_code | default(a.code) }}
```

{% if a.curated_stdout %}
**关键输出摘要**：
```
{{ a.curated_stdout }}
```
{% endif %}
{% endif %}
{% endfor %}
```

- [ ] **Step 3: Add appendix block to paper.tex.j2**

Current `paper.tex.j2` ends at L66 (`\end{document}`) with no appendix. Insert before `\end{document}` (L66):

```latex
{% if code_artifacts %}
\appendix
\section{关键算法代码}
{% for art in code_artifacts %}
{% if art.success %}
\subsection{ {{- art.purpose -}} }
\begin{lstlisting}[language=Python]
{{ art.curated_code | default(art.code) }}
\end{lstlisting}

{% if art.curated_stdout %}
\subsubsection*{关键输出摘要}
\begin{verbatim}
{{ art.curated_stdout }}
\end{verbatim}
{% endif %}

{% endif %}
{% endfor %}
{% endif %}
```

Also update `paper.tex.j2` section headings L31-35:

Current:
```latex
\section{模型的建立与演化}
{{ paper.model_section }}

\section{模型的求解}
{{ paper.solution }}
```

Replace with:
```latex
\section{模型的建立}
{{ paper.model_section }}

\section{模型的求解与实验}
{{ paper.solution }}
```

Note: `paper.tex.j2` currently does not pass `code_artifacts` to the template. This is fixed in Task 13 (latex_node passes it).

- [ ] **Step 4: Verify both templates render**

Run:
```bash
python -c "
from jinja2 import Environment, FileSystemLoader, select_autoescape
env = Environment(loader=FileSystemLoader('src/math_agent/templates'), autoescape=select_autoescape([]))
from math_agent.state import PaperSections
for name in ['paper.md.j2', 'paper.tex.j2']:
    t = env.get_template(name)
    out = t.render(problem='test', paper=PaperSections(), figures=[], sensitivity_runs=[], code_artifacts=[])
    assert '关键算法代码' in out or '附录' in out or 'end{document}' in out
print('OK')
"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/math_agent/templates/paper.md.j2 src/math_agent/templates/paper.tex.j2
git commit -m "feat(templates): 8-section headings + curated appendix for md and default tex"
```

---

## Task 13: Add appendix curation in `latex_node`

**Files:**
- Modify: `src/math_agent/nodes/latex.py:547-628`

- [ ] **Step 1: Read the current latex_node appendix handling**

The `latex_node` (L547-628) builds `render_kwargs`. For gmcm it passes `code_artifacts=[a for a in state.code_artifacts if a.success]` (L604). We need to add `curated_code` and `curated_stdout` to each artifact before passing.

- [ ] **Step 2: Add curation helper functions**

In `src/math_agent/nodes/latex.py`, after `_truncate_caption` (around L545), add:

```python
def _curate_code(code: str, max_lines: int = 80) -> str:
    """截取代码前 max_lines 行（ponytail: 一行 Python，不注册 Jinja2 过滤器）。"""
    lines = code.split("\n")
    if len(lines) <= max_lines:
        return code
    return "\n".join(lines[:max_lines]) + f"\n# ... (共 {len(lines)} 行，截取前 {max_lines} 行)"


def _curate_stdout(stdout: str) -> str:
    """提取 stdout 关键行：RESULT: 行 + 末尾 5 行。"""
    if not stdout:
        return ""
    lines = stdout.splitlines()
    result_lines = [l for l in lines if l.strip().startswith("RESULT:")]
    tail = lines[-5:]
    # 去重保序
    seen = set()
    out = []
    for l in result_lines + tail:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return "\n".join(out)
```

- [ ] **Step 3: Apply curation in latex_node**

In `latex_node`, find the gmcm block (around L604):

```python
        render_kwargs.update(
            school=state.school,
            team_id=state.team_id,
            member_a=mem[0] or None,
            member_b=mem[1] or None,
            member_c=mem[2] or None,
            keywords=(state.paper.keywords or "数学建模").strip(),
            code_artifacts=[a for a in state.code_artifacts if a.success],
        )
```

Replace the `code_artifacts` line with curated versions:

```python
            code_artifacts=[
                type(a)(  # 重建带 curated 字段的副本
                    purpose=a.purpose, code=a.code, stdout=a.stdout,
                    stderr=a.stderr, success=a.success, artifact_paths=a.artifact_paths,
                    curated_code=_curate_code(a.code),
                    curated_stdout=_curate_stdout(a.stdout),
                )
                for a in state.code_artifacts if a.success
            ],
```

Wait — `CodeArtifact` doesn't have `curated_code`/`curated_stdout` fields. We need to add them OR use a simpler approach. **Ponytail**: don't add fields to the state model for rendering-only data. Use a plain dict or a small dataclass local to latex_node.

Simpler approach — use a wrapper dict:

Replace the gmcm `code_artifacts` line with:

```python
            code_artifacts=[
                {
                    "purpose": a.purpose, "code": a.code, "stdout": a.stdout,
                    "success": a.success, "artifact_paths": a.artifact_paths,
                    "curated_code": _curate_code(a.code),
                    "curated_stdout": _curate_stdout(a.stdout),
                }
                for a in state.code_artifacts if a.success
            ],
```

And for the default `paper.tex.j2` path (which currently doesn't pass code_artifacts), add to the base `render_kwargs` (around L589-592):

```python
    render_kwargs = dict(
        problem=_latex_escape(_wrap_unicode_math(title_line)),
        paper=safe_paper, figures=safe_figures, sensitivity_runs=safe_sens,
        code_artifacts=[
            {
                "purpose": a.purpose, "code": a.code, "stdout": a.stdout,
                "success": a.success, "artifact_paths": a.artifact_paths,
                "curated_code": _curate_code(a.code),
                "curated_stdout": _curate_stdout(a.stdout),
            }
            for a in state.code_artifacts if a.success
        ],
    )
```

The templates use `art.curated_code | default(art.code)` — but dict access in Jinja2 uses `.` syntax which works for dicts. However `| default()` only triggers for `Undefined`, not for missing dict keys. Since we always set `curated_code` in the dict, this is fine. Update templates to just `{{ art.curated_code }}` (no default needed since we always provide it). But keep `| default(art.code)` in templates as a safety net for any caller that passes raw `CodeArtifact` objects — Jinja2's `| default` does work for `Undefined` attributes on objects without that attr.

Actually, `CodeArtifact` is a Pydantic model — accessing `art.curated_code` on it raises `AttributeError`, not `Undefined`. So `| default()` won't catch it. The cleanest fix: always pass dicts from `latex_node`. The templates always get dicts. Remove `| default(art.code)` from templates, use `{{ art.curated_code }}` directly.

**Revise Task 11 and 12 templates**: change `{{ art.curated_code | default(art.code) }}` to `{{ art.curated_code }}` and `{{ art.curated_stdout }}` (no default). Since `latex_node` always provides these in the dict, this is safe.

- [ ] **Step 4: Update render_markdown call**

`latex_node` L623 calls `render_markdown(state)` (defined in `nodes/writer.py:108-116`). It passes `code_artifacts=state.code_artifacts` (L113) — raw Pydantic objects, not dicts. The `paper.md.j2` template now accesses `a.curated_code` which doesn't exist on `CodeArtifact`. Fix `render_markdown` to pass curated dicts:

In `src/math_agent/nodes/writer.py`, change `render_markdown` (L108-116) to:

```python
def render_markdown(state: MathModelingState) -> str:
    from math_agent.nodes.latex import _curate_code, _curate_stdout
    tmpl = _env.get_template("paper.md.j2")
    curated = [
        {
            "purpose": a.purpose, "code": a.code, "stdout": a.stdout,
            "success": a.success, "artifact_paths": a.artifact_paths,
            "curated_code": _curate_code(a.code),
            "curated_stdout": _curate_stdout(a.stdout),
        }
        for a in state.code_artifacts if a.success
    ]
    return tmpl.render(
        problem=state.problem,
        paper=state.paper,
        code_artifacts=curated,
        figures=state.figures,
        sensitivity_runs=state.sensitivity_runs,
    )
```

- [ ] **Step 5: Run existing tests**

Run: `python -m pytest tests/test_graph_smoke.py tests/test_table_assembler.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/math_agent/nodes/latex.py src/math_agent/templates/gmcm.tex.j2 src/math_agent/templates/paper.tex.j2 src/math_agent/templates/paper.md.j2 src/math_agent/nodes/writer.py
git commit -m "feat(latex): curate appendix — truncate code + extract key stdout"
```

---

## Task 14: Create e2e integration test

**Files:**
- Create: `scripts/e2e_table_assembler.py`

- [ ] **Step 1: Write the integration script**

Create `scripts/e2e_table_assembler.py`:

```python
"""table_assembler 端到端验证：mock state → run node → verify output。

不依赖 LLM / xelatex。验证：
1) 变量表注入到 notation
2) 敏感性表注入到 sensitivity
3) 禁用词被清洗
4) table_warnings 非空
5) 空 state 不崩
"""
from __future__ import annotations

from math_agent.state import (
    MathModelingState, ModelVersion, PaperSections, SensitivityRun,
)
from math_agent.nodes.table_assembler import table_assembler_node


def main():
    # 构造 mock state
    s = MathModelingState(problem="共享单车调度问题")
    s.model_versions.append(ModelVersion(
        stage="final",
        description="XGBoost + 随机规划滚动优化",
        variables={
            "x_i": "第i时段调度量(辆)",
            "d_i": "第i时段需求量(辆)",
            "s_i": "第i时段初始存量",
            "alpha": "学习率",
            "beta": "惩罚系数",
        },
        equations=["\\min \\sum_i (c_i x_i + p_i (d_i - x_i)^+)"],
    ))
    s.sensitivity_runs = [
        SensitivityRun(parameter="alpha", values=[0.01, 0.1, 0.5],
                       metric="MAE", results=[3.2, 2.4, 2.8]),
        SensitivityRun(parameter="beta", values=[0.5, 1.0, 2.0],
                       metric="total_cost", results=[1200, 1245, 1180]),
    ]
    s.paper = PaperSections(
        abstract="本文研究共享单车调度问题。",
        problem_restatement="问题分析：需要决策变量 x_i。",
        assumptions="**假设1**：需求服从泊松分布。",
        notation="| 符号 | 含义 |\n|---|---|\n| x_i | 调度量 |",
        model_section="## 基础预测模型\n使用 XGBoost。PaperCritic 评分 8 分。",
        solution="## 求解算法与流程\n见代码1。Claim: 成本下降 15%。",
        sensitivity="对参数 alpha 做敏感性分析。",
        conclusion="模型优点：精度高。",
        references="[1] 作者. 标题. 期刊, 2024.",
    )

    result = table_assembler_node(s)
    paper = result["paper"]
    warnings = result["table_warnings"]

    # 1) 变量表
    assert "| 符号 | 含义 | 单位 |" in paper.notation, "变量表未注入 notation"
    assert "x_i" in paper.notation
    print("✓ 变量表注入 notation")

    # 2) 敏感性表
    assert "| 参数 | 取值范围 |" in paper.sensitivity, "敏感性表未注入"
    assert "alpha" in paper.sensitivity
    print("✓ 敏感性表注入 sensitivity")

    # 3) 禁用词清洗
    assert "PaperCritic" not in paper.model_section, "PaperCritic 未清洗"
    assert "Claim" not in paper.solution, "Claim 未清洗"
    assert "代码1" not in paper.solution, "代码1 未清洗"
    print("✓ 禁用词已清洗")

    # 4) warnings
    assert len(warnings) >= 3, f"warnings 不足: {warnings}"
    print(f"✓ table_warnings: {len(warnings)} 条")

    # 5) 空 state
    empty = MathModelingState(problem="empty")
    empty.paper = PaperSections()
    result2 = table_assembler_node(empty)
    assert "paper" in result2
    print("✓ 空 state 不崩")

    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the integration script**

Run: `python scripts/e2e_table_assembler.py`
Expected: prints all `✓` lines + `全部通过。`

- [ ] **Step 3: Commit**

```bash
git add scripts/e2e_table_assembler.py
git commit -m "test: add e2e validation script for table_assembler"
```

---

## Task 15: Run full regression — bench + smoke tests

**Files:** none (verification only)

- [ ] **Step 1: Run all unit tests**

Run: `python -m pytest tests/ -v --tb=short`
Expected: all PASS (existing tests + new table_assembler tests)

- [ ] **Step 2: Run Plan D e2e (mock LLM, real graph)**

Run: `python scripts/e2e_plan_d.py`
Expected: PASS — verifies the new `table_assembler` node doesn't break the graph topology. The mock LLM path may need updating if `e2e_plan_d.py` doesn't mock `table_assembler` — but since `table_assembler` makes **no LLM calls** (pure code), no mock is needed. It will run on whatever state the graph produced.

- [ ] **Step 3: If e2e_plan_d breaks, fix the mock**

If `e2e_plan_d.py` fails because the graph now expects `table_assembler` between `paper_critic` and `evaluation`, the script itself should still work — `table_assembler` is a pure function that doesn't need mocking. If it fails for another reason, debug.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "test: fix e2e regression after table_assembler insertion"
```
(only if fixes were needed)

---

## Task 16: Final commit + branch summary

- [ ] **Step 1: Verify clean working tree**

Run: `git status`
Expected: clean (or only untracked `analysis_pdf/`)

- [ ] **Step 2: Verify all commits are in**

Run: `git log --oneline -12`
Expected: see all Task commits from 1-15

- [ ] **Step 3: Tag Phase 1 completion**

```bash
git tag phase-1-table-assembler-complete
```

---

## Self-Review Notes

**Spec coverage check:**
- Section A (graph topology): Task 7 ✓
- Section B (8-section structure): Tasks 8, 9, 10, 11, 12 ✓
- Section C (2 Phase-1 tables): Tasks 3, 4, 5, 6 ✓
- Section D (forbidden-word cleaner): Tasks 2, 6, 8 ✓
- Section G (curated appendix): Tasks 11, 12, 13 ✓
- Phase 2 items (E, F): explicitly out of scope — separate plan

**Known simplification:** Task 13 uses plain dicts for `code_artifacts` in `latex_node` rather than adding `curated_code`/`curated_stdout` to the `CodeArtifact` Pydantic model. This avoids polluting the state schema with rendering-only fields. If other nodes ever need curated code, revisit.

**Risk:** Task 13's interaction with `render_markdown` (in `writer.py`) needs verification — the function also renders `paper.md.j2` with `code_artifacts`. If it passes raw `CodeArtifact` objects (not dicts), the template's `art.curated_code` access will fail. The plan flags this in Task 13 Step 4.
