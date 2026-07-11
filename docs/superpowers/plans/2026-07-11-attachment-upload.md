# 附件上传功能 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让用户上传 PDF/Word 题面和 xlsx/csv 附件，系统提取摘要注入 analyst prompt，传递文件路径给 coder/sensitivity 直接读取真实数据。

**Architecture:** 前端新增附件上传区 -> 服务端 multipart 上传 API + Python 摘要脚本 -> CLI 读取 data_files/data_dir 注入 state -> analyst prompt 注入摘要 / coder/sensitivity prompt 注入绝对路径。

**Tech Stack:** Node.js (零依赖 multipart parser) / Python (pypdf, python-docx, openpyxl, pandas) / 现有 pydantic state + typer CLI

**Spec:** `docs/superpowers/specs/2026-07-11-attachment-upload-design.md`

---

## File Structure

| 文件 | 操作 | 职责 |
|------|------|------|
| `pyproject.toml` | 修改 | 新增 python-docx, openpyxl 依赖 |
| `scripts/extract_file_meta.py` | 新建 | 对上传文件生成摘要 JSON |
| `src/math_agent/state.py` | 修改 | 新增 DataFileSheet / DataFileInfo 模型 + state 字段 |
| `src/math_agent/cli.py` | 修改 | `_read_problem_spec` 读取 data_files/data_dir |
| `src/math_agent/prompts/_data_hint.py` | 新建 | coder/sensitivity 共用的数据路径提示 helper |
| `src/math_agent/prompts/analyst.py` | 修改 | `build_prompt` 注入附件摘要 |
| `src/math_agent/nodes/analyst.py` | 修改 | 传入 state.data_files |
| `src/math_agent/prompts/coder.py` | 修改 | SYSTEM 微调文件读取约束 |
| `src/math_agent/prompts/coder_figure_one.py` | 修改 | `build_prompt_figure_one` 注入数据路径 |
| `src/math_agent/nodes/coder.py` | 修改 | 传入 state.data_dir/data_files |
| `src/math_agent/prompts/sensitivity.py` | 修改 | `build_code_prompt` 注入数据路径 |
| `src/math_agent/nodes/sensitivity.py` | 修改 | 传入 state.data_dir/data_files |
| `frontend/server.mjs` | 修改 | 新增 multipart parser + `/api/upload` + `/api/run` 附件复制 |
| `frontend/index.html` | 修改 | 新增附件上传区 + 扩展题面 accept |
| `frontend/app.js` | 修改 | 附件上传/移除/展示 + PDF 提取 + run 时发送 attachments |
| `frontend/styles.css` | 修改 | 附件区样式 |

---

### Task 1: 新增 Python 依赖

**Files:**
- Modify: `pyproject.toml:6-23`

- [ ] **Step 1: 添加 openpyxl 和 python-docx 到 dependencies**

在 `pyproject.toml` 的 dependencies 数组末尾（`"requests>=2.31",` 之后）追加：

```toml
    "openpyxl>=3.1",
    "python-docx>=1.1",
```

- [ ] **Step 2: 安装新依赖**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && uv sync`
Expected: 成功安装 openpyxl 和 python-docx

- [ ] **Step 3: 验证导入可用**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && uv run python -c "import openpyxl, docx, pypdf; print('ok')"`
Expected: 输出 `ok`

- [ ] **Step 4: Commit**

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
git add pyproject.toml uv.lock
git commit -m "deps: add openpyxl + python-docx for attachment parsing"
```

---

### Task 2: 摘要提取脚本 extract_file_meta.py

**Files:**
- Create: `scripts/extract_file_meta.py`

- [ ] **Step 1: 编写脚本**

创建 `scripts/extract_file_meta.py`：

```python
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
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    raw = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    # ponytail: lone-surrogate 清洗，复用 rag/ingest.py 同款逻辑
    text = raw.encode("utf-8", errors="ignore").decode("utf-8")
    return {"text_excerpt": text[:5000], "total_pages": len(reader.pages)}


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
```

- [ ] **Step 2: 创建测试 CSV 并验证脚本**

Run:
```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
echo "name,value,city" > /tmp/test_data.csv
echo "Alice,30,Beijing" >> /tmp/test_data.csv
echo "Bob,25,Shanghai" >> /tmp/test_data.csv
uv run python scripts/extract_file_meta.py /tmp/test_data.csv
```
Expected: JSON 输出含 `"file_type": "csv"`, `"rows": 3`, `"columns": ["name", "value", "city"]`

- [ ] **Step 3: 用华中杯 xlsx 附件验证（如果可用）**

Run:
```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
ls "C:/Users/lwh86/Downloads/A题：城市绿色物流配送调度_1776993150913/附件/" 2>/dev/null && uv run python scripts/extract_file_meta.py "C:/Users/lwh86/Downloads/A题：城市绿色物流配送调度_1776993150913/附件/订单信息.xlsx" || echo "skip xlsx test (file not found)"
```
Expected: 如果文件存在，输出 JSON 含 sheets 数组和列名；否则跳过

- [ ] **Step 4: Commit**

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
git add scripts/extract_file_meta.py
git commit -m "feat: add extract_file_meta.py for attachment summary extraction"
```

---

### Task 3: State 新增数据文件模型

**Files:**
- Modify: `src/math_agent/state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_state.py` 末尾追加：

```python
from math_agent.state import DataFileInfo, DataFileSheet


def test_state_accepts_data_files():
    s = MathModelingState(
        problem="x",
        data_dir="/tmp/data",
        data_files=[DataFileInfo(filename="orders.xlsx", file_type="xlsx", path="orders.xlsx")],
    )
    assert s.data_dir == "/tmp/data"
    assert len(s.data_files) == 1
    assert s.data_files[0].filename == "orders.xlsx"


def test_state_defaults_no_data_files():
    s = MathModelingState(problem="x")
    assert s.data_dir is None
    assert s.data_files == []


def test_data_file_sheet_model():
    sheet = DataFileSheet(name="Sheet1", rows=100, cols=3, columns=["a", "b", "c"])
    assert sheet.rows == 100
    assert sheet.preview == []
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && uv run pytest tests/test_state.py::test_state_accepts_data_files tests/test_state.py::test_state_defaults_no_data_files tests/test_state.py::test_data_file_sheet_model -v`
Expected: FAIL with `ImportError: cannot import name 'DataFileInfo'`

- [ ] **Step 3: 添加模型和字段**

在 `src/math_agent/state.py` 中，在 `class MetricSpec` 定义之后、`class ModelingCandidate` 之前插入：

```python
class DataFileSheet(BaseModel):
    """xlsx/csv 的单个 sheet/表摘要。"""
    name: str
    rows: int = 0
    cols: int = 0
    columns: list[str] = Field(default_factory=list)
    preview: list[list[str]] = Field(default_factory=list)


class DataFileInfo(BaseModel):
    """上传附件的元信息，供 analyst/coder 理解数据。"""
    filename: str
    file_type: str                           # "xlsx"|"csv"|"pdf"|"docx"|"txt"|"md"
    path: str                                # 相对 data_dir 的文件名
    summary: dict = Field(default_factory=dict)  # 灵活 dict，不同类型结构不同
```

然后在 `MathModelingState` 类中，在 `output_dir: Optional[str] = None` 之前插入：

```python
    # 附件数据（覆盖语义；run 启动时确定，中途不变）
    data_dir: Optional[str] = None
    data_files: list[DataFileInfo] = Field(default_factory=list)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && uv run pytest tests/test_state.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
git add src/math_agent/state.py tests/test_state.py
git commit -m "feat(state): add DataFileInfo/DataFileSheet models + data_dir/data_files fields"
```

---

### Task 4: CLI 读取 data_files/data_dir

**Files:**
- Modify: `src/math_agent/cli.py:192-263`
- Test: `tests/test_cli.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_cli.py` 末尾追加：

```python
def test_run_passes_data_files_to_initial_state(tmp_path):
    problem = tmp_path / "problem.json"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "orders.xlsx").write_bytes(b"fake")
    problem.write_text(json.dumps({
        "title": "t",
        "questions": ["q"],
        "data_dir": str(data_dir),
        "data_files": [
            {"filename": "orders.xlsx", "file_type": "xlsx", "path": "orders.xlsx",
             "summary": {"sheets": [{"name": "Sheet1", "rows": 10, "cols": 3}]}}
        ],
    }), encoding="utf-8")
    out = tmp_path / "run"
    out.mkdir()

    fake_graph = MagicMock()
    fake_graph.get_state.return_value = MagicMock(values={"problem": "p"})
    saver_cm = MagicMock()
    saver_cm.__enter__.return_value = object()
    saver_cm.__exit__.return_value = False
    with patch("math_agent.cli._saver_cm", return_value=saver_cm), \
         patch("math_agent.cli.build_graph", return_value=fake_graph), \
         patch("math_agent.cli._dump_state_summary"):
        result = runner.invoke(app, [
            "run", "--problem", str(problem), "--out", str(out), "--no-interrupt",
        ])

    assert result.exit_code == 0, result.output
    initial = fake_graph.invoke.call_args.args[0]
    assert initial["data_dir"] == str(data_dir)
    assert len(initial["data_files"]) == 1
    assert initial["data_files"][0].filename == "orders.xlsx"


def test_run_rejects_nonexistent_data_dir(tmp_path):
    problem = tmp_path / "problem.json"
    problem.write_text(json.dumps({
        "title": "t", "questions": ["q"],
        "data_dir": str(tmp_path / "nonexistent"),
    }), encoding="utf-8")
    out = tmp_path / "run"
    out.mkdir()
    result = runner.invoke(app, ["run", "--problem", str(problem), "--out", str(out), "--no-interrupt"])
    assert result.exit_code != 0
    assert "data_dir" in result.output
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && uv run pytest tests/test_cli.py::test_run_passes_data_files_to_initial_state tests/test_cli.py::test_run_rejects_nonexistent_data_dir -v`
Expected: FAIL

- [ ] **Step 3: 修改 `_read_problem_spec`**

在 `src/math_agent/cli.py` 的 `_read_problem_spec` 函数中，在 `return {"title": title, "background": background, "questions": questions}` 之前插入附件读取逻辑：

```python
    data_files = spec.get("data_files", [])
    data_dir = spec.get("data_dir", "")
    if not isinstance(data_files, list):
        raise typer.BadParameter("data_files 必须是数组", param_hint="--problem")
    if not isinstance(data_dir, str):
        raise typer.BadParameter("data_dir 必须是字符串", param_hint="--problem")
    if data_dir:
        data_dir_path = Path(data_dir)
        if not data_dir_path.is_absolute():
            data_dir_path = problem.parent / data_dir_path
        if not data_dir_path.is_dir():
            raise typer.BadParameter(
                f"data_dir 不存在: {data_dir_path}", param_hint="--problem"
            )
        data_dir = str(data_dir_path.resolve())

    return {"title": title, "background": background, "questions": questions,
            "data_files": data_files, "data_dir": data_dir}
```

- [ ] **Step 4: 修改 `run` 命令的 `initial` dict**

在 `src/math_agent/cli.py` 的 `run` 函数中，找到 `initial = {` 字典，在 `"output_dir": str(out),` 之后添加：

```python
        "data_dir": spec.get("data_dir") or None,
        "data_files": [DataFileInfo(**f) for f in spec.get("data_files", [])],
```

在 `cli.py` 文件顶部的 import 区域，确保 `DataFileInfo` 已导入。找到现有的 `from math_agent.state import` 行，追加 `DataFileInfo`。

- [ ] **Step 5: 运行测试验证通过**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && uv run pytest tests/test_cli.py -v`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
git add src/math_agent/cli.py tests/test_cli.py
git commit -m "feat(cli): read data_files/data_dir from problem.json into initial state"
```

---

### Task 5: 共享数据路径提示 helper

**Files:**
- Create: `src/math_agent/prompts/_data_hint.py`

- [ ] **Step 1: 编写 helper**

创建 `src/math_agent/prompts/_data_hint.py`：

```python
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
```

- [ ] **Step 2: 自检**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && uv run python -c "from math_agent.prompts._data_hint import build_data_hint, build_data_summary_hint; print(build_data_hint(None, [])); print(build_data_summary_hint([])); print('ok')"`
Expected: 两个空字符串 + `ok`

- [ ] **Step 3: Commit**

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
git add src/math_agent/prompts/_data_hint.py
git commit -m "feat(prompts): add shared data hint helpers for coder/sensitivity/analyst"
```

---

### Task 6: Analyst prompt 注入附件摘要

**Files:**
- Modify: `src/math_agent/prompts/analyst.py:54-69`
- Modify: `src/math_agent/nodes/analyst.py:29-30`
- Test: `tests/nodes/test_analyst.py`

- [ ] **Step 1: 写失败测试**

在 `tests/nodes/test_analyst.py` 末尾追加：

```python
def test_analyst_receives_data_files_in_prompt(mocker):
    from math_agent.state import MathModelingState, DataFileInfo
    from math_agent.nodes.analyst import analyst_node
    from math_agent.prompts.analyst import build_prompt

    state = MathModelingState(
        problem="城市绿色物流配送",
        background="",
        questions=["建立配送调度模型"],
        data_dir="/tmp/data",
        data_files=[DataFileInfo(
            filename="orders.xlsx", file_type="xlsx", path="orders.xlsx",
            summary={"sheets": [{"name": "Sheet1", "rows": 1200, "cols": 5,
                                  "columns": ["订单号", "客户ID", "需求量"]}]}  # noqa: E501
        )],
    )

    captured_prompt = {}
    def fake_complete(prompt, **kwargs):
        captured_prompt["prompt"] = prompt
        return mocker.MagicMock()
    mocker.patch("math_agent.nodes.analyst.complete", side_effect=fake_complete)

    analyst_node(state)
    assert "订单信息" not in captured_prompt["prompt"]  # filename is orders.xlsx
    assert "orders.xlsx" in captured_prompt["prompt"]
    assert "1200行×5列" in captured_prompt["prompt"]
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && uv run pytest tests/nodes/test_analyst.py::test_analyst_receives_data_files_in_prompt -v`
Expected: FAIL（prompt 中不包含 data_files 内容）

- [ ] **Step 3: 修改 `build_prompt`**

在 `src/math_agent/prompts/analyst.py` 的 `build_prompt` 函数签名添加 `data_files=None` 参数，并在 prompt 中注入摘要。修改后的完整函数：

```python
def build_prompt(problem: str, background: str, questions: list[str],
                 retrieved_context: str = "",
                 critic_feedback=None,
                 data_files=None) -> str:
    qs = "\n".join(f"- {q}" for q in questions) or "（题目本身未列出独立小问）"
    ctx = f"\n{retrieved_context}\n\n" if retrieved_context else ""
    fb = ""
    if critic_feedback is not None:
        issues = "\n".join(f"- {i.problem}" for i in critic_feedback.issues)
        sugs = "\n".join(f"- {s}" for s in critic_feedback.suggestions)
        fb = f"\n# 上一轮 Blueprint Critic 反馈\n问题：\n{issues}\n建议：\n{sugs}\n请据此修正蓝图。\n"
    data_hint = ""
    if data_files:
        from math_agent.prompts._data_hint import build_data_summary_hint
        data_hint = build_data_summary_hint(data_files)
    return (
        f"# 题目\n{problem}\n\n"
        f"# 背景\n{background or '（无）'}\n\n"
        f"# 小问\n{qs}\n\n"
        f"{ctx}{data_hint}{fb}{_SCHEMA_HINT}"
    )
```

- [ ] **Step 4: 修改 `analyst_node`**

在 `src/math_agent/nodes/analyst.py` 的 `analyst_node` 中，修改 `build_prompt` 调用，传入 `data_files=state.data_files`：

```python
    prompt = build_prompt(state.problem, state.background, state.questions,
                          retrieved_context=ctx, critic_feedback=critic_fb,
                          data_files=state.data_files)
```

- [ ] **Step 5: 运行测试验证通过**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && uv run pytest tests/nodes/test_analyst.py -v`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
git add src/math_agent/prompts/analyst.py src/math_agent/nodes/analyst.py tests/nodes/test_analyst.py
git commit -m "feat(analyst): inject data file summaries into analyst prompt"
```

---

### Task 7: Coder prompt 注入数据路径

**Files:**
- Modify: `src/math_agent/prompts/coder.py:5` (SYSTEM 微调)
- Modify: `src/math_agent/prompts/coder_figure_one.py:22-59`
- Modify: `src/math_agent/nodes/coder.py:41-47`
- Test: `tests/test_coder_baseline.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_coder_baseline.py` 末尾追加：

```python
def test_coder_prompt_includes_data_file_paths():
    from math_agent.state import ModelVersion, DataFileInfo
    from math_agent.prompts.coder_figure_one import build_prompt_figure_one

    model = ModelVersion(
        stage="basic", description="VRP model",
        equations=["min total_cost"], variables={"x": "binary route"},
    )
    data_files = [DataFileInfo(
        filename="orders.xlsx", file_type="xlsx", path="orders.xlsx",
        summary={"sheets": [{"name": "Sheet1", "rows": 100, "cols": 3}]}
    )]
    prompt = build_prompt_figure_one(
        model, "plot cost chart", data_dir="/data/run1", data_files=data_files,
    )
    assert "/data/run1" in prompt
    assert "orders.xlsx" in prompt
    assert "pd.read_excel" in prompt


def test_coder_prompt_no_data_hint_when_empty():
    from math_agent.state import ModelVersion
    from math_agent.prompts.coder_figure_one import build_prompt_figure_one

    model = ModelVersion(
        stage="basic", description="model", equations=[], variables={},
    )
    prompt = build_prompt_figure_one(model, "plot chart", data_dir=None, data_files=[])
    assert "可用数据文件" not in prompt
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && uv run pytest tests/test_coder_baseline.py::test_coder_prompt_includes_data_file_paths tests/test_coder_baseline.py::test_coder_prompt_no_data_hint_when_empty -v`
Expected: FAIL

- [ ] **Step 3: 微调 SYSTEM prompt**

在 `src/math_agent/prompts/coder.py` 的 SYSTEM 字符串中，将：

```
"不联网；不读取本地未声明的文件；"
```

改为：

```
"不联网；只读取 prompt 中明确给出路径的数据文件，不读取其他本地文件；"
```

- [ ] **Step 4: 修改 `build_prompt_figure_one`**

在 `src/math_agent/prompts/coder_figure_one.py` 的 `build_prompt_figure_one` 函数签名添加 `data_dir=None, data_files=None`，并注入提示。修改后的函数签名和返回部分：

```python
def build_prompt_figure_one(model, purpose: str, prev_failure=None, prev_error_kind: str = "",
                            blueprint=None, data_dir=None, data_files=None):
```

在 `metrics_hint = _blueprint_metrics_hint(blueprint)` 之后添加：

```python
    data_hint = ""
    if data_dir and data_files:
        from math_agent.prompts._data_hint import build_data_hint
        data_hint = build_data_hint(data_dir, data_files)
```

在 return 字符串中，将 `{metrics_hint}{fb}` 改为 `{metrics_hint}{data_hint}{fb}`：

```python
    return (
        f"# 模型描述\n{model.description}\n\n# 方程\n{eqs}\n\n# 变量\n{vars_}\n\n"
        f"# 当前绘图任务\n{purpose}\n{metrics_hint}{data_hint}{fb}\n\n"
        f"请为上述绘图任务生成一段**独立可运行**的 Python 脚本。\n"
        f"脚本末尾必须用 print 输出关键指标，格式严格如下：\n"
        f"print(f'RESULT: baseline=ours total_cost={{total_cost}} service_rate={{service_rate}}')\n"
        f"（指标名按题目调整，但必须以 RESULT: baseline=ours 开头，供对比表使用）\n"
        f"stdout 不允许只输出自然语言总结，必须包含 RESULT: 行带具体数值。\n\n"
        f"请输出 JSON：{{\"purpose\": str, \"code\": str}}，code 字段是完整的 Python 源码。"
    )
```

- [ ] **Step 5: 修改 `coder_node` 传参**

在 `src/math_agent/nodes/coder.py` 的 `coder_node` 中，修改 `build_prompt_figure_one` 调用（主图代码和 baseline 代码两处），添加 `data_dir=state.data_dir, data_files=state.data_files`：

主图调用（约第 41 行）：
```python
            draft: CoderDraft = complete(
                build_prompt_figure_one(model, purpose, prev_err, prev_kind,
                                        blueprint=state.problem_blueprint,
                                        data_dir=state.data_dir,
                                        data_files=state.data_files),
                schema=CoderDraft,
                system=SYSTEM,
                model=MODEL_ROUTING["coder"],
            )
```

baseline 调用（约第 84 行）也加上同样的 `data_dir` 和 `data_files` 参数。

- [ ] **Step 6: 运行测试验证通过**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && uv run pytest tests/test_coder_baseline.py -v`
Expected: 全部 PASS

- [ ] **Step 7: Commit**

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
git add src/math_agent/prompts/coder.py src/math_agent/prompts/coder_figure_one.py src/math_agent/nodes/coder.py tests/test_coder_baseline.py
git commit -m "feat(coder): inject data file paths into coder prompt + relax SYSTEM file constraint"
```

---

### Task 8: Sensitivity prompt 注入数据路径

**Files:**
- Modify: `src/math_agent/prompts/sensitivity.py` (`build_code_prompt`)
- Modify: `src/math_agent/nodes/sensitivity.py` (调用处)
- Test: `tests/nodes/test_sensitivity.py`

- [ ] **Step 1: 写失败测试**

在 `tests/nodes/test_sensitivity.py` 末尾追加：

```python
def test_sensitivity_code_prompt_includes_data_paths():
    from math_agent.state import ModelVersion, DataFileInfo
    from math_agent.prompts.sensitivity import build_code_prompt

    model = ModelVersion(
        stage="improved", description="VRP with time windows",
        equations=["min total_cost"], variables={},
    )
    plan_runs = [{"parameter": "alpha", "values": [0.1, 0.3, 0.5], "metric": "total_cost"}]
    data_files = [DataFileInfo(
        filename="distances.xlsx", file_type="xlsx", path="distances.xlsx",
        summary={},
    )]
    prompt = build_code_prompt(model, plan_runs, data_dir="/data/run1", data_files=data_files)
    assert "/data/run1" in prompt
    assert "distances.xlsx" in prompt
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && uv run pytest tests/nodes/test_sensitivity.py::test_sensitivity_code_prompt_includes_data_paths -v`
Expected: FAIL

- [ ] **Step 3: 修改 `build_code_prompt`**

在 `src/math_agent/prompts/sensitivity.py` 的 `build_code_prompt` 函数签名添加 `data_dir=None, data_files=None`，并在返回中注入 `data_hint`。找到函数定义，修改签名：

```python
def build_code_prompt(model, plan_runs, prev_failure: str | None = None,
                      prev_error_kind: str = "",
                      data_dir=None, data_files=None):
```

在 `fb = ...` 之后、`return` 之前添加：

```python
    data_hint = ""
    if data_dir and data_files:
        from math_agent.prompts._data_hint import build_data_hint
        data_hint = build_data_hint(data_dir, data_files)
```

将 return 字符串中的 `{fb}` 改为 `{data_hint}{fb}`：

```python
    return (
        f"# 最终模型\n{model.description}\n方程：\n{chr(10).join(model.equations)}\n\n"
        f"# 敏感性分析计划\n{desc}\n{data_hint}{fb}\n"
        f"请输出 JSON：{{\"code\": str}}。"
    )
```

- [ ] **Step 4: 修改 `sensitivity_node` 传参**

在 `src/math_agent/nodes/sensitivity.py` 中，找到调用 `build_code_prompt` 的地方（在 sensitivity CODE 阶段），添加 `data_dir=state.data_dir, data_files=state.data_files` 参数。

- [ ] **Step 5: 运行测试验证通过**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && uv run pytest tests/nodes/test_sensitivity.py -v`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
git add src/math_agent/prompts/sensitivity.py src/math_agent/nodes/sensitivity.py tests/nodes/test_sensitivity.py
git commit -m "feat(sensitivity): inject data file paths into sensitivity code prompt"
```

---

### Task 9: 服务端 multipart parser + `/api/upload`

**Files:**
- Modify: `frontend/server.mjs`
- Test: `frontend/server.test.mjs`

- [ ] **Step 1: 写失败测试**

在 `frontend/server.test.mjs` 末尾追加：

```javascript
test("POST /api/upload 接受 CSV 附件并返回摘要", async () => {
  const boundary = "----testboundary12345";
  const csvContent = "name,value\nAlice,30\nBob,25\n";
  const body = [
    `--${boundary}\r\n`,
    `Content-Disposition: form-data; name="purpose"\r\n\r\n`,
    `attachment\r\n`,
    `--${boundary}\r\n`,
    `Content-Disposition: form-data; name="file"; filename="test.csv"\r\n`,
    `Content-Type: text/csv\r\n\r\n`,
    csvContent,
    `\r\n--${boundary}--\r\n`,
  ].join("");

  const response = await fetch(`${base}/api/upload`, {
    method: "POST",
    headers: { "Content-Type": `multipart/form-data; boundary=${boundary}` },
    body,
  });
  assert.equal(response.status, 200);
  const data = await response.json();
  assert.equal(data.filename, "test.csv");
  assert.equal(data.fileType, "csv");
  assert.ok(data.summary);
  assert.ok(data.summary.sheets || data.summary.text_excerpt);
  assert.ok(data.id);
});

test("POST /api/upload 拒绝不支持的文件类型", async () => {
  const boundary = "----testboundary99999";
  const body = [
    `--${boundary}\r\n`,
    `Content-Disposition: form-data; name="purpose"\r\n\r\n`,
    `attachment\r\n`,
    `--${boundary}\r\n`,
    `Content-Disposition: form-data; name="file"; filename="evil.exe"\r\n`,
    `Content-Type: application/octet-stream\r\n\r\n`,
    `binarydata`,
    `\r\n--${boundary}--\r\n`,
  ].join("");

  const response = await fetch(`${base}/api/upload`, {
    method: "POST",
    headers: { "Content-Type": `multipart/form-data; boundary=${boundary}` },
    body,
  });
  assert.equal(response.status, 415);
});
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && node --test frontend/server.test.mjs`
Expected: FAIL（`/api/upload` 返回 404）

- [ ] **Step 3: 添加 multipart parser 和 `/api/upload` 路由**

在 `frontend/server.mjs` 中，在 `handleApi` 函数开头（在 `GET /api/health` 路由之前）添加 `/api/upload` 路由。先在文件顶部（import 之后）添加辅助函数：

```javascript
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
const MAX_UPLOAD_BODY = 200 * 1024 * 1024; // 200MB
const ACCEPTED_SUFFIXES = new Set([".json", ".md", ".txt", ".pdf", ".docx", ".xlsx", ".xls", ".csv"]);

function parseMultipart(buffer, boundary) {
  const parts = [];
  const boundaryBytes = Buffer.from(`--${boundary}`);
  let start = 0;
  while (true) {
    const bStart = buffer.indexOf(boundaryBytes, start);
    if (bStart === -1) break;
    const nextStart = bStart + boundaryBytes.length;
    const nextBoundary = buffer.indexOf(boundaryBytes, nextStart);
    if (nextBoundary === -1) break;
    const partData = buffer.slice(nextStart + 2, nextBoundary - 2); // skip \r\n, trim trailing \r\n
    const headerEnd = partData.indexOf("\r\n\r\n");
    if (headerEnd === -1) break;
    const headerStr = partData.slice(0, headerEnd).toString("utf8");
    const content = partData.slice(headerEnd + 4);
    const nameMatch = headerStr.match(/name="([^"]*)"/);
    const filenameMatch = headerStr.match(/filename="([^"]*)"/);
    parts.push({
      name: nameMatch ? nameMatch[1] : "",
      filename: filenameMatch ? filenameMatch[1] : null,
      content: filenameMatch ? content : content.toString("utf8").trim(),
    });
    start = nextBoundary;
  }
  return parts;
}

async function generateFileMeta(filePath) {
  const python = env.PYTHON || "python";
  const py = spawn(python, ["scripts/extract_file_meta.py", filePath], {
    cwd: projectRoot,
    windowsHide: true,
  });
  let stdout = "", stderr = "";
  py.stdout.on("data", (c) => { stdout += c; });
  py.stderr.on("data", (c) => { stderr += c; });
  return new Promise((resolve, reject) => {
    py.on("close", (code) => {
      if (code === 0) {
        try { resolve(JSON.parse(stdout.trim())); }
        catch { reject(new Error("meta script returned invalid JSON")); }
      } else {
        reject(new Error(stderr || "meta script failed"));
      }
    });
    py.on("error", reject);
  });
}
```

然后在 `handleApi` 函数中，在 `if (request.method === "GET" && url.pathname === "/api/health")` 之前添加：

```javascript
  if (request.method === "POST" && url.pathname === "/api/upload") {
    const contentType = request.headers["content-type"] || "";
    const boundaryMatch = contentType.match(/boundary=(.+)/);
    if (!boundaryMatch) {
      sendJson(response, 400, { error: "Missing multipart boundary." });
      return;
    }
    const chunks = [];
    let size = 0;
    for await (const chunk of request) {
      size += chunk.length;
      if (size > MAX_UPLOAD_BODY) {
        sendJson(response, 413, { error: "Upload body exceeds 200MB." });
        return;
      }
      chunks.push(chunk);
    }
    const buffer = Buffer.concat(chunks);
    const parts = parseMultipart(buffer, boundaryMatch[1]);
    const filePart = parts.find((p) => p.filename);
    const purposePart = parts.find((p) => p.name === "purpose");
    if (!filePart) {
      sendJson(response, 400, { error: "No file in upload." });
      return;
    }
    const ext = extname(filePart.filename).toLowerCase();
    if (!ACCEPTED_SUFFIXES.has(ext)) {
      sendJson(response, 415, { error: `Unsupported file type: ${ext}` });
      return;
    }
    if (filePart.content.length > MAX_FILE_SIZE) {
      sendJson(response, 413, { error: "File exceeds 50MB." });
      return;
    }
    const purpose = purposePart ? purposePart.content : "attachment";
    const uploadId = `att-${Date.now()}-${randomUUID().slice(0, 8)}`;
    const uploadDir = safeProjectPath(`runs/ui-server/uploads/${uploadId}`);
    await mkdir(uploadDir, { recursive: true });
    const filePath = resolve(uploadDir, filePart.filename);
    await writeFile(filePath, filePart.content);

    let meta;
    try {
      meta = await generateFileMeta(filePath);
    } catch (e) {
      sendJson(response, 500, { error: `File meta extraction failed: ${e.message}` });
      return;
    }
    const result = {
      id: uploadId,
      filename: filePart.filename,
      fileType: meta.file_type,
      size: filePart.content.length,
      storedPath: relative(projectRoot, filePath).replace(/\\/g, "/"),
      summary: meta.summary,
      text: "",
    };
    if (purpose === "problem" && meta.summary && meta.summary.text_excerpt) {
      result.text = meta.summary.text_excerpt;
    }
    sendJson(response, 200, result);
    return;
  }
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && node --test frontend/server.test.mjs`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
git add frontend/server.mjs frontend/server.test.mjs
git commit -m "feat(server): add POST /api/upload with multipart parser + file meta extraction"
```

---

### Task 10: 服务端 `/api/run` 处理附件复制

**Files:**
- Modify: `frontend/server.mjs` (`POST /api/run` handler, 约 343-461 行)
- Test: `frontend/server.test.mjs`

- [ ] **Step 1: 写失败测试**

在 `frontend/server.test.mjs` 末尾追加：

```javascript
test("POST /api/run 接受 attachments 并复制到 data 目录", async () => {
  // 先上传一个文件
  const boundary = "----testboundary777";
  const csvContent = "a,b\n1,2\n";
  const uploadBody = [
    `--${boundary}\r\n`,
    `Content-Disposition: form-data; name="purpose"\r\n\r\n`,
    `attachment\r\n`,
    `--${boundary}\r\n`,
    `Content-Disposition: form-data; name="file"; filename="data.csv"\r\n\r\n`,
    csvContent,
    `\r\n--${boundary}--\r\n`,
  ].join("");
  const uploadRes = await fetch(`${base}/api/upload`, {
    method: "POST",
    headers: { "Content-Type": `multipart/form-data; boundary=${boundary}` },
    body: uploadBody,
  });
  const uploadData = await uploadRes.json();

  // 启动 run（用 mock，不真正跑 pipeline）
  const runRes = await fetch(`${base}/api/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title: "test",
      background: "test bg",
      outputDir: "runs/ui-test-attachments",
      threadId: "test-att",
      noInterrupt: true,
      ragEnabled: false,
      attachments: [uploadData],
    }),
  });
  assert.equal(runRes.status, 202);
  const runJson = await runRes.json();
  assert.ok(runJson.run.id);

  // 验证 data 目录中文件存在
  const artifactsRes = await fetch(`${base}/api/artifacts?out=runs/ui-test-attachments`);
  // 清理：stop the run
  await fetch(`${base}/api/runs/${runJson.run.id}/stop`, { method: "POST" });
});
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && node --test frontend/server.test.mjs`
Expected: FAIL（run 的 problem.json 中没有 data_files）

- [ ] **Step 3: 修改 `/api/run` handler**

在 `frontend/server.mjs` 的 `POST /api/run` handler 中，在写 `problem.json` 之前添加附件复制逻辑。找到这段代码：

```javascript
    const problemPath = resolve(runDir, "problem.json");
    const problem = {
      title: body.title || "Beacon UI Problem",
      background: body.background || "",
      questions: ...
    };
```

替换为：

```javascript
    // 附件复制到 run 专属 data 目录
    let dataDir = "";
    let dataFiles = [];
    if (Array.isArray(body.attachments) && body.attachments.length) {
      const dataDirPath = resolve(runDir, "data");
      await mkdir(dataDirPath, { recursive: true });
      for (const att of body.attachments) {
        if (!att.storedPath) continue;
        const src = safeProjectPath(att.storedPath);
        const dst = resolve(dataDirPath, att.filename);
        await writeFile(dst, await readFile(src));
        dataFiles.push({
          filename: att.filename,
          file_type: att.fileType,
          path: att.filename,
          summary: att.summary || {},
        });
      }
      dataDir = dataDirPath;
    }

    const problemPath = resolve(runDir, "problem.json");
    const problem = {
      title: body.title || "Beacon UI Problem",
      background: body.background || "",
      questions: String(body.background || "")
        .split(/\n+/)
        .map((line) => line.trim())
        .filter(Boolean)
        .slice(0, 6),
    };
    if (problem.questions.length === 0) problem.questions = [body.title || "请完成数学建模分析。"];
    if (dataDir) {
      problem.data_dir = dataDir;
      problem.data_files = dataFiles;
    }
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && node --test frontend/server.test.mjs`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
git add frontend/server.mjs frontend/server.test.mjs
git commit -m "feat(server): copy attachments to run data dir + write data_files in problem.json"
```

---

### Task 11: 前端 HTML - 附件上传区

**Files:**
- Modify: `frontend/index.html:88-95`

- [ ] **Step 1: 扩展题面文件 accept + 新增附件区**

在 `frontend/index.html` 中，找到现有题面文件上传区域：

```html
            <input class="visually-hidden" id="problemFile" type="file" accept=".json,.md,.txt" />
            <div class="upload-zone" tabindex="0" role="button" aria-label="上传题目 JSON、Markdown 或文本" id="uploadZone">
              <div class="upload-icon">+</div>
              <div>
                <strong id="uploadTitle">拖入题目文件</strong>
                <span id="uploadMeta">支持 JSON、Markdown、TXT 题面</span>
              </div>
            </div>
```

替换为：

```html
            <input class="visually-hidden" id="problemFile" type="file" accept=".json,.md,.txt,.pdf,.docx" />
            <div class="upload-zone" tabindex="0" role="button" aria-label="上传题目 JSON、Markdown、文本、PDF 或 Word" id="uploadZone">
              <div class="upload-icon">+</div>
              <div>
                <strong id="uploadTitle">拖入题目文件</strong>
                <span id="uploadMeta">支持 JSON、Markdown、TXT、PDF、Word 题面</span>
              </div>
            </div>

            <input class="visually-hidden" id="attachmentFile" type="file" accept=".xlsx,.xls,.csv,.pdf,.docx,.txt,.md" multiple />
            <div class="upload-zone attachment-zone" tabindex="0" role="button" aria-label="上传数据附件" id="attachmentZone">
              <div class="upload-icon">📎</div>
              <div>
                <strong id="attachmentTitle">拖入数据附件</strong>
                <span id="attachmentMeta">支持 Excel、CSV、PDF、Word、TXT（可多选）</span>
              </div>
            </div>
            <div class="attachment-list" id="attachmentList"></div>
```

- [ ] **Step 2: 验证 HTML 加载正常**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && node -e "const fs=require('fs'); const html=fs.readFileSync('frontend/index.html','utf8'); console.log(html.includes('attachmentZone') ? 'ok' : 'missing')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
git add frontend/index.html
git commit -m "feat(html): add attachment upload zone + extend problem file accept"
```

---

### Task 12: 前端 CSS - 附件区样式

**Files:**
- Modify: `frontend/styles.css`

- [ ] **Step 1: 添加附件区样式**

在 `frontend/styles.css` 中，在 `.upload-zone` 相关样式之后（约第 403 行 `.upload-icon` 之后）添加：

```css
.attachment-zone {
  min-height: 64px;
  border-color: #b8d4c9;
  background: #f0faf6;
}

.attachment-zone .upload-icon {
  background: var(--green);
}

.attachment-list {
  display: grid;
  gap: 8px;
  margin-bottom: 16px;
}

.attachment-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--white);
  font-size: 0.86rem;
}

.attachment-item .att-name {
  font-weight: 600;
  color: var(--ink);
}

.attachment-item .att-meta {
  color: var(--muted);
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.attachment-item .att-remove {
  flex: 0 0 auto;
  padding: 2px 8px;
  border: none;
  border-radius: 4px;
  background: var(--rose);
  color: var(--white);
  font-size: 0.8rem;
  cursor: pointer;
}
```

- [ ] **Step 2: Commit**

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
git add frontend/styles.css
git commit -m "style: add attachment upload zone and list item styles"
```

---

### Task 13: 前端 JS - 附件上传/展示/PDF 提取 + run 时发送

**Files:**
- Modify: `frontend/app.js`

- [ ] **Step 1: 添加 DOM 引用和状态变量**

在 `frontend/app.js` 顶部（约第 26 行 `const uploadMeta = ...` 之后）添加：

```javascript
const attachmentZone = document.querySelector("#attachmentZone");
const attachmentFile = document.querySelector("#attachmentFile");
const attachmentList = document.querySelector("#attachmentList");
```

在约第 49 行 `let currentFixturePath = null;` 之后添加：

```javascript
let uploadedAttachments = [];
```

- [ ] **Step 2: 添加附件上传函数**

在 `frontend/app.js` 的 `loadProblemFile` 函数之后（约第 540 行之后）添加：

```javascript

async function uploadFile(file, purpose) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("purpose", purpose);
  const response = await fetch("/api/upload", { method: "POST", body: formData });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function renderAttachmentList() {
  attachmentList.innerHTML = uploadedAttachments.map((att, i) => {
    let meta = "";
    if (att.summary?.sheets?.length) {
      const s = att.summary.sheets[0];
      meta = `${s.rows}行×${s.cols}列`;
      if (s.columns?.length) meta += ` · ${s.columns.slice(0, 4).join(", ")}`;
    } else if (att.summary?.text_excerpt) {
      meta = `${Math.ceil(att.summary.text_excerpt.length / 1024)} KB 文本`;
    }
    return `<div class="attachment-item">
      <span class="att-name">${escapeHtml(att.filename)}</span>
      <span class="att-meta">${escapeHtml(att.fileType)} · ${escapeHtml(meta)}</span>
      <button class="att-remove" type="button" data-idx="${i}">×</button>
    </div>`;
  }).join("");
  attachmentList.querySelectorAll(".att-remove").forEach((btn) => {
    btn.addEventListener("click", () => {
      uploadedAttachments.splice(Number(btn.dataset.idx), 1);
      renderAttachmentList();
    });
  });
}

async function loadAttachmentFile(file) {
  if (!file) return;
  showToast(`正在上传 ${file.name}...`);
  try {
    const result = await uploadFile(file, "attachment");
    uploadedAttachments.push(result);
    renderAttachmentList();
    showToast(`${file.name} 已上传`);
  } catch (error) {
    showToast(`上传失败：${error.message}`);
  }
}
```

- [ ] **Step 3: 修改 `loadProblemFile` 支持 PDF/DOCX**

将现有的 `loadProblemFile` 函数替换为：

```javascript
function loadProblemFile(file) {
  if (!file) return;
  uploadTitle.textContent = file.name;
  uploadMeta.textContent = `${Math.ceil(file.size / 1024)} KB · ${file.type || "本地文件"}`;
  problemBadge.textContent = file.name.split(".").pop()?.toUpperCase() || "FILE";
  currentFixturePath = null;
  const ext = file.name.split(".").pop()?.toLowerCase();
  if (ext === "json" || ext === "md" || ext === "txt") {
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      const text = String(reader.result || "");
      try {
        const data = JSON.parse(text);
        const questions = Array.isArray(data.questions)
          ? data.questions.filter((item) => typeof item === "string")
          : [];
        if (typeof data.title === "string" && data.title) problemTitle.value = data.title;
        const background = typeof data.background === "string" ? data.background : "";
        problemBrief.value = [background, ...questions].filter(Boolean).join("\n") || text.slice(0, MAX_PROBLEM_TEXT_CHARS);
      } catch {
        problemBrief.value = text.slice(0, MAX_PROBLEM_TEXT_CHARS) || problemBrief.value;
      }
      showToast("题目文件已读取");
    });
    reader.readAsText(file, "utf-8");
  } else if (ext === "pdf" || ext === "docx") {
    showToast(`正在提取 ${file.name} 文本...`);
    uploadFile(file, "problem").then((result) => {
      if (result.text) {
        problemBrief.value = result.text.slice(0, MAX_PROBLEM_TEXT_CHARS);
        showToast(`${file.name} 文本已提取`);
      } else {
        showToast("未能从文件中提取文本");
      }
    }).catch((error) => {
      showToast(`提取失败：${error.message}`);
    });
  } else {
    showToast("当前仅支持 JSON、Markdown、TXT、PDF、Word 文件");
  }
}
```

- [ ] **Step 4: 添加附件区事件监听**

在 `frontend/app.js` 的 `uploadZone?.addEventListener("drop", ...)` 之后添加：

```javascript

attachmentZone?.addEventListener("click", () => attachmentFile.click());
attachmentZone?.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    attachmentFile.click();
  }
});
attachmentFile?.addEventListener("change", () => {
  for (const file of attachmentFile.files) {
    loadAttachmentFile(file);
  }
  attachmentFile.value = "";
});
attachmentZone?.addEventListener("dragover", (event) => {
  event.preventDefault();
  attachmentZone.classList.add("dragging");
});
attachmentZone?.addEventListener("dragleave", () => attachmentZone.classList.remove("dragging"));
attachmentZone?.addEventListener("drop", (event) => {
  event.preventDefault();
  attachmentZone.classList.remove("dragging");
  for (const file of event.dataTransfer.files) {
    loadAttachmentFile(file);
  }
});
```

- [ ] **Step 5: 修改 `startProjectRun` 发送 attachments**

在 `frontend/app.js` 的 `startProjectRun` 函数中，找到 `const { run } = await api("/api/run", {` 调用，在 body 对象中添加 `attachments` 字段：

```javascript
  const { run } = await api("/api/run", {
    method: "POST",
    body: JSON.stringify({
      title: problemTitle.value,
      background: problemBrief.value,
      fixturePath: currentFixturePath,
      outputDir: outputDir.value || "runs/ui-latest",
      threadId: threadId.value || "default",
      template: activeTemplate,
      noInterrupt: !hitlToggle.checked,
      ragEnabled: ragToggle.checked,
      iterationDepth: Number.parseInt(iterationDepth.value, 10),
      force: forceToggle.checked,
      attachments: uploadedAttachments,
    }),
  });
```

- [ ] **Step 6: 验证前端加载无报错**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && node -e "const fs=require('fs'); const js=fs.readFileSync('frontend/app.js','utf8'); console.log(js.includes('uploadFile') && js.includes('renderAttachmentList') && js.includes('uploadedAttachments') ? 'ok' : 'missing')"`
Expected: `ok`

- [ ] **Step 7: Commit**

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
git add frontend/app.js
git commit -m "feat(frontend): attachment upload/remove/display + PDF/DOCX text extraction + send on run"
```

---

### Task 14: 端到端验证

**Files:** 无修改，仅验证

- [ ] **Step 1: 运行全部 Python 测试**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && uv run pytest tests/ -x -q`
Expected: 全部 PASS（无 regression）

- [ ] **Step 2: 运行前端测试**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && node --test frontend/server.test.mjs`
Expected: 全部 PASS

- [ ] **Step 3: 启动服务端手动验证**

Run: `cd C:/Users/lwh86/Desktop/progame/build-agent && node frontend/server.mjs`
Expected: 服务启动，访问 `http://127.0.0.1:5173` 能看到附件上传区

- [ ] **Step 4: 用华中杯题目手动测试（如果文件可用）**

手动操作：
1. 打开浏览器访问 `http://127.0.0.1:5173`
2. 上传 `A题：城市绿色物流配送调度.pdf` 作为题面 -> 验证题面文本自动填充
3. 上传 4 个 xlsx 附件 -> 验证附件卡片展示行数/列名
4. 启动流水线 -> 验证日志中出现数据文件路径

- [ ] **Step 5: 最终 commit（如有修复）**

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
git add -A
git commit -m "test: e2e verification of attachment upload feature"
```

---

## Self-Review 结果

**1. Spec coverage:**
- ✅ 前端题面扩展 PDF/DOCX -> Task 11, 13
- ✅ 前端数据附件区 -> Task 11, 12, 13
- ✅ 服务端 multipart parser + `/api/upload` -> Task 9
- ✅ 摘要脚本 -> Task 2
- ✅ `/api/run` 附件复制 -> Task 10
- ✅ State 新增模型 -> Task 3
- ✅ CLI 读取 data_files -> Task 4
- ✅ analyst 注入摘要 -> Task 6
- ✅ coder 注入路径 + SYSTEM 微调 -> Task 7
- ✅ sensitivity 注入路径 -> Task 8
- ✅ 共享 helper -> Task 5
- ✅ 新增依赖 -> Task 1
- ✅ 端到端验证 -> Task 14

**2. Placeholder scan:** 无 TBD/TODO，所有步骤含完整代码。

**3. Type consistency:**
- `DataFileInfo(filename, file_type, path, summary)` - Task 3 定义，Task 4 CLI 用 `DataFileInfo(**f)`，Task 5/6/7/8 通过属性访问。一致。
- `build_data_hint(data_dir, data_files)` - Task 5 定义，Task 7/8 调用。一致。
- `build_data_summary_hint(data_files)` - Task 5 定义，Task 6 调用。一致。
- 服务端 `storedPath` -> Task 9 返回，Task 10 用 `att.storedPath` 读取。一致。
- `uploadedAttachments` -> Task 13 定义并在 `startProjectRun` 发送。一致。
