# 附件上传功能设计

> **历史设计记录（非现行事实源）**：附件上传已进入现行实现；旧状态与路径保留当时上下文，当前说明见 [`../../README.md`](../../README.md)。

> 日期：2026-07-11
> 状态：设计确认，待实现

## 背景与问题

当前系统上传题目只支持 JSON/Markdown/TXT 文本文件，无法上传附件。华中杯 A 题（城市绿色物流配送调度）等竞赛题目包含 PDF 题面 + xlsx 附件（客户坐标、时间窗、订单信息、距离矩阵），没有附件数据则建模结果不准确。

## 目标

让用户能上传 PDF/Word 题面和 xlsx/csv/pdf/docx/txt/md 数据附件，系统：
1. 自动提取题面文本（PDF/Word）
2. 自动生成数据附件摘要（行数、列名、预览）
3. 将附件摘要注入 analyst prompt（理解数据概况）
4. 将附件文件路径注入 coder/sensitivity prompt（直接读取真实数据）

## 整体架构与数据流

```
前端 (app.js / index.html)
  │  1. 题面文件上传：扩展支持 PDF/DOCX（调服务端提取文本填充题面）
  │  2. 新增"数据附件"区域：拖拽/选择多文件上传
  │  3. 上传后展示每个附件的摘要
  ▼
服务端 (server.mjs)
  │  4. POST /api/upload：接收 multipart/form-data，存到临时目录
  │  5. 调用 extract_file_meta.py 生成摘要
  │  6. POST /api/run：接收 attachments，复制到 run 专属 data 目录，写入 problem.json
  ▼
CLI (cli.py)
  │  7. _read_problem_spec：读取 data_files + data_dir
  │  8. initial state：注入 data_dir / data_files
  ▼
State (state.py)
  │  9. MathModelingState 新增 data_dir + data_files 字段
  ▼
Nodes (analyst / coder / sensitivity)
  10. analyst prompt 注入数据摘要
  11. coder/sensitivity prompt 注入绝对路径
```

## 前端 UI 设计

### HTML 结构

在现有"题面文件上传区域"（uploadZone）下方，新增"数据附件"区域：

```
题目配置面板
├── 赛题标题 input
├── 背景与目标 textarea
├── 题面文件上传区 (uploadZone)  ← 扩展 accept 加 .pdf,.docx
│   └── 拖入/点击选择 JSON/MD/TXT/PDF/DOCX
├── 【新增】数据附件区 (attachmentZone)  ← 支持多文件
│   ├── 拖入/点击选择 xlsx/xls/csv/pdf/docx/txt/md
│   └── 附件列表 (attachmentList)
│       └── 每个附件卡片：文件名 · 类型 · N行×M列 · [×]移除
└── 控制开关 (RAG / HITL / Force)
```

### 文件类型支持矩阵

| 文件类型 | 作为题面文件 | 作为数据附件 | 摘要方式 |
|---------|------------|------------|---------|
| JSON / MD / TXT | ✅ 浏览器端直接读文本 | ✅ 存储传递 | 行数 + 前3000字 |
| PDF | ✅ 服务端提取文本填充题面 | ✅ 存储传递 | 前5000字文本摘录 |
| Word (.docx) | ✅ 服务端提取文本填充题面 | ✅ 存储传递 | 段落数 + 表格数 + 前3000字 |
| Excel (.xlsx/.xls) | ❌ 不适合 | ✅ 存储传递 | sheet列表 + 行/列/前5行预览 |
| CSV | ❌ 不适合 | ✅ 存储传递 | 行数 + 列名 + 前5行预览 |

### 交互流程

**题面文件（PDF/DOCX）**：
1. 用户选择/拖入文件 → 前端 `fetch("/api/upload", { body: FormData, purpose: "problem" })`
2. 服务端提取文本 → 返回 `{ text, filename, summary }`
3. 前端把 `text` 填充到 `problemBrief`，用户可检查修改

**数据附件**：
1. 用户拖入文件 → `POST /api/upload` (purpose: "attachment")
2. 服务端存储 + 生成摘要 → 返回 `{ id, filename, fileType, summary }`
3. 前端在 attachmentList 渲染卡片：`[订单信息.xlsx · Excel · 1200行×5列 · 订单号/客户ID/...] [×]`
4. 支持多文件累加，每个可单独移除

**启动流水线**：`POST /api/run` body 新增 `attachments` 数组。

### 不做的事
- 不在前端 JS 里解析 xlsx/docx（交给服务端 Python）
- 不做附件内容全文展示（只展示摘要）
- 不做附件的编辑功能（只上传/移除）

## 服务端 API 设计

### multipart parser

纯 Node.js 手写（约50行），不引入 formidable。按 boundary 分割 buffer，提取 Content-Disposition 头，流式写入磁盘。

限制：
- 单文件 50MB
- 总请求体 200MB
- 流式写入磁盘，不在内存缓存完整文件

### `POST /api/upload`（新增）

```
请求：multipart/form-data
  - field "file": 二进制文件内容
  - field "purpose": "problem" | "attachment"

响应 200：
{
  "id": "att-1717706400-ab12",
  "filename": "订单信息.xlsx",
  "fileType": "xlsx",
  "size": 77107,
  "storedPath": "runs/ui-server/uploads/att-1717706400-ab12/订单信息.xlsx",
  "summary": { ... },
  "text": "..."                // 仅 purpose=problem 返回提取文本；attachment 为空
}

错误：
  415 - 不支持的文件类型
  413 - 文件过大（限制 50MB）
  500 - 摘要生成失败
```

存储路径：`runs/ui-server/uploads/{id}/{原文件名}`，每个上传分配独立子目录。

### 摘要生成脚本

`scripts/extract_file_meta.py`：

服务端 `spawn(python, ["scripts/extract_file_meta.py", filePath])` 调用，脚本输出 JSON：

```json
{
  "file_type": "xlsx",
  "filename": "订单信息.xlsx",
  "summary": {
    "sheets": [
      {"name": "Sheet1", "rows": 1200, "cols": 5,
       "columns": ["订单号", "客户ID", "需求量", "送达时间窗", "优先级"],
       "preview": [["D001", "C001", "50", "08:00-10:00", "高"], ...]}
    ]
  },
  "text_excerpt": ""
}
```

不同文件类型摘要：
- **xlsx/xls**：sheet 列表，每 sheet 的行数/列名/前5行预览（openpyxl）。最多摘要 5 个 sheet，超出部分只记 sheet 名
- **csv**：行数、列名、前5行预览（pandas）
- **pdf**：提取全文文本前5000字（pypdf，复用 rag/ingest.py 逻辑）
- **docx**：段落数、表格数、前3000字文本（python-docx）
- **txt/md**：行数、前3000字文本

### `POST /api/run`（修订）

body 新增 `attachments` 数组：

```js
{
  title, background, outputDir, threadId, ...,
  attachments: [
    { id, filename, fileType, storedPath, summary }
  ]
}
```

服务端在 `POST /api/run` 时把所有附件复制到 run 专属 data 目录：

```
runs/ui-server/{runId}/data/
  ├── 订单信息.xlsx
  ├── 客户坐标信息.xlsx
  ├── 时间窗.xlsx
  └── 距离矩阵.xlsx
```

`problem.json` 写入：

```json
{
  "title": "...",
  "background": "...",
  "questions": [...],
  "data_dir": "runs/ui-server/{runId}/data/",
  "data_files": [
    {
      "filename": "订单信息.xlsx",
      "file_type": "xlsx",
      "path": "订单信息.xlsx",
      "summary": { "sheets": [...] }
    }
  ]
}
```

`data_dir` 为绝对路径，`data_files[].path` 为相对于 `data_dir` 的文件名。

### 不做的事
- 不做断点续传
- 不做文件持久化管理（随 run 目录手动清理）
- 不做病毒扫描（单用户本机环境）

## CLI 与 State 改动

### State 新增字段

```python
class DataFileSheet(BaseModel):
    """xlsx/csv 的单个 sheet/表摘要。"""
    name: str
    rows: int = 0
    cols: int = 0
    columns: list[str] = Field(default_factory=list)
    preview: list[list[str]] = Field(default_factory=list)

class DataFileInfo(BaseModel):
    """上传附件的元信息。"""
    filename: str
    file_type: str                           # "xlsx"|"csv"|"pdf"|"docx"|"txt"|"md"
    path: str                                # 相对 data_dir 的文件名
    summary: dict = Field(default_factory=dict)  # 灵活 dict，不同类型结构不同

class MathModelingState(BaseModel):
    # ... 现有字段不变 ...
    data_dir: Optional[str] = None
    data_files: list[DataFileInfo] = Field(default_factory=list)
```

设计决策：
- `summary` 用 `dict` 而非强类型模型，因为不同 file_type 结构差异大
- `data_dir` 是绝对路径
- `data_files` 用覆盖语义（非 add），附件在 run 启动时确定

### CLI `_read_problem_spec` 扩展

新增读取 `data_files` 和 `data_dir`（可选字段）：
- 校验 `data_dir` 存在且是目录
- 相对路径基于 problem.json 所在目录解析
- 转为绝对路径供 coder 使用

### CLI `run` 命令

`initial` dict 新增：

```python
initial = {
    # ... 现有字段 ...
    "data_dir": spec.get("data_dir") or None,
    "data_files": [DataFileInfo(**f) for f in spec.get("data_files", [])],
}
```

### 兼容性
- 旧 fixture 无 data_files/data_dir → 空列表/None，不受影响
- bench problems 已有空 `data_files: []` 字段，这次让它真正被消费
- resume 从 checkpoint 恢复，state 中已有 data_dir/data_files

### 不做的事
- 不新增 `--data-dir` CLI 参数（全部通过 problem.json 传递）
- 不在 state 中存储文件内容（只存路径+摘要）
- 不在 resume 中处理附件

## 节点 Prompt 注入

### Analyst 注入

`prompts/analyst.py` 的 `build_prompt` 新增 `data_files` 参数：

注入格式：
```
# 附件数据概况
已有以下数据文件可用：
- 订单信息.xlsx (xlsx)
  └ Sheet1: 1200行×5列 [订单号, 客户ID, 需求量, 送达时间窗, 优先级]
- 距离矩阵.xlsx (xlsx)
  └ Sheet1: 50行×50列 [客户001, 客户002, ...]
请在 data_requirements 中将对应字段标注为 given，并在建模路线中考虑如何使用这些真实数据。
```

`analyst_node` 传入 `state.data_files`。

### Coder 注入

`prompts/coder_figure_one.py` 的 `build_prompt_figure_one` 新增 `data_dir` 和 `data_files` 参数：

注入格式：
```
# 可用数据文件
数据目录: C:\...\runs\ui-server\{runId}\data
- C:\...\data\订单信息.xlsx (Excel, 用 pd.read_excel 读取)
- C:\...\data\距离矩阵.xlsx (Excel, 用 pd.read_excel 读取)
请优先读取这些真实数据进行计算，不要编造 mock 数据。
路径中含中文/空格时用 r-string：pd.read_excel(r"...")
```

`coder_node` 传入 `state.data_dir` 和 `state.data_files`。

### Sensitivity 注入

`prompts/sensitivity.py` 的 `build_code_prompt` 新增 `data_dir` 和 `data_files` 参数，注入逻辑与 coder 相同。仅注入到 CODE 阶段，不注入 PLAN 阶段。

### SYSTEM prompt 微调

coder.py 的 SYSTEM 当前写着"不读取本地未声明的文件"：

```
# 现有
"不联网；不读取本地未声明的文件；"

# 改为
"不联网；只读取 prompt 中明确给出路径的数据文件，不读取其他本地文件；"
```

### 共享 helper

coder 和 sensitivity 的 data_hint 构造逻辑提取为 `prompts/_data_hint.py`：

```python
def build_data_hint(data_dir: str | None, data_files: list) -> str:
    """构造数据文件路径提示，供 coder/sensitivity prompt 共用。"""
    ...
```

### 不做的事
- 不修改 modeler 节点（通过 blueprint 间接获知数据可用性）
- 不修改 writer 节点（从 code_artifacts stdout 拿结果）
- 不在 sensitivity PLAN 阶段注入数据文件
- 不把 pandas/openpyxl 加入 runner.py auto-import 列表（LLM 应自己写 import）

## 新增依赖

- `python-docx>=1.1`（pyproject.toml）— 读取 .docx 文件
- `openpyxl>=3.1`（pyproject.toml）— 读取 .xlsx 文件（pandas 读取 xlsx 需要它作为引擎）
- pandas 已在环境中可用（coder 的 auto-import 逻辑暗示）

## 不做的事（全局）

- 不做附件版本管理
- 不做附件预览的全文展示
- 不做附件内容的自动清洗/格式转换（LLM 自己在代码中处理）
- 不修改 modeler/writer/paper_critic 等不直接使用数据文件的节点
