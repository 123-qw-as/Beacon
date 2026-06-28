# 数学建模多智能体系统 — 质量与产出层实现计划（Plan B）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Plan A（MVP 端到端跑通）之上，把"强制敏感性分析、图表生成-评审-解读闭环、PaperCritic、独立 Evaluation Module、HITL+Checkpointer、LaTeX 生成与编译"六块质量/产出能力补齐，让系统能产出对齐国一评审标准的 LaTeX 论文与 PDF。

**Architecture:** 继续以 `LangGraph StateGraph` 为单一编排器；在 Plan A 的 `Coder` 之后插入 *Sensitivity → Figure pipeline → Writer → PaperCritic → Evaluation → HITL(human_review) → LaTeX compile* 的链路；新增节点遵循 Plan A 的约定（`(MathModelingState) -> dict` 增量返回，LLM 调用必须经 `math_agent.llm.complete`）。HITL 通过 LangGraph 的 `interrupt_before` + `SqliteSaver` 实现，便于断点恢复。LaTeX 用 Jinja2 模板渲染 + `xelatex` 子进程编译；任何阶段失败都回退到 Markdown，保证总有产出。

**Tech Stack:**（在 Plan A 的基础上新增）
- `langgraph.checkpoint.sqlite.SqliteSaver`（HITL 检查点）
- `pillow`（图像尺寸/DPI 校验，FigureCritic 用）
- `xelatex`（系统二进制；通过 `subprocess` 调用）
- `tectonic`（可选；作为 `xelatex` 不可用时的备选编译器）
- 仍然只通过 `litellm` 调用 LLM；对图像理解直接走 LiteLLM 的多模态消息格式（`image_url`）

> **前置条件**：Plan A 已完成且 `pytest -q` 全绿。Plan B 不会修改 Plan A 已交付的 prompt/节点行为，只通过新增节点 + 改 `graph.py` 的接线来扩展。

---

## 文件结构

新增（all paths relative to `build-agent/`）：

```
src/math_agent/
├── prompts/
│   ├── sensitivity.py
│   ├── figure_critic.py
│   ├── figure_analyst.py
│   ├── paper_critic.py
│   └── evaluation.py
├── nodes/
│   ├── sensitivity.py
│   ├── figure_pipeline.py       # generate -> critic -> analyst (一个节点内闭环)
│   ├── paper_critic.py
│   ├── evaluation.py
│   ├── human_review.py          # interrupt-based passthrough
│   └── latex.py                 # render + compile + fallback
├── tools/
│   ├── image.py                 # PIL 检查 + base64 编码
│   └── latex_compile.py         # xelatex 子进程
├── templates/
│   ├── paper.tex.j2
│   └── figure_caption.md.j2     # 给 sensitivity/figure section 拼装图说
└── state.py                     # 修改：新增 figures, evaluation, human_decision 字段

tests/
├── nodes/
│   ├── test_sensitivity.py
│   ├── test_figure_pipeline.py
│   ├── test_paper_critic.py
│   ├── test_evaluation.py
│   ├── test_human_review.py
│   └── test_latex.py
├── tools/
│   ├── test_image.py
│   └── test_latex_compile.py
└── test_graph_full_smoke.py
```

修改：
- `src/math_agent/state.py` — 新增字段：`figures`, `sensitivity_runs`, `evaluation`, `human_decision`
- `src/math_agent/graph.py` — 串接新节点并加入 checkpointer/interrupt
- `src/math_agent/config.py` — 新增模型路由项（`figure_critic`/`paper_critic`/`evaluation`）和 LaTeX 路径
- `src/math_agent/templates/paper.md.j2` — 在 `sensitivity`/`solution` 段插入 figure 引用

---

## Phase 1：强制敏感性分析节点

### Task 1.1：扩展 State（先写测试）

**Files:**
- Modify: `tests/test_state.py`

- [ ] **Step 1：在 `tests/test_state.py` 末尾追加测试**

```python
from math_agent.state import (
    MathModelingState, SensitivityRun, FigureArtifact, EvaluationReport,
)


def test_state_has_sensitivity_runs():
    s = MathModelingState(problem="p")
    s.sensitivity_runs.append(
        SensitivityRun(
            parameter="lambda", values=[0.5, 1.0, 1.5],
            metric="avg_wait", results=[2.1, 3.5, 8.0],
            interpretation="敏感度高",
        )
    )
    assert s.sensitivity_runs[-1].parameter == "lambda"


def test_state_has_figures():
    s = MathModelingState(problem="p")
    s.figures.append(
        FigureArtifact(
            path="runs/x/fig1.png", purpose="对比", caption="见正文",
            quality_score=8, analysis="单调上升",
        )
    )
    assert s.figures[-1].quality_score == 8


def test_state_has_evaluation_default_none():
    s = MathModelingState(problem="p")
    assert s.evaluation is None


def test_state_has_human_decision_default_none():
    s = MathModelingState(problem="p")
    assert s.human_decision is None
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/test_state.py -v`
Expected: ImportError（缺 `SensitivityRun` / `FigureArtifact` / `EvaluationReport`）。

---

### Task 1.2：实现 State 扩展

**Files:**
- Modify: `src/math_agent/state.py`

- [ ] **Step 1：在 `state.py` 中追加模型与字段**

在已有的 import 后追加：

```python
from typing import Any
```

在 `class CriticReport(BaseModel):` 上方追加：

```python
class SensitivityRun(BaseModel):
    parameter: str
    values: list[float]
    metric: str
    results: list[float]
    interpretation: str = ""
    figure_path: str | None = None


class FigureArtifact(BaseModel):
    path: str
    purpose: str
    caption: str = ""
    quality_score: int = 0           # 0-10，FigureCritic 打分
    quality_issues: list[str] = Field(default_factory=list)
    analysis: str = ""               # FigureAnalyst 产出的段落


class EvaluationReport(BaseModel):
    """对齐国赛四大标准 + 国一加分项。每项 0-10。"""
    assumption_reasonableness: int
    modeling_creativity: int
    result_correctness: int
    writing_clarity: int
    extra_depth: int                 # 加分项：敏感性/创新/分析深度
    overall: float                   # 加权总评
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class HumanDecision(BaseModel):
    approved: bool
    notes: str = ""
```

在 `PaperSections` 中加回 `sensitivity` 字段（Plan A 删除，Plan B 引入 Sensitivity 节点后加回）：

```python
class PaperSections(BaseModel):
    abstract: str = ""
    problem_restatement: str = ""
    assumptions: str = ""
    notation: str = ""
    model_section: str = ""
    solution: str = ""
    sensitivity: str = ""            # Plan B 引入；Plan A 默认空
    conclusion: str = ""
    references: str = ""
```

在 `class MathModelingState(BaseModel):` 内已有字段下方追加（list 字段必须用 `Annotated[..., add]`
以匹配 Plan A 已确立的"Pydantic state 直接当 LangGraph state schema"约定）：

```python
    sensitivity_runs: Annotated[list[SensitivityRun], add] = Field(default_factory=list)
    figures: Annotated[list[FigureArtifact], add] = Field(default_factory=list)
    evaluation: EvaluationReport | None = None
    human_decision: HumanDecision | None = None
```

- [ ] **Step 2：同步更新 Plan A 已落地的 Writer 产物**

Plan A 把 `paper.sensitivity` 字段、Writer prompt 输出键、Markdown 模板对应章节都删了。
Plan B 现在加回字段，对应的 Writer 也要补回输出键和模板章节：

修改 `src/math_agent/prompts/writer.py` 的 `build_prompt`，把 JSON schema 描述加回 `sensitivity`：

```python
        f"请输出 JSON：{{\"abstract\":str,\"problem_restatement\":str,\"assumptions\":str,"
        f"\"notation\":str,\"model_section\":str,\"solution\":str,\"sensitivity\":str,"
        f"\"conclusion\":str,\"references\":str}}。每段不少于 150 字。"
        f"（sensitivity 段：综述 state.sensitivity_runs 的关键发现；若无 runs 则写'未执行敏感性分析'。）"
```

在 `src/math_agent/templates/paper.md.j2` 的 `## 5. 模型的求解` 与 `## 6. 模型评价与结论`
之间插入：

```jinja
## 6. 敏感性分析
{{ paper.sensitivity }}
```

并把后续 `## 6. 模型评价与结论` 改为 `## 7. 模型评价与结论`。

修改 `tests/nodes/test_writer.py` 的 `test_writer_fills_paper` 与 `test_render_markdown_contains_sections`：
`PaperSections(...)` 构造加上 `sensitivity="..."`；`assert "## 6. 模型评价" in md` 改为 `assert "## 7. 模型评价" in md`。

- [ ] **Step 3：跑测试**

Run: `pytest tests/test_state.py tests/nodes/test_writer.py -v`
Expected: 全部通过（含 4 个 state 新增）。

- [ ] **Step 4：提交**

```bash
git add src/math_agent/state.py src/math_agent/prompts/writer.py src/math_agent/templates/paper.md.j2 tests/
git commit -m "feat(state): add sensitivity_runs, figures, evaluation, human_decision; restore PaperSections.sensitivity"
```

---

### Task 1.3：Sensitivity prompt

**Files:**
- Create: `src/math_agent/prompts/sensitivity.py`

- [ ] **Step 1：写 prompt**

```python
"""Sensitivity：选择关键参数 + 给出扫参代码（仍由沙箱执行）+ 解读结果。

设计：把"选参数+造代码"和"读结果"拆成两个 prompt，避免一次返回过大的 JSON。
"""

PLAN_SYSTEM = (
    "你是国赛评委关心的敏感性分析专家。请基于已有的最终模型和已确认假设，"
    "选出 1-3 个最值得做敏感性分析的参数。优先选标记了 sensitivity_relevant=True 的假设里出现的参数。"
)


def build_plan_prompt(model, assumptions):
    asum = "\n".join(
        f"- [{'敏感' if a.sensitivity_relevant else '常规'}] {a.statement}"
        for a in assumptions
    )
    eqs = "\n".join(f"- {e}" for e in model.equations)
    return (
        f"# 最终模型\n{model.description}\n方程：\n{eqs}\n\n# 假设\n{asum}\n\n"
        f"请输出 JSON：{{\"runs\": [{{\"parameter\": str, \"values\": [float, ...], "
        f"\"metric\": str, \"rationale\": str}}, ...]}}，"
        f"每个 run 的 values 至少 5 个点，跨度合理（涵盖参数典型范围的 ±30%~50%）。"
    )


CODE_SYSTEM = (
    "你是建模队工程师。根据敏感性分析计划，写一段独立可运行的 Python，"
    "对每个 run 计算 metric 随 parameter 变化的曲线，并保存 PNG 到当前目录。"
    "约束：只用 numpy/scipy/matplotlib；为每个 run 单独保存一张 *.png；"
    "用 print 输出 `RESULT: parameter=... values=... results=...` 行（每个 run 一行），方便解析。"
)


def build_code_prompt(model, plan_runs):
    desc = "\n".join(
        f"- parameter={r['parameter']}, values={r['values']}, metric={r['metric']}"
        for r in plan_runs
    )
    return (
        f"# 最终模型\n{model.description}\n方程：\n{chr(10).join(model.equations)}\n\n"
        f"# 敏感性分析计划\n{desc}\n\n"
        f"请输出 JSON：{{\"code\": str}}。"
    )


INTERPRET_SYSTEM = (
    "你是国赛主笔。根据敏感性分析的数值结果，写出每个参数的解读（趋势 + 含义 + 对结论的影响），"
    "每条 80-150 字，避免空话。"
)


def build_interpret_prompt(runs):
    rows = "\n".join(
        f"- {r.parameter}={r.values} → {r.metric}={r.results}"
        for r in runs
    )
    return (
        f"# 数值结果\n{rows}\n\n"
        f"请输出 JSON：{{\"interpretations\": [str, ...]}}，长度与上面行数一致。"
    )
```

- [ ] **Step 2：提交**

```bash
git add src/math_agent/prompts/sensitivity.py
git commit -m "feat: sensitivity prompts (plan/code/interpret)"
```

---

### Task 1.4：Sensitivity 节点（先写测试）

**Files:**
- Create: `tests/nodes/test_sensitivity.py`

- [ ] **Step 1：写失败的测试**

```python
from pathlib import Path
from math_agent.state import MathModelingState, ModelVersion, Assumption, SensitivityRun
from math_agent.nodes.sensitivity import sensitivity_node, SensitivityPlan, SensitivityCode, Interpretations


def _ok_state(workdir):
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.assumptions.append(Assumption(statement="lambda 是常数", rationale="r", sensitivity_relevant=True))
    s.model_versions.append(ModelVersion(stage="final", description="d"*200, equations=["x=lambda"]))
    return s


def test_sensitivity_runs_plan_then_code_then_interpret(mocker, workdir):
    plan = SensitivityPlan(runs=[{"parameter": "lambda", "values": [0.5, 1, 1.5, 2, 2.5],
                                  "metric": "y", "rationale": "核心参数"}])
    code = SensitivityCode(code=(
        "import matplotlib.pyplot as plt\n"
        "vals=[0.5,1,1.5,2,2.5]; res=[v*2 for v in vals]\n"
        "plt.plot(vals,res); plt.savefig('lambda.png')\n"
        "print(f'RESULT: parameter=lambda values={vals} results={res}')\n"
    ))
    interp = Interpretations(interpretations=["参数 lambda 上升时 y 线性增长，敏感度中等。"])
    mocker.patch("math_agent.nodes.sensitivity.complete", side_effect=[plan, code, interp])

    delta = sensitivity_node(_ok_state(workdir))
    assert len(delta["sensitivity_runs"]) == 1
    run = delta["sensitivity_runs"][0]
    assert run.parameter == "lambda"
    assert run.results == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert run.interpretation.startswith("参数 lambda")
    assert run.figure_path and Path(run.figure_path).exists()


def test_sensitivity_records_error_when_no_final_model(mocker, workdir):
    s = MathModelingState(problem="p", output_dir=str(workdir))
    delta = sensitivity_node(s)
    assert delta["errors"]
    assert delta.get("sensitivity_runs", []) == []


def test_sensitivity_falls_back_when_code_fails(mocker, workdir):
    plan = SensitivityPlan(runs=[{"parameter": "lambda", "values": [1, 2, 3, 4, 5],
                                  "metric": "y", "rationale": "x"}])
    bad = SensitivityCode(code="raise RuntimeError('x')")
    # 失败时不应再调 interpret；node 应记录 errors 并仍写入一个空 run 占位
    mocker.patch("math_agent.nodes.sensitivity.complete", side_effect=[plan, bad])
    delta = sensitivity_node(_ok_state(workdir))
    assert delta["errors"]
    assert delta["sensitivity_runs"] == []  # 不放半成品
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/nodes/test_sensitivity.py -v`
Expected: ImportError。

---

### Task 1.5：实现 Sensitivity 节点

**Files:**
- Create: `src/math_agent/nodes/sensitivity.py`

- [ ] **Step 1：写节点**

```python
"""Sensitivity 节点：作为'必经'节点存在。

流程：
  1. PLAN：LLM 选参数 + 给出每个参数的扫值。
  2. CODE：LLM 写一段扫参 Python，沙箱执行；解析每行 `RESULT: ...`。
  3. INTERPRET：把数值结果回灌给 LLM 生成每个 run 的解读段。
失败策略：
  - 任何一步失败 → 记录 errors，**不写入半成品 sensitivity_runs**。
  - 调用方（graph）在敏感性失败时不应阻塞流水线，但 PaperCritic / Evaluation 会因 sensitivity_runs 为空而扣分。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from pydantic import BaseModel

from math_agent.config import MODEL_ROUTING
from math_agent.llm import complete
from math_agent.prompts.sensitivity import (
    PLAN_SYSTEM, CODE_SYSTEM, INTERPRET_SYSTEM,
    build_plan_prompt, build_code_prompt, build_interpret_prompt,
)
from math_agent.state import MathModelingState, SensitivityRun
from math_agent.tools.runner import run_python


class _PlanRun(BaseModel):
    parameter: str
    values: list[float]
    metric: str
    rationale: str = ""


class SensitivityPlan(BaseModel):
    runs: list[_PlanRun]


class SensitivityCode(BaseModel):
    code: str


class Interpretations(BaseModel):
    interpretations: list[str]


_RESULT_RE = re.compile(r"RESULT:\s*parameter=(\S+)\s+values=(\[[^\]]+\])\s+results=(\[[^\]]+\])")


def _parse_results(stdout: str) -> list[tuple[str, list[float], list[float]]]:
    out = []
    for line in stdout.splitlines():
        m = _RESULT_RE.search(line)
        if not m:
            continue
        param = m.group(1)
        values = [float(x) for x in ast.literal_eval(m.group(2))]
        results = [float(x) for x in ast.literal_eval(m.group(3))]
        out.append((param, values, results))
    return out


def sensitivity_node(state: MathModelingState) -> dict:
    final = next((m for m in reversed(state.model_versions) if m.stage == "final"), None)
    if final is None:
        return {"errors": ["sensitivity: 缺少 final 阶段模型，跳过敏感性分析"]}

    workdir = Path(state.output_dir or ".") / "sensitivity"
    workdir.mkdir(parents=True, exist_ok=True)

    # 1) PLAN
    plan: SensitivityPlan = complete(
        build_plan_prompt(final, state.assumptions),
        schema=SensitivityPlan, system=PLAN_SYSTEM,
        model=MODEL_ROUTING.get("modeler"),
    )
    if not plan.runs:
        return {"errors": ["sensitivity: LLM 未给出可执行的 runs"]}

    # 2) CODE
    code_out: SensitivityCode = complete(
        build_code_prompt(final, [r.model_dump() for r in plan.runs]),
        schema=SensitivityCode, system=CODE_SYSTEM,
        model=MODEL_ROUTING.get("coder"),
    )
    sandbox_result = run_python(code_out.code, workdir=workdir)
    if not sandbox_result.success:
        return {"errors": [f"sensitivity: 扫参代码执行失败：{sandbox_result.stderr[:500]}"]}

    parsed = _parse_results(sandbox_result.stdout)
    if not parsed:
        return {"errors": ["sensitivity: 未在 stdout 中解析到任何 `RESULT:` 行"]}

    # 把 parsed 与 plan.runs 对齐（按 parameter 名匹配；缺失的剔除）
    by_name = {p.parameter: p for p in plan.runs}
    aligned: list[SensitivityRun] = []
    for param, vals, res in parsed:
        plan_entry = by_name.get(param)
        if plan_entry is None:
            continue
        fig = next((p for p in sandbox_result.artifact_paths if Path(p).stem == param), None)
        aligned.append(SensitivityRun(
            parameter=param, values=vals, metric=plan_entry.metric,
            results=res, figure_path=fig,
        ))
    if not aligned:
        return {"errors": ["sensitivity: 解析结果与计划无法对齐"]}

    # 3) INTERPRET
    interp: Interpretations = complete(
        build_interpret_prompt(aligned),
        schema=Interpretations, system=INTERPRET_SYSTEM,
        model=MODEL_ROUTING.get("writer"),
    )
    for r, text in zip(aligned, interp.interpretations):
        r.interpretation = text

    return {"sensitivity_runs": aligned}
```

- [ ] **Step 2：跑测试**

Run: `pytest tests/nodes/test_sensitivity.py -v`
Expected: 3 passed.

- [ ] **Step 3：提交**

```bash
git add src/math_agent/nodes/sensitivity.py tests/nodes/test_sensitivity.py
git commit -m "feat: sensitivity node (plan -> sweep -> interpret)"
```

---

## Phase 2：图表流水线（生成 → 评审 → 解读）

> 设计要点：把"图的质量评审 + 图说生成"作为**一个节点 `figure_pipeline_node`** 实现，内部最多重试 1 次。理由：图表数量不固定，外部循环会让 graph 拓扑变复杂；放进单节点更易测试和限重试。

### Task 2.1：image 工具

**Files:**
- Create: `tests/tools/test_image.py`
- Create: `src/math_agent/tools/image.py`

- [ ] **Step 1：写失败的测试**

```python
from pathlib import Path
from PIL import Image
from math_agent.tools.image import inspect_image, encode_image_to_data_url


def test_inspect_image_returns_size_and_dpi(workdir):
    p = workdir / "t.png"
    Image.new("RGB", (300, 200), "white").save(p, dpi=(150, 150))
    info = inspect_image(p)
    assert info.width == 300 and info.height == 200
    assert info.dpi[0] == 150


def test_encode_image_to_data_url(workdir):
    p = workdir / "t.png"
    Image.new("RGB", (10, 10), "white").save(p)
    url = encode_image_to_data_url(p)
    assert url.startswith("data:image/png;base64,")
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/tools/test_image.py -v`
Expected: ImportError.

- [ ] **Step 3：实现 `tools/image.py`**

```python
"""图像辅助工具：尺寸/DPI 检查 + base64 编码（供多模态 LLM 输入）。"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class ImageInfo:
    width: int
    height: int
    dpi: tuple[float, float]


def inspect_image(path: str | Path) -> ImageInfo:
    with Image.open(path) as im:
        dpi = im.info.get("dpi", (72.0, 72.0))
        return ImageInfo(width=im.width, height=im.height, dpi=tuple(dpi))


def encode_image_to_data_url(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
```

- [ ] **Step 4：跑测试**

Run: `pytest tests/tools/test_image.py -v`
Expected: 2 passed.

- [ ] **Step 5：提交**

```bash
git add src/math_agent/tools/image.py tests/tools/test_image.py
git commit -m "feat: image tool (inspect + data url encoding)"
```

---

### Task 2.2：扩展 `llm.complete` 支持多模态输入

**Files:**
- Modify: `src/math_agent/llm.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1：在 `tests/test_llm.py` 末尾加测试**

```python
import math_agent.llm as llm


def test_complete_with_images_packs_multimodal_content(mocker):
    captured = {}

    def _fake(model, messages, **kw):
        captured["messages"] = messages
        m = mocker.MagicMock()
        m.choices = [mocker.MagicMock(message=mocker.MagicMock(content="ok"))]
        return m

    mocker.patch("litellm.completion", side_effect=_fake)
    out = llm.complete(
        "describe this", model="gpt-4o-mini",
        images=["data:image/png;base64,AAA="],
    )
    assert out == "ok"
    user_msg = captured["messages"][-1]
    # 多模态消息应该是 list，包含 text + image_url
    assert isinstance(user_msg["content"], list)
    kinds = {p["type"] for p in user_msg["content"]}
    assert kinds == {"text", "image_url"}
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/test_llm.py::test_complete_with_images_packs_multimodal_content -v`
Expected: FAIL（`complete` 不接受 `images`）。

- [ ] **Step 3：修改 `complete`，加 `images` 参数**

在 `complete(...)` 的签名上追加 `images: list[str] | None = None,`，并把构造 messages 的部分改为：

```python
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if images:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *[{"type": "image_url", "image_url": {"url": u}} for u in images],
            ],
        })
    else:
        messages.append({"role": "user", "content": prompt})
```

- [ ] **Step 4：跑全部 `test_llm.py`**

Run: `pytest tests/test_llm.py -v`
Expected: 4 passed.

- [ ] **Step 5：提交**

```bash
git add src/math_agent/llm.py tests/test_llm.py
git commit -m "feat(llm): support multimodal image_url messages"
```

---

### Task 2.3：FigureCritic + FigureAnalyst prompts

**Files:**
- Create: `src/math_agent/prompts/figure_critic.py`
- Create: `src/math_agent/prompts/figure_analyst.py`

- [ ] **Step 1：写 `figure_critic.py`**

```python
"""FigureCritic：从清晰度/标签完整性/配色/数据是否传达 4 维度评分。"""

SYSTEM = (
    "你是论文图表评审。基于给定图像和它的目的，给 0-10 评分（>=8 视为可用）。"
    "检查：标题/坐标轴/单位/图例是否齐全；配色是否专业；点线密度是否过载；"
    "信息是否与目的匹配。"
)


def build_prompt(purpose: str, image_info: str) -> str:
    return (
        f"# 图的目的\n{purpose}\n\n# 图的元信息\n{image_info}\n\n"
        f"请输出 JSON：{{\"score\": int, \"issues\": [str], \"suggestions\": [str], \"approved\": bool}}。"
    )
```

- [ ] **Step 2：写 `figure_analyst.py`**

```python
"""FigureAnalyst：基于图像 + 数据上下文写一段专业图说。"""

SYSTEM = (
    "你是国赛论文图说撰写者。给定一张图与它对应的数据/参数信息，"
    "写一段 100-200 字的中文专业解读，覆盖：趋势、关键拐点、对模型结论的支撑。"
    "不要复述坐标轴标签。"
)


def build_prompt(purpose: str, context: str) -> str:
    return (
        f"# 图的目的\n{purpose}\n\n# 数据上下文\n{context}\n\n"
        f"请输出 JSON：{{\"analysis\": str}}。"
    )
```

- [ ] **Step 3：提交**

```bash
git add src/math_agent/prompts/figure_critic.py src/math_agent/prompts/figure_analyst.py
git commit -m "feat: figure critic and analyst prompts"
```

---

### Task 2.4：figure_pipeline 节点（先写测试）

**Files:**
- Create: `tests/nodes/test_figure_pipeline.py`

- [ ] **Step 1：写失败的测试**

```python
from pathlib import Path
from PIL import Image

from math_agent.state import (
    MathModelingState, CodeArtifact, SensitivityRun, FigureArtifact,
)
from math_agent.nodes.figure_pipeline import (
    figure_pipeline_node, FigureCriticOut, FigureAnalysisOut,
)


def _png(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), "white").save(p, dpi=(150, 150))
    return str(p)


def test_pipeline_collects_pngs_from_code_artifacts_and_sensitivity(mocker, workdir):
    p1 = _png(workdir / "code" / "fig_a.png")
    p2 = _png(workdir / "sensitivity" / "lambda.png")

    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.code_artifacts.append(CodeArtifact(
        purpose="主结果", code="...", success=True,
        artifact_paths=[p1, "ignore.txt"],
    ))
    s.sensitivity_runs.append(SensitivityRun(
        parameter="lambda", values=[1, 2], metric="y", results=[1, 2],
        figure_path=p2,
    ))

    critic = FigureCriticOut(score=9, issues=[], suggestions=[], approved=True)
    analysis = FigureAnalysisOut(analysis="图显示 lambda 越大 y 越大，敏感度高。")
    mocker.patch("math_agent.nodes.figure_pipeline.complete",
                 side_effect=[critic, analysis, critic, analysis])

    delta = figure_pipeline_node(s)
    assert len(delta["figures"]) == 2
    paths = {f.path for f in delta["figures"]}
    assert paths == {p1, p2}
    assert all(f.quality_score == 9 for f in delta["figures"])
    assert all("lambda" in f.analysis or "敏感度" in f.analysis for f in delta["figures"])


def test_pipeline_skips_non_png_artifacts(mocker, workdir):
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.code_artifacts.append(CodeArtifact(
        purpose="x", code="...", success=True, artifact_paths=["a.csv", "b.txt"],
    ))
    mocker.patch("math_agent.nodes.figure_pipeline.complete")
    delta = figure_pipeline_node(s)
    assert delta.get("figures", []) == []


def test_pipeline_records_issue_for_low_quality_after_retry(mocker, workdir):
    p1 = _png(workdir / "code" / "x.png")
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.code_artifacts.append(CodeArtifact(
        purpose="x", code="...", success=True, artifact_paths=[p1],
    ))
    bad = FigureCriticOut(score=4, issues=["缺图例"], suggestions=["加图例"], approved=False)
    analysis = FigureAnalysisOut(analysis="尽管质量一般，趋势仍可读出。")
    # 一张图：critic(差) -> critic(再次差) -> analyst(出图说) ：达到重试上限后仍生成解读，但 quality 标低
    mocker.patch("math_agent.nodes.figure_pipeline.complete",
                 side_effect=[bad, bad, analysis])
    delta = figure_pipeline_node(s)
    fig = delta["figures"][0]
    assert fig.quality_score == 4
    assert "缺图例" in fig.quality_issues
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/nodes/test_figure_pipeline.py -v`
Expected: ImportError.

---

### Task 2.5：实现 figure_pipeline 节点

**Files:**
- Create: `src/math_agent/nodes/figure_pipeline.py`

- [ ] **Step 1：写节点**

```python
"""figure_pipeline：扫描 code_artifacts/sensitivity_runs 里的 PNG，
对每张图做 Critic 评分（最多重试 1 次）+ Analyst 写图说。

不重新生成图（重生成的成本/收益不划算）；只评分、解读，
低质量图保留但 quality_score 反映在 Evaluation 中。
"""
from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel

from math_agent.config import MODEL_ROUTING
from math_agent.llm import complete
from math_agent.prompts.figure_critic import (
    SYSTEM as FC_SYSTEM, build_prompt as fc_prompt,
)
from math_agent.prompts.figure_analyst import (
    SYSTEM as FA_SYSTEM, build_prompt as fa_prompt,
)
from math_agent.state import FigureArtifact, MathModelingState
from math_agent.tools.image import inspect_image, encode_image_to_data_url


class FigureCriticOut(BaseModel):
    score: int
    issues: list[str] = []
    suggestions: list[str] = []
    approved: bool = False


class FigureAnalysisOut(BaseModel):
    analysis: str


_MAX_CRITIC_RETRIES = 1  # critic 不通过时，最多再问一次（不重新生成图）


def _collect_pngs(state: MathModelingState) -> list[tuple[str, str, str]]:
    """返回 [(path, purpose, context_text), ...]"""
    out: list[tuple[str, str, str]] = []
    for art in state.code_artifacts:
        if not art.success:
            continue
        for p in art.artifact_paths:
            if p.lower().endswith(".png"):
                out.append((p, art.purpose, art.stdout[:500]))
    for r in state.sensitivity_runs:
        if r.figure_path and r.figure_path.lower().endswith(".png"):
            ctx = f"parameter={r.parameter} values={r.values} {r.metric}={r.results}"
            out.append((r.figure_path, f"敏感性分析: {r.parameter}", ctx))
    return out


def figure_pipeline_node(state: MathModelingState) -> dict:
    figures: list[FigureArtifact] = []
    for path, purpose, context in _collect_pngs(state):
        info = inspect_image(path)
        meta = f"{info.width}x{info.height}px, dpi={info.dpi}"
        url = encode_image_to_data_url(path)

        critic: FigureCriticOut | None = None
        for _ in range(_MAX_CRITIC_RETRIES + 1):
            critic = complete(
                fc_prompt(purpose, meta),
                schema=FigureCriticOut, system=FC_SYSTEM,
                model=MODEL_ROUTING.get("figure_critic", MODEL_ROUTING["model_critic"]),
                images=[url],
            )
            if critic.approved:
                break

        analysis: FigureAnalysisOut = complete(
            fa_prompt(purpose, context),
            schema=FigureAnalysisOut, system=FA_SYSTEM,
            model=MODEL_ROUTING.get("writer"),
            images=[url],
        )

        figures.append(FigureArtifact(
            path=path, purpose=purpose,
            caption=analysis.analysis[:60],   # 简短题注（正文用）
            quality_score=critic.score if critic else 0,
            quality_issues=list(critic.issues) if critic else [],
            analysis=analysis.analysis,
        ))

    return {"figures": figures} if figures else {}
```

- [ ] **Step 2：在 `config.py` 追加模型路由项**

打开 `src/math_agent/config.py`，把 `MODEL_ROUTING` 字典扩成：

```python
MODEL_ROUTING = {
    "analyst": STRONG_MODEL,
    "modeler": STRONG_MODEL,
    "model_critic": STRONG_MODEL,
    "coder": DEFAULT_MODEL,
    "writer": STRONG_MODEL,
    "figure_critic": STRONG_MODEL,   # 多模态
    "paper_critic": STRONG_MODEL,
    "evaluation": STRONG_MODEL,
}
```

- [ ] **Step 3：跑测试**

Run: `pytest tests/nodes/test_figure_pipeline.py -v`
Expected: 3 passed.

- [ ] **Step 4：提交**

```bash
git add src/math_agent/nodes/figure_pipeline.py src/math_agent/config.py tests/nodes/test_figure_pipeline.py
git commit -m "feat: figure pipeline node (critic+analyst on collected pngs)"
```

---

## Phase 3：PaperCritic

### Task 3.1：PaperCritic prompt

**Files:**
- Create: `src/math_agent/prompts/paper_critic.py`

- [ ] **Step 1：写 prompt**

```python
"""PaperCritic：对组装好的论文初稿做整体评审，输出 CriticReport(target='paper')。"""

SYSTEM = (
    "你是国赛资深评委。请审阅一份建模论文初稿。要点："
    "（1）摘要是否凸显方法和结论；（2）假设是否被正文承接；"
    "（3）模型与求解是否一致、可复现；（4）是否有敏感性分析；"
    "（5）图表是否被正文引用并解读；（6）整体行文是否专业。"
    "总评 0-10，>=8 approved。"
)


def build_prompt(paper, n_figures, n_sensitivity):
    sections = {
        "abstract": paper.abstract, "problem_restatement": paper.problem_restatement,
        "assumptions": paper.assumptions, "notation": paper.notation,
        "model_section": paper.model_section, "solution": paper.solution,
        "sensitivity": paper.sensitivity, "conclusion": paper.conclusion,
    }
    body = "\n\n".join(f"## {k}\n{v[:1000]}" for k, v in sections.items())
    return (
        f"# 章节素材\n{body}\n\n"
        f"# 客观信号\n- 图表数：{n_figures}\n- 敏感性 run 数：{n_sensitivity}\n\n"
        f"请输出 JSON：{{\"target\":\"paper\",\"score\":int,\"issues\":[str],"
        f"\"suggestions\":[str],\"approved\":bool}}。"
    )
```

- [ ] **Step 2：提交**

```bash
git add src/math_agent/prompts/paper_critic.py
git commit -m "feat: paper critic prompt"
```

---

### Task 3.2：PaperCritic 节点（先写测试）

**Files:**
- Create: `tests/nodes/test_paper_critic.py`

- [ ] **Step 1：写失败的测试**

```python
from math_agent.state import (
    MathModelingState, PaperSections, FigureArtifact, SensitivityRun, CriticReport,
)
from math_agent.nodes.paper_critic import paper_critic_node


def test_paper_critic_appends_report(mocker):
    fake = CriticReport(target="paper", score=8, issues=[], suggestions=[], approved=True)
    mocker.patch("math_agent.nodes.paper_critic.complete", return_value=fake)
    s = MathModelingState(problem="p")
    s.paper = PaperSections(
        abstract="a"*200, problem_restatement="b"*200, assumptions="c"*200,
        notation="d"*200, model_section="e"*200, solution="f"*200,
        sensitivity="g"*200, conclusion="h"*200, references="-",
    )
    s.figures.append(FigureArtifact(path="x.png", purpose="t"))
    s.sensitivity_runs.append(SensitivityRun(
        parameter="a", values=[1], metric="m", results=[1],
    ))
    delta = paper_critic_node(s)
    assert delta["critic_reports"][0].target == "paper"
    assert delta["critic_reports"][0].approved is True


def test_paper_critic_handles_missing_paper(mocker):
    s = MathModelingState(problem="p")
    delta = paper_critic_node(s)
    assert delta["errors"]
    assert delta.get("critic_reports", []) == []
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/nodes/test_paper_critic.py -v`
Expected: ImportError.

---

### Task 3.3：实现 PaperCritic 节点

**Files:**
- Create: `src/math_agent/nodes/paper_critic.py`

- [ ] **Step 1：写节点**

```python
from math_agent.config import MODEL_ROUTING
from math_agent.llm import complete
from math_agent.prompts.paper_critic import SYSTEM, build_prompt
from math_agent.state import CriticReport, MathModelingState


def paper_critic_node(state: MathModelingState) -> dict:
    p = state.paper
    if not any([p.abstract, p.model_section, p.solution]):
        return {"errors": ["paper_critic: 论文初稿为空，跳过整体评审"]}

    out: CriticReport = complete(
        build_prompt(p, len(state.figures), len(state.sensitivity_runs)),
        schema=CriticReport, system=SYSTEM,
        model=MODEL_ROUTING["paper_critic"],
    )
    out.target = "paper"
    return {"critic_reports": [out]}
```

- [ ] **Step 2：跑测试**

Run: `pytest tests/nodes/test_paper_critic.py -v`
Expected: 2 passed.

- [ ] **Step 3：提交**

```bash
git add src/math_agent/nodes/paper_critic.py tests/nodes/test_paper_critic.py
git commit -m "feat: paper critic node"
```

---

## Phase 4：Evaluation Module（独立打分）

### Task 4.1：Evaluation prompt

**Files:**
- Create: `src/math_agent/prompts/evaluation.py`

- [ ] **Step 1：写 prompt**

```python
"""Evaluation Module：与 PaperCritic 解耦的独立打分，更结构化（对齐国赛四大标准 + 加分项）。"""

SYSTEM = (
    "你是国赛阅卷打分官。请独立、严格地按下列维度打分（每项 0-10，整数）："
    "assumption_reasonableness（假设合理性）、modeling_creativity（建模创造性）、"
    "result_correctness（结果正确性）、writing_clarity（文字清晰度）、"
    "extra_depth（加分项：分析深度/敏感性/创新点）。"
    "overall = round("
    "0.2*assumption_reasonableness + 0.25*modeling_creativity + "
    "0.25*result_correctness + 0.2*writing_clarity + 0.1*extra_depth, 2)。"
    "请认真给出 issues 和 suggestions，但不要重复 PaperCritic 已经说过的内容。"
)


def build_prompt(paper, figures, sensitivity_runs, paper_critic):
    crit_summary = "（无 PaperCritic 报告）"
    if paper_critic:
        crit_summary = (
            f"score={paper_critic.score}; issues={paper_critic.issues[:5]}; "
            f"suggestions={paper_critic.suggestions[:5]}"
        )
    return (
        f"# 论文摘要\n{paper.abstract[:1000]}\n\n"
        f"# 主体（截断）\n模型：{paper.model_section[:800]}\n\n"
        f"求解：{paper.solution[:800]}\n\n敏感性：{paper.sensitivity[:800]}\n\n"
        f"结论：{paper.conclusion[:500]}\n\n"
        f"# 客观信号\n图数={len(figures)}; 平均图质量="
        f"{sum(f.quality_score for f in figures)/max(1,len(figures)):.1f}; "
        f"sensitivity 数={len(sensitivity_runs)}\n\n"
        f"# PaperCritic 摘要\n{crit_summary}\n\n"
        f"请按 schema 输出 JSON。"
    )
```

- [ ] **Step 2：提交**

```bash
git add src/math_agent/prompts/evaluation.py
git commit -m "feat: evaluation prompt"
```

---

### Task 4.2：Evaluation 节点（先写测试）

**Files:**
- Create: `tests/nodes/test_evaluation.py`

- [ ] **Step 1：写失败的测试**

```python
import pytest
from math_agent.state import (
    MathModelingState, PaperSections, FigureArtifact, SensitivityRun,
    CriticReport, EvaluationReport,
)
from math_agent.nodes.evaluation import evaluation_node


def _full_state():
    s = MathModelingState(problem="p")
    s.paper = PaperSections(
        abstract="a"*200, problem_restatement="x"*200, assumptions="x"*200,
        notation="x"*200, model_section="x"*200, solution="x"*200,
        sensitivity="x"*200, conclusion="x"*200, references="-",
    )
    s.figures.append(FigureArtifact(path="a.png", purpose="t", quality_score=8))
    s.sensitivity_runs.append(SensitivityRun(parameter="a", values=[1], metric="m", results=[1]))
    s.critic_reports.append(CriticReport(target="paper", score=8, approved=True))
    return s


def test_evaluation_returns_report(mocker):
    fake = EvaluationReport(
        assumption_reasonableness=8, modeling_creativity=8,
        result_correctness=8, writing_clarity=8, extra_depth=8, overall=8.0,
        issues=[], suggestions=[],
    )
    mocker.patch("math_agent.nodes.evaluation.complete", return_value=fake)
    delta = evaluation_node(_full_state())
    assert isinstance(delta["evaluation"], EvaluationReport)
    assert delta["evaluation"].overall == 8.0


def test_evaluation_recomputes_overall_if_llm_wrong(mocker):
    # 给出 overall=10（与权重不符），节点应当用规则自校正
    fake = EvaluationReport(
        assumption_reasonableness=8, modeling_creativity=8,
        result_correctness=8, writing_clarity=8, extra_depth=8, overall=10.0,
    )
    mocker.patch("math_agent.nodes.evaluation.complete", return_value=fake)
    delta = evaluation_node(_full_state())
    assert delta["evaluation"].overall == pytest.approx(8.0, abs=0.01)


def test_evaluation_skips_without_paper(mocker):
    s = MathModelingState(problem="p")
    delta = evaluation_node(s)
    assert delta["errors"]
    assert delta.get("evaluation") is None
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/nodes/test_evaluation.py -v`
Expected: ImportError.

---

### Task 4.3：实现 Evaluation 节点

**Files:**
- Create: `src/math_agent/nodes/evaluation.py`

- [ ] **Step 1：写节点**

```python
"""Evaluation Module：与 PaperCritic 解耦的独立量化打分。

为避免 LLM 在 overall 上算错，节点最终用确定性公式重算 overall。
"""
from __future__ import annotations

from math_agent.config import MODEL_ROUTING
from math_agent.llm import complete
from math_agent.prompts.evaluation import SYSTEM, build_prompt
from math_agent.state import EvaluationReport, MathModelingState


_WEIGHTS = {
    "assumption_reasonableness": 0.20,
    "modeling_creativity": 0.25,
    "result_correctness": 0.25,
    "writing_clarity": 0.20,
    "extra_depth": 0.10,
}


def _compute_overall(r: EvaluationReport) -> float:
    total = sum(getattr(r, k) * w for k, w in _WEIGHTS.items())
    return round(total, 2)


def evaluation_node(state: MathModelingState) -> dict:
    p = state.paper
    if not any([p.abstract, p.model_section, p.solution]):
        return {"errors": ["evaluation: 论文初稿为空，跳过评估"]}

    paper_critic = state.latest_critic("paper")
    out: EvaluationReport = complete(
        build_prompt(p, state.figures, state.sensitivity_runs, paper_critic),
        schema=EvaluationReport, system=SYSTEM,
        model=MODEL_ROUTING["evaluation"],
    )
    out.overall = _compute_overall(out)  # 确定性自校正
    return {"evaluation": out}
```

- [ ] **Step 2：跑测试**

Run: `pytest tests/nodes/test_evaluation.py -v`
Expected: 3 passed.

- [ ] **Step 3：提交**

```bash
git add src/math_agent/nodes/evaluation.py tests/nodes/test_evaluation.py
git commit -m "feat: evaluation node with deterministic overall recompute"
```

---

## Phase 5：Human-in-the-loop + Checkpointer

> 设计要点：HITL 通过 LangGraph 的 `interrupt_before=["human_review"]` 实现。`human_review_node` 本身只做"把 `state.human_decision` 写回 + 校验"；真正的"暂停-恢复"由 checkpointer 完成。CLI 与测试中可注入预设决定，便于自动化测试。

### Task 5.1：human_review 节点（先写测试）

**Files:**
- Create: `tests/nodes/test_human_review.py`

- [ ] **Step 1：写失败的测试**

```python
from math_agent.state import MathModelingState, HumanDecision
from math_agent.nodes.human_review import human_review_node


def test_human_review_passes_through_existing_decision():
    s = MathModelingState(problem="p")
    s.human_decision = HumanDecision(approved=True, notes="looks good")
    delta = human_review_node(s)
    # 节点不能擅自改 decision；它只把现有 decision 透传，并标记 reviewed
    assert delta == {}


def test_human_review_records_error_when_no_decision():
    s = MathModelingState(problem="p")
    delta = human_review_node(s)
    # 当 graph 通过 interrupt 暂停又被恢复但 state 没填 decision，应当记录 error 而非崩溃
    assert delta["errors"]
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/nodes/test_human_review.py -v`
Expected: ImportError.

---

### Task 5.2：实现 human_review 节点

**Files:**
- Create: `src/math_agent/nodes/human_review.py`

- [ ] **Step 1：写节点**

```python
"""human_review 节点：被 graph 在 interrupt 之后运行；
预期 state.human_decision 已被外部填充。
"""
from math_agent.state import MathModelingState


def human_review_node(state: MathModelingState) -> dict:
    if state.human_decision is None:
        return {"errors": ["human_review: 恢复后未发现 human_decision，请填入后再恢复"]}
    return {}
```

- [ ] **Step 2：跑测试**

Run: `pytest tests/nodes/test_human_review.py -v`
Expected: 2 passed.

- [ ] **Step 3：提交**

```bash
git add src/math_agent/nodes/human_review.py tests/nodes/test_human_review.py
git commit -m "feat: human review node (interrupt-based)"
```

---

### Task 5.3：把 Checkpointer + interrupt 接入 graph（仅 graph 改动；连线在 Phase 7 一起做）

**Files:**
- Modify: `src/math_agent/graph.py`

- [ ] **Step 1：在 `build_graph()` 增加可选 checkpointer 参数**

替换 `build_graph` 签名为：

```python
def build_graph(
    *,
    checkpointer=None,
    interrupt_before: list[str] | None = None,
):
```

并把最后一行的 `g.compile()` 改成：

```python
    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or [],
    )
```

> 在 Phase 7，CLI 会传入 `SqliteSaver` 与 `interrupt_before=["human_review"]`。

- [ ] **Step 2：跑现有 smoke 测试，确保签名变更向后兼容**

Run: `pytest tests/test_graph_smoke.py -v`
Expected: 全绿（默认 checkpointer=None / interrupt_before=None，行为不变）。

- [ ] **Step 3：提交**

```bash
git add src/math_agent/graph.py
git commit -m "feat(graph): optional checkpointer and interrupt_before"
```

---

## Phase 6：LaTeX 渲染 + 编译 + 回退

### Task 6.1：xelatex 子进程封装

**Files:**
- Create: `tests/tools/test_latex_compile.py`
- Create: `src/math_agent/tools/latex_compile.py`

- [ ] **Step 1：写失败的测试**

```python
import shutil
import pytest
from pathlib import Path

from math_agent.tools.latex_compile import compile_latex, LatexResult


HAS_XELATEX = shutil.which("xelatex") is not None


@pytest.mark.skipif(not HAS_XELATEX, reason="xelatex not installed")
def test_compile_latex_minimal(workdir):
    tex = workdir / "main.tex"
    tex.write_text(r"""
\documentclass{article}
\begin{document}
hello
\end{document}
""", encoding="utf-8")
    res = compile_latex(tex)
    assert res.success
    assert Path(res.pdf_path).exists()


def test_compile_latex_returns_failure_when_xelatex_missing(monkeypatch, workdir, tmp_path):
    monkeypatch.setattr("shutil.which", lambda _: None)
    tex = workdir / "main.tex"
    tex.write_text(r"\documentclass{article}\begin{document}x\end{document}", encoding="utf-8")
    res = compile_latex(tex)
    assert not res.success
    assert "xelatex" in res.log.lower()
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/tools/test_latex_compile.py -v`
Expected: ImportError.

- [ ] **Step 3：实现**

```python
"""xelatex 子进程封装。失败时返回结构化结果，不抛异常。"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LatexResult:
    success: bool
    pdf_path: str = ""
    log: str = ""


def compile_latex(tex_path: str | Path, *, timeout: int = 120) -> LatexResult:
    tex_path = Path(tex_path)
    if shutil.which("xelatex") is None:
        return LatexResult(success=False, log="xelatex not found on PATH")

    workdir = tex_path.parent
    try:
        # 跑两遍以解决交叉引用 / TOC
        log_acc = []
        for _ in range(2):
            proc = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                cwd=workdir, capture_output=True, text=True, timeout=timeout,
            )
            log_acc.append(proc.stdout + "\n" + proc.stderr)
            if proc.returncode != 0:
                return LatexResult(success=False, log="\n".join(log_acc))
        pdf = workdir / (tex_path.stem + ".pdf")
        if not pdf.exists():
            return LatexResult(success=False, log="\n".join(log_acc) + "\nno pdf produced")
        return LatexResult(success=True, pdf_path=str(pdf), log="\n".join(log_acc))
    except subprocess.TimeoutExpired as e:
        return LatexResult(success=False, log=f"timeout after {timeout}s: {e}")
```

- [ ] **Step 4：跑测试**

Run: `pytest tests/tools/test_latex_compile.py -v`
Expected: 2 passed（xelatex 未装时第一个被 skip，第二个仍 pass）。

- [ ] **Step 5：提交**

```bash
git add src/math_agent/tools/latex_compile.py tests/tools/test_latex_compile.py
git commit -m "feat: xelatex compile wrapper (graceful failure)"
```

---

### Task 6.2：LaTeX 模板

**Files:**
- Create: `src/math_agent/templates/paper.tex.j2`

- [ ] **Step 1：写模板**

```jinja
\documentclass[12pt,a4paper]{article}
\usepackage{xeCJK}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{margin=2.5cm}
\setCJKmainfont{SimSun}  % 系统无 SimSun 时由编译失败回退到 Markdown
\title{ {{- problem | e -}} }
\author{Math Agent}
\date{}
\begin{document}
\maketitle

\begin{abstract}
{{ paper.abstract }}
\end{abstract}

\section{问题重述}
{{ paper.problem_restatement }}

\section{模型假设}
{{ paper.assumptions }}

\section{符号说明}
{{ paper.notation }}

\section{模型的建立与演化}
{{ paper.model_section }}

\section{模型的求解}
{{ paper.solution }}

{% if figures %}
\subsection{关键图表}
{% for fig in figures %}
\begin{figure}[h]
\centering
\includegraphics[width=0.75\linewidth]{ {{- fig.path -}} }
\caption{ {{- fig.caption | default(fig.purpose) -}} }
\end{figure}

{{ fig.analysis }}

{% endfor %}
{% endif %}

\section{敏感性分析}
{{ paper.sensitivity }}

{% for r in sensitivity_runs %}
\paragraph{ {{- r.parameter -}} }
{{ r.interpretation }}

{% endfor %}

\section{模型评价与结论}
{{ paper.conclusion }}

\section*{参考文献}
{{ paper.references }}

\end{document}
```

- [ ] **Step 2：提交**

```bash
git add src/math_agent/templates/paper.tex.j2
git commit -m "feat: latex paper template (xeCJK)"
```

---

### Task 6.3：latex 节点（render + compile + fallback）

**Files:**
- Create: `tests/nodes/test_latex.py`
- Create: `src/math_agent/nodes/latex.py`

- [ ] **Step 1：写失败的测试**

```python
from pathlib import Path

from math_agent.state import (
    MathModelingState, PaperSections, FigureArtifact, SensitivityRun, CodeArtifact,
)
from math_agent.nodes.latex import latex_node


def _state(workdir):
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.paper = PaperSections(
        abstract="a"*100, problem_restatement="b"*100, assumptions="c"*100,
        notation="d"*100, model_section="e"*100, solution="f"*100,
        sensitivity="g"*100, conclusion="h"*100, references="-",
    )
    return s


def test_latex_node_writes_tex_and_markdown_fallback(mocker, workdir):
    mocker.patch(
        "math_agent.nodes.latex.compile_latex",
        return_value=type("R", (), {"success": False, "pdf_path": "", "log": "no xelatex"})(),
    )
    s = _state(workdir)
    delta = latex_node(s)
    assert (workdir / "paper.tex").exists()
    assert (workdir / "paper.md").exists()
    assert delta["errors"]
    assert "no xelatex" in delta["errors"][0]


def test_latex_node_records_pdf_path_on_success(mocker, workdir):
    pdf = workdir / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    mocker.patch(
        "math_agent.nodes.latex.compile_latex",
        return_value=type("R", (), {"success": True, "pdf_path": str(pdf), "log": ""})(),
    )
    s = _state(workdir)
    delta = latex_node(s)
    assert (workdir / "paper.tex").exists()
    assert delta == {} or "errors" not in delta
```

- [ ] **Step 2：实现节点**

```python
"""latex 节点：渲染 .tex → 编译 .pdf → 失败时回退到 Markdown。"""
from __future__ import annotations

from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

from math_agent.nodes.writer import render_markdown
from math_agent.state import MathModelingState
from math_agent.tools.latex_compile import compile_latex


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=select_autoescape([]))


def latex_node(state: MathModelingState) -> dict:
    workdir = Path(state.output_dir or ".")
    workdir.mkdir(parents=True, exist_ok=True)

    tmpl = _env.get_template("paper.tex.j2")
    tex = tmpl.render(
        problem=state.problem, paper=state.paper,
        figures=state.figures, sensitivity_runs=state.sensitivity_runs,
    )
    tex_path = workdir / "paper.tex"
    tex_path.write_text(tex, encoding="utf-8")

    # 始终也写一份 Markdown，作为降级 / 备查
    (workdir / "paper.md").write_text(render_markdown(state), encoding="utf-8")

    res = compile_latex(tex_path)
    if not res.success:
        return {"errors": [f"latex compile failed: {res.log[:500]}"]}
    return {}
```

- [ ] **Step 3：跑测试**

Run: `pytest tests/nodes/test_latex.py -v`
Expected: 2 passed.

- [ ] **Step 4：提交**

```bash
git add src/math_agent/nodes/latex.py tests/nodes/test_latex.py
git commit -m "feat: latex node with markdown fallback"
```

---

## Phase 7：串接图 + 端到端 smoke

### Task 7.1：扩展 build_graph 串接新节点

**Files:**
- Modify: `src/math_agent/graph.py`

> Plan A 已确立"`MathModelingState`（Pydantic）直接作为 LangGraph state schema、
> 节点签名 `(MathModelingState) -> dict` 无 `_wrap`"。Plan B 在 Phase 1.2 把新字段
> 直接加在 `MathModelingState` 上并标注 `Annotated[..., add]`，本任务只追加 import
> 与节点接线，**不**重新引入 `GraphState` TypedDict 或 `_wrap`。

- [ ] **Step 1：扩展 imports**

在文件顶部 import 区追加：

```python
from math_agent.nodes.sensitivity import sensitivity_node
from math_agent.nodes.figure_pipeline import figure_pipeline_node
from math_agent.nodes.paper_critic import paper_critic_node
from math_agent.nodes.evaluation import evaluation_node
from math_agent.nodes.human_review import human_review_node
from math_agent.nodes.latex import latex_node
```

> 不需要 import `SensitivityRun/FigureArtifact/EvaluationReport/HumanDecision` —— 它们
> 已经是 `MathModelingState` 的字段，由 state schema 自带。

- [ ] **Step 2：把 Plan A 中 `coder -> writer -> finalize -> END` 的路径替换为完整链路**

把 `build_graph` 函数体里 `g.add_node("coder", coder_node)` 之后到 `return g.compile(...)` 之前的部分替换为：

```python
    g.add_node("coder", coder_node)
    g.add_node("sensitivity", sensitivity_node)
    g.add_node("figure_pipeline", figure_pipeline_node)
    g.add_node("writer", writer_node)
    g.add_node("paper_critic", paper_critic_node)
    g.add_node("evaluation", evaluation_node)
    g.add_node("human_review", human_review_node)
    g.add_node("latex", latex_node)

    g.add_edge("coder", "sensitivity")
    g.add_edge("sensitivity", "figure_pipeline")
    g.add_edge("figure_pipeline", "writer")
    g.add_edge("writer", "paper_critic")
    g.add_edge("paper_critic", "evaluation")
    g.add_edge("evaluation", "human_review")
    g.add_edge("human_review", "latex")
    g.add_edge("latex", END)
```

把 Plan A 中保留的 `finalize` 节点删除（它的工作已由 `latex_node` 完成 —— Markdown 也由 latex_node 写）。

- [ ] **Step 3：跑 Plan A 的 smoke 测试，确保未破坏**

Run: `pytest tests/test_graph_smoke.py -v`
Expected: 旧测试中"end-to-end 写 paper.md"的那个测试需要更新 mock 集合。在该测试里加入对 sensitivity / figure_pipeline / paper_critic / evaluation / human_review / latex 的 mock，或直接删掉旧用例（在 Task 7.2 用新 smoke 替代）。

最直接的做法：把 Plan A 中的 `test_graph_writes_paper_md` 改名为旧版语义，并 `@pytest.mark.skip("replaced by Plan B end-to-end smoke")`。

```python
import pytest

@pytest.mark.skip("replaced by tests/test_graph_full_smoke.py in Plan B")
def test_graph_writes_paper_md(): ...
```

- [ ] **Step 4：提交**

```bash
git add src/math_agent/graph.py tests/test_graph_smoke.py
git commit -m "feat(graph): wire sensitivity/figure/critic/eval/human/latex into pipeline"
```

---

### Task 7.2：端到端 smoke（含 HITL）

**Files:**
- Create: `tests/test_graph_full_smoke.py`

- [ ] **Step 1：写测试**

```python
"""端到端 smoke：mock 掉所有 LLM 与 LaTeX 编译，验证：
1) 节点按顺序被命中；
2) interrupt_before=['human_review'] 时 graph 暂停；
3) 注入 human_decision 后恢复，能产出 paper.tex + paper.md。
"""
from pathlib import Path
import pytest
from PIL import Image

from math_agent.graph import build_graph
from math_agent.state import (
    Assumption, ModelVersion, CriticReport, PaperSections,
    SensitivityRun, FigureArtifact, EvaluationReport, HumanDecision,
)
from math_agent.nodes.analyst import AnalystOutput
from math_agent.nodes.coder import CoderDraft
from math_agent.nodes.figure_pipeline import FigureCriticOut, FigureAnalysisOut


def _png(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 240), "white").save(p)
    return str(p)


def _setup_all_mocks(mocker, workdir):
    mocker.patch("math_agent.nodes.analyst.complete",
                 return_value=AnalystOutput(assumptions=[
                     Assumption(statement="A", rationale="r", sensitivity_relevant=True)]))

    stage_iter = iter(["basic", "improved", "final"])
    mocker.patch("math_agent.nodes.modeler.complete",
                 side_effect=lambda *a, **k: ModelVersion(stage=next(stage_iter), description="d"*200))
    mocker.patch("math_agent.nodes.model_critic.complete",
                 return_value=CriticReport(target="modeler", score=9, approved=True))

    # coder 生成一张图
    fig_path = _png(workdir / "code" / "fig.png")
    code = (
        "from pathlib import Path\n"
        f"Path(r'{fig_path}').parent.mkdir(parents=True, exist_ok=True)\n"
        "print('coder done')\n"
    )
    mocker.patch("math_agent.nodes.coder.complete",
                 return_value=CoderDraft(purpose="主结果", code=code))

    # sensitivity 三段
    from math_agent.nodes.sensitivity import SensitivityPlan, SensitivityCode, Interpretations
    sens_png = workdir / "sensitivity" / "lambda.png"
    sens_png.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 240), "white").save(sens_png)
    sens_plan = SensitivityPlan(runs=[{"parameter": "lambda", "values": [1,2,3,4,5], "metric": "y", "rationale": "r"}])
    sens_code = SensitivityCode(code=(
        "import matplotlib.pyplot as plt\n"
        "v=[1,2,3,4,5]; r=[x*2 for x in v]\n"
        "plt.plot(v,r); plt.savefig('lambda.png')\n"
        "print(f'RESULT: parameter=lambda values={v} results={r}')\n"
    ))
    sens_interp = Interpretations(interpretations=["lambda 越大 y 线性增长，敏感度中等。"])
    mocker.patch("math_agent.nodes.sensitivity.complete",
                 side_effect=[sens_plan, sens_code, sens_interp])

    # figure pipeline：coder 图 + sensitivity 图 = 2 张
    fc = FigureCriticOut(score=9, approved=True)
    fa = FigureAnalysisOut(analysis="趋势单调，敏感度中等。")
    mocker.patch("math_agent.nodes.figure_pipeline.complete",
                 side_effect=[fc, fa, fc, fa])

    # writer / paper_critic / evaluation
    mocker.patch("math_agent.nodes.writer.complete",
                 return_value=PaperSections(
                     abstract="a"*200, problem_restatement="x"*200, assumptions="x"*200,
                     notation="x"*200, model_section="x"*200, solution="x"*200,
                     sensitivity="x"*200, conclusion="x"*200, references="-",
                 ))
    mocker.patch("math_agent.nodes.paper_critic.complete",
                 return_value=CriticReport(target="paper", score=9, approved=True))
    mocker.patch("math_agent.nodes.evaluation.complete",
                 return_value=EvaluationReport(
                     assumption_reasonableness=8, modeling_creativity=8,
                     result_correctness=8, writing_clarity=8, extra_depth=8, overall=8.0,
                 ))

    # latex compile —— 不依赖系统 xelatex
    mocker.patch("math_agent.nodes.latex.compile_latex",
                 return_value=type("R",(object,),{"success": True, "pdf_path":"", "log":""})())


def test_full_pipeline_with_hitl_interrupt_and_resume(mocker, workdir, tmp_path):
    from langgraph.checkpoint.memory import MemorySaver  # 测试用内存 checkpointer
    _setup_all_mocks(mocker, workdir)

    saver = MemorySaver()
    g = build_graph(checkpointer=saver, interrupt_before=["human_review"])

    config = {"configurable": {"thread_id": "t1"}}
    initial = {
        "problem": "p", "stage_target": "basic", "iteration": 0,
        "output_dir": str(workdir),
    }
    # 第一次 invoke：流到 human_review 之前应当被中断
    g.invoke(initial, config=config)
    snap = saver.get(config)
    assert snap is not None
    # 此时论文 latex 还没生成
    assert not (workdir / "paper.tex").exists()

    # 注入决定，再恢复
    g.update_state(config, {"human_decision": HumanDecision(approved=True, notes="ok")})
    g.invoke(None, config=config)

    assert (workdir / "paper.tex").exists()
    assert (workdir / "paper.md").exists()
```

- [ ] **Step 2：跑测试**

Run: `pytest tests/test_graph_full_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 3：提交**

```bash
git add tests/test_graph_full_smoke.py
git commit -m "test: end-to-end smoke with HITL interrupt and resume"
```

---

### Task 7.3：CLI 接入 HITL（resume 子命令）

**Files:**
- Modify: `src/math_agent/cli.py`

- [ ] **Step 1：在 `cli.py` 中重写**

```python
"""math-agent CLI（Plan B 版）。

run  : 启动一次任务（默认在 human_review 处中断）
resume: 提供 human decision 并续跑
"""
from __future__ import annotations
import json
from pathlib import Path

import typer
from langgraph.checkpoint.sqlite import SqliteSaver

from math_agent.graph import build_graph
from math_agent.state import HumanDecision


app = typer.Typer(help="Math modeling multi-agent system.")


def _saver(out: Path) -> SqliteSaver:
    out.mkdir(parents=True, exist_ok=True)
    return SqliteSaver.from_conn_string(str(out / "checkpoints.sqlite"))


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@app.command()
def run(
    problem: Path = typer.Option(..., exists=True, readable=True),
    out: Path = typer.Option(Path("runs/latest")),
    thread: str = typer.Option("default"),
    no_interrupt: bool = typer.Option(False, help="跳过 HITL，直接跑到底"),
):
    spec = json.loads(problem.read_text(encoding="utf-8"))
    saver = _saver(out)
    interrupt = [] if no_interrupt else ["human_review"]
    g = build_graph(checkpointer=saver, interrupt_before=interrupt)

    initial = {
        "problem": spec.get("title", "") + "\n" + "\n".join(spec.get("questions", [])),
        "background": spec.get("background", ""),
        "questions": spec.get("questions", []),
        "stage_target": "basic",
        "iteration": 0,
        "output_dir": str(out),
    }
    g.invoke(initial, config=_config(thread))
    typer.echo(f"pipeline paused before human_review (thread={thread}).")
    typer.echo("use `math-agent resume --thread {thread} --approve` to continue.")


@app.command()
def resume(
    out: Path = typer.Option(Path("runs/latest")),
    thread: str = typer.Option("default"),
    approve: bool = typer.Option(True),
    notes: str = typer.Option(""),
):
    saver = _saver(out)
    g = build_graph(checkpointer=saver, interrupt_before=["human_review"])
    g.update_state(_config(thread),
                   {"human_decision": HumanDecision(approved=approve, notes=notes)})
    g.invoke(None, config=_config(thread))
    typer.echo(f"done. tex/md written to {out}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 2：手测**（依赖真实 API key 与本地 xelatex；可在 CI 中跳过）

Run:
```bash
math-agent run --problem tests/fixtures/sample_problem.json --out runs/planb-smoke
math-agent resume --out runs/planb-smoke --approve --notes "lgtm"
ls runs/planb-smoke/
```
Expected: `paper.tex` + `paper.md`（若装了 xelatex 还会有 `paper.pdf`）。

- [ ] **Step 3：提交**

```bash
git add src/math_agent/cli.py
git commit -m "feat(cli): run/resume with sqlite checkpointer and HITL"
```

---

### Task 7.4：更新 README

**Files:**
- Modify: `README.md`

- [ ] **Step 1：把 README 的"下一步"段落替换为 Plan B 已完成项 + Plan C roadmap**

把原本的"下一步（Plan B/C）"段落替换为：

```markdown
## 已完成（Plan B）

- 强制 **Sensitivity** 节点：plan→sweep→interpret 三段式
- **Figure pipeline**：扫描 PNG → 多模态 FigureCritic 评分 → FigureAnalyst 写图说
- **PaperCritic** + 独立 **Evaluation Module**（确定性 overall）
- **Human-in-the-loop**：`interrupt_before=["human_review"]` + `SqliteSaver`
- **LaTeX** 渲染 + `xelatex` 编译；失败回退 Markdown

## 用法（含 HITL）

```bash
math-agent run --problem tests/fixtures/sample_problem.json --out runs/demo
# 流水线在 human_review 前停下；查看 runs/demo/checkpoints.sqlite 与中间产物
math-agent resume --out runs/demo --approve --notes "ok"
# 或一次跑到底：
math-agent run --problem ... --out runs/demo2 --no-interrupt
```

## 下一步（Plan C）

- 统一的 LLM/沙箱/编译错误重试与限流
- 历年国一题回归基准
- RAG（历年论文 / 经典模型库）
- LangSmith / OTel 链路追踪
```

- [ ] **Step 2：提交**

```bash
git add README.md
git commit -m "docs: plan B usage + plan C roadmap"
```

---

## 自我审查

**1. Spec 覆盖**

| 方案要点 | 覆盖位置 |
|---|---|
| 敏感性分析作为必经节点 | Phase 1，graph 中 `coder → sensitivity → figure_pipeline`，无条件边可绕过 ✓ |
| 图表生成 + 专业分析闭环 | Phase 2，`figure_pipeline_node`（critic + analyst，重试 1 次）✓ |
| ModelCritic / PaperCritic / Figure Critic / Evaluation Module 多层评审 | ModelCritic 已在 Plan A；其余三个在 Phase 2/3/4 ✓ |
| 独立 Evaluation Module（对齐国一四大标准 + 加分项） | Phase 4，确定性 overall 自校正 ✓ |
| Human-in-the-loop（最终把关） | Phase 5/7，`interrupt_before=["human_review"]` + `SqliteSaver` ✓ |
| LaTeX 生成 + xelatex 编译 + Markdown 回退 | Phase 6，`latex_node` 始终也写 Markdown ✓ |
| 错误捕获与降级 | sensitivity/coder/latex 均结构化返回 errors，不抛 ✓ |
| 鲁棒性（统一重试 / 限流）/ RAG / 评分基准 | 显式声明落到 Plan C，本计划范围外 ✓ |

**2. Placeholder 扫描**

无 TODO / TBD / "实现 X 即可" 等占位。

**3. 类型一致性**

- `MathModelingState` 是 LangGraph 的 state schema（Plan A 已确立）；Plan B 新增字段
  `sensitivity_runs / figures` 用 `Annotated[list[...], add]` 标记追加语义，
  `evaluation / human_decision` 为覆盖语义。节点签名统一 `(MathModelingState) -> dict`，
  无 `_wrap` 适配层、无独立 `GraphState` TypedDict。
- `PaperSections.sensitivity` 字段在 Plan B 加回（Plan A 已删），Writer prompt / 模板
  对应章节同步补回，保持渲染端到端一致。
- `SensitivityRun` 在 Phase 1.2 定义为 `parameter:str, values:list[float], metric:str, results:list[float], interpretation:str, figure_path:str|None`；Phase 1.4 节点测试、Phase 2.4 `figure_pipeline` 中读取 `r.figure_path / r.parameter / r.values / r.metric / r.results`，签名一致 ✓
- `FigureArtifact` 字段：`path/purpose/caption/quality_score/quality_issues/analysis`；`figure_pipeline_node` 与模板 `paper.tex.j2`（用 `fig.path / fig.caption / fig.purpose / fig.analysis`）使用一致 ✓
- `EvaluationReport` 字段 5 个维度 + overall，权重计算在 `evaluation_node._compute_overall` 中与 prompt 里的描述一致（0.20/0.25/0.25/0.20/0.10）✓
- `HumanDecision` 字段 `approved/notes`；CLI `resume` 与测试 `test_full_pipeline_with_hitl_interrupt_and_resume` 使用一致 ✓
- `LatexResult` 字段 `success/pdf_path/log`；`latex_node` 与 `test_latex.py` 中使用一致 ✓
- LLM `complete()` 新增 `images` 参数：Phase 2.2 在 `llm.py` 与 `figure_pipeline_node` 中签名一致 ✓
- graph 中 `interrupt_before=["human_review"]` 与节点名 `"human_review"` 一致 ✓
- `tools/runner.py`（Plan A 已重命名）在 Phase 1.5 sensitivity_node 中正确 import；
  不再残留 `tools.sandbox` 引用。

无类型/命名漂移。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-28-math-agent-plan-b.md`. Two execution options:

**1. Subagent-Driven (recommended)** — 每个 Task 派一个全新 subagent 执行，我在任务间审查，迭代快，状态干净。

**2. Inline Execution** — 我在当前会话里按 executing-plans 流程逐 Task 执行，带 checkpoint 让你审查。

哪种方式？或者你想先**调整计划**（例如：Sensitivity 失败时改为允许写入空 run、Figure pipeline 允许重新生成图、LaTeX 改用 tectonic、HITL 加第二个断点在"模型 final 确认"），告诉我即可。
