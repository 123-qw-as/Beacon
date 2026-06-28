# 数学建模多智能体系统 MVP 实现计划（Plan A）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 LiteLLM + LangGraph 跑通一条"问题 → 假设 → 渐进式建模（含 Critic 循环）→ 代码执行 → 论文初稿（Markdown）"的端到端最小闭环，作为后续 Sensitivity / Figure / LaTeX / Evaluation 等子系统的承载骨架。

**Architecture:** LangGraph StateGraph 驱动多节点协作；LiteLLM 作为唯一 LLM 出口，统一结构化输出 + 重试 + 模型路由；State 用 Pydantic 模型集中管理；代码执行通过本地 subprocess 沙箱（受限超时 + 工作目录隔离）；输出阶段 MVP 先产 Markdown，LaTeX/PDF 留给 Plan B。

**Tech Stack:**
- Python 3.11+
- `langgraph` (>=0.2)
- `litellm` (>=1.40)
- `pydantic` (>=2.7)
- `jinja2`（模板渲染，MVP 先用 Markdown 模板）
- `pytest`, `pytest-mock`, `pytest-asyncio`
- `matplotlib`（沙箱可调用）
- `typer`（CLI）

---

## 文件结构

新建的项目骨架（所有路径相对 `build-agent/`）：

```
build-agent/
├── pyproject.toml                  # 项目元数据 + 依赖
├── .env.example                    # 环境变量模板（OPENAI_API_KEY 等）
├── .gitignore
├── README.md
├── src/
│   └── math_agent/
│       ├── __init__.py
│       ├── config.py               # 模型路由、运行参数、路径常量
│       ├── llm.py                  # LiteLLM 统一封装（结构化输出 + 重试）
│       ├── state.py                # MathModelingState（Pydantic）
│       ├── routing.py              # 图的条件边路由函数
│       ├── graph.py                # build_graph()：装配 StateGraph
│       ├── cli.py                  # Typer CLI 入口
│       ├── prompts/
│       │   ├── __init__.py
│       │   ├── analyst.py
│       │   ├── modeler.py
│       │   ├── model_critic.py
│       │   ├── coder.py
│       │   └── writer.py
│       ├── nodes/
│       │   ├── __init__.py
│       │   ├── analyst.py
│       │   ├── modeler.py
│       │   ├── model_critic.py
│       │   ├── coder.py
│       │   └── writer.py
│       ├── tools/
│       │   ├── __init__.py
│       │   └── runner.py            # subprocess 代码执行器（非隔离沙箱，仅本机可信使用）
│       └── templates/
│           └── paper.md.j2         # Markdown 论文模板
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   └── sample_problem.json
│   ├── test_state.py
│   ├── test_llm.py
│   ├── test_routing.py
│   ├── test_runner.py
│   ├── nodes/
│   │   ├── test_analyst.py
│   │   ├── test_modeler.py
│   │   ├── test_model_critic.py
│   │   ├── test_coder.py
│   │   └── test_writer.py
│   └── test_graph_smoke.py
└── docs/superpowers/plans/2026-06-28-math-agent-mvp.md  # 本文件
```

职责划分原则：
- `llm.py` 是唯一与 LiteLLM 交互的入口，节点不直接 import litellm。
- `state.py` 是单一真理来源，所有节点签名都是 `(state: MathModelingState) -> dict`，只返回 *增量*。
- `prompts/` 与 `nodes/` 一一对应；prompt 是纯字符串/函数，不调用 LLM。
- `routing.py` 集中放条件边函数，便于单元测试。
- `tools/runner.py` 与节点解耦，可被 Coder 之外的未来节点复用。
  **注意：这不是隔离沙箱**，只是受限超时 + 临时工作目录 + 清环境变量的 subprocess 包装。
  适用于本机可信使用（开发者亲自跑），不适合托管多租户。真正的隔离（docker/firejail）放 Plan C。

---

## Phase 0：项目骨架

### Task 0.1：初始化 pyproject 与依赖

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `README.md`

- [ ] **Step 1：写 `pyproject.toml`**

```toml
[project]
name = "math-agent"
version = "0.1.0"
description = "Multi-agent system for math modeling competition papers"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2.0",
    "litellm>=1.40.0",
    "pydantic>=2.7",
    "jinja2>=3.1",
    "typer>=0.12",
    "python-dotenv>=1.0",
    "matplotlib>=3.8",
    "numpy>=1.26",
    "scipy>=1.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.12",
]

[project.scripts]
math-agent = "math_agent.cli:app"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2：写 `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.env
.pytest_cache/
dist/
build/
*.egg-info/
runs/
```

- [ ] **Step 3：写 `.env.example`**

```
# LiteLLM 路由的任何提供商均可
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=
# 默认模型（可被 config.py 覆盖）
MATH_AGENT_DEFAULT_MODEL=gpt-4o-mini
MATH_AGENT_STRONG_MODEL=gpt-4o
```

- [ ] **Step 4：写最小 `README.md`**

```markdown
# math-agent

LangGraph + LiteLLM 多智能体数学建模助手（MVP）。

## 安装

```bash
pip install -e ".[dev]"
cp .env.example .env  # 填入你的 API key
```

## 运行

```bash
math-agent run --problem tests/fixtures/sample_problem.json
```
```

- [ ] **Step 5：建立空包目录与 `__init__.py`**

```bash
mkdir -p src/math_agent/prompts src/math_agent/nodes src/math_agent/tools src/math_agent/templates
mkdir -p tests/nodes tests/fixtures
touch src/math_agent/__init__.py src/math_agent/prompts/__init__.py src/math_agent/nodes/__init__.py src/math_agent/tools/__init__.py
touch tests/__init__.py tests/nodes/__init__.py
```

- [ ] **Step 6：安装并验证 import**

Run:
```bash
pip install -e ".[dev]"
python -c "import math_agent; import langgraph; import litellm; print('ok')"
```
Expected: `ok`

- [ ] **Step 7：提交**

```bash
git init
git add .
git commit -m "chore: bootstrap math-agent project skeleton"
```

---

### Task 0.2：写测试夹具与 conftest

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/fixtures/sample_problem.json`

- [ ] **Step 1：写 `tests/fixtures/sample_problem.json`**

```json
{
  "title": "城市共享单车调度优化",
  "background": "某城市投放了 5000 辆共享单车，分布在 200 个站点。",
  "questions": [
    "建立模型预测各站点未来一小时的需求量。",
    "在不超过 100 辆运力的前提下，设计最优调度方案。"
  ],
  "data_files": []
}
```

- [ ] **Step 2：写 `tests/conftest.py`**（提供示例题目、临时工作目录）

```python
import json
from pathlib import Path
import pytest


@pytest.fixture
def sample_problem():
    p = Path(__file__).parent / "fixtures" / "sample_problem.json"
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path
```

> 节点测试各自用 `mocker.patch("math_agent.nodes.<name>.complete", ...)` 拦截 LLM；
> 不在 conftest 里提供通用 LLM mock fixture，避免 patch 目标错配的隐性 bug。

- [ ] **Step 3：跑空测试，验证 pytest 能发现 conftest**

Run: `pytest -q`
Expected: `no tests ran` 但 0 errors

- [ ] **Step 4：提交**

```bash
git add tests/
git commit -m "test: add shared fixtures and sample problem"
```

---

## Phase 1：LLM 封装（LiteLLM）

### Task 1.1：定义 `llm.complete` 的契约（先写测试）

**Files:**
- Create: `tests/test_llm.py`

- [ ] **Step 1：写失败的测试**

```python
import pytest
from pydantic import BaseModel
import math_agent.llm as llm


class _Answer(BaseModel):
    summary: str
    score: int


def test_complete_returns_text_when_no_schema(mocker):
    mocker.patch(
        "litellm.completion",
        return_value=mocker.MagicMock(
            choices=[mocker.MagicMock(message=mocker.MagicMock(content="hello"))]
        ),
    )
    out = llm.complete("say hi", model="gpt-4o-mini")
    assert out == "hello"


def test_complete_returns_pydantic_when_schema(mocker):
    payload = '{"summary": "ok", "score": 9}'
    mocker.patch(
        "litellm.completion",
        return_value=mocker.MagicMock(
            choices=[mocker.MagicMock(message=mocker.MagicMock(content=payload))]
        ),
    )
    out = llm.complete("rate it", schema=_Answer, model="gpt-4o-mini")
    assert isinstance(out, _Answer)
    assert out.score == 9


def test_complete_retries_on_invalid_json(mocker):
    bad = mocker.MagicMock(choices=[mocker.MagicMock(message=mocker.MagicMock(content="not json"))])
    good = mocker.MagicMock(choices=[mocker.MagicMock(message=mocker.MagicMock(content='{"summary":"x","score":1}'))])
    mocker.patch("litellm.completion", side_effect=[bad, good])
    out = llm.complete("x", schema=_Answer, model="gpt-4o-mini", max_retries=2)
    assert out.score == 1


def test_complete_raises_after_all_retries_exhausted(mocker):
    bad = mocker.MagicMock(choices=[mocker.MagicMock(message=mocker.MagicMock(content="nope"))])
    mocker.patch("litellm.completion", return_value=bad)
    with pytest.raises(llm.LLMError):
        llm.complete("x", schema=_Answer, max_retries=1)
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/test_llm.py -v`
Expected: ImportError 或 `AttributeError: complete`。

---

### Task 1.2：实现 `llm.complete`

**Files:**
- Create: `src/math_agent/config.py`
- Create: `src/math_agent/llm.py`

- [ ] **Step 1：写 `config.py`**

```python
"""集中配置。所有可调参数都从这里读，避免节点里散落硬编码。"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("MATH_AGENT_DEFAULT_MODEL", "gpt-4o-mini")
STRONG_MODEL = os.getenv("MATH_AGENT_STRONG_MODEL", "gpt-4o")

# 节点 -> 模型 的路由表。便于 Critic 用强模型，常规节点用便宜模型。
MODEL_ROUTING = {
    "analyst": STRONG_MODEL,
    "modeler": STRONG_MODEL,
    "model_critic": STRONG_MODEL,
    "coder": DEFAULT_MODEL,
    "writer": STRONG_MODEL,
}

# 循环 / 重试上限
MAX_MODEL_ITERATIONS = 3      # basic -> improved -> final 之外的修正轮次
MAX_LLM_RETRIES = 2           # 单次 LLM 调用的结构化解析重试

# 路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"
```

- [ ] **Step 2：写 `llm.py`**

```python
"""LiteLLM 统一入口。

设计原则：
- 所有节点必须通过 complete() 调用 LLM，不直接 import litellm。
- 支持 Pydantic schema → 结构化输出 + JSON 解析重试。
- 错误统一抛 LLMError，调用方决定降级策略。
"""
from __future__ import annotations

import json
from typing import Any, Type, TypeVar

import litellm
from pydantic import BaseModel, ValidationError

from math_agent.config import DEFAULT_MODEL, MAX_LLM_RETRIES

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """LLM 调用或结构化解析失败。"""


def complete(
    prompt: str,
    *,
    schema: Type[T] | None = None,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.3,
    max_retries: int = MAX_LLM_RETRIES,
    **kwargs: Any,
) -> str | T:
    """统一 LLM 调用。

    - schema 为 None 时返回纯文本；
    - schema 为 Pydantic 模型时强制 JSON 输出，解析失败会自动重试，并把"上次输出 + 错误"喂回模型让它修正。

    错误状态分两类，互不污染：
    - 网络/服务异常 (litellm.completion 抛错)：换一次，不构造反馈消息。
    - JSON 解析失败：把上次内容 + 解析错误打包成 parse_feedback，下一轮注入。
    """
    model = model or DEFAULT_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response_format = None
    if schema is not None:
        response_format = {"type": "json_object"}

    last_err: Exception | None = None
    parse_feedback: tuple[str, Exception] | None = None  # (上次内容, 解析错误)
    for attempt in range(max_retries + 1):
        msgs = list(messages)
        if schema and parse_feedback is not None:
            prev_content, prev_err = parse_feedback
            msgs.append({"role": "assistant", "content": prev_content})
            msgs.append(
                {
                    "role": "user",
                    "content": (
                        f"上一次响应无法被解析为目标 schema：{prev_err}。"
                        f"请只输出严格符合以下 JSON Schema 的对象：\n"
                        f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
                    ),
                }
            )
        try:
            raw = litellm.completion(
                model=model,
                messages=msgs,
                temperature=temperature,
                response_format=response_format,
                **kwargs,
            )
        except Exception as e:  # 网络 / 鉴权 / 限流：不动 parse_feedback
            last_err = e
            continue

        content = raw.choices[0].message.content or ""
        if schema is None:
            return content
        try:
            return schema.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError) as e:
            last_err = e
            parse_feedback = (content, e)
            continue

    raise LLMError(f"LLM 调用失败：{last_err}")
```

- [ ] **Step 3：跑测试**

Run: `pytest tests/test_llm.py -v`
Expected: 4 passed.

- [ ] **Step 4：提交**

```bash
git add src/math_agent/config.py src/math_agent/llm.py tests/test_llm.py
git commit -m "feat: litellm wrapper with pydantic structured output and retry"
```

---

## Phase 2：State Schema

### Task 2.1：定义 State（先写测试）

**Files:**
- Create: `tests/test_state.py`

- [ ] **Step 1：写失败的测试**

```python
from math_agent.state import (
    MathModelingState,
    Assumption,
    ModelVersion,
    CriticReport,
    PaperSections,
)


def test_initial_state_minimal():
    s = MathModelingState(problem="某共享单车调度优化问题")
    assert s.problem
    assert s.assumptions == []
    assert s.model_versions == []
    assert s.iteration == 0
    assert s.errors == []


def test_state_can_append_model_version():
    s = MathModelingState(problem="x")
    s.model_versions.append(
        ModelVersion(stage="basic", description="排队论 M/M/1", equations=["lambda < mu"])
    )
    assert s.latest_model().stage == "basic"


def test_state_can_record_critic():
    s = MathModelingState(problem="x")
    s.critic_reports.append(
        CriticReport(
            target="modeler",
            score=7,
            issues=["假设过强"],
            suggestions=["放宽到时变需求"],
            stage="basic",
        )
    )
    assert s.critic_reports[-1].score == 7
    assert s.critic_reports[-1].stage == "basic"


def test_latest_critic_for_stage_filters_by_stage():
    s = MathModelingState(problem="x")
    s.critic_reports.append(CriticReport(target="modeler", score=4, stage="basic"))
    s.critic_reports.append(CriticReport(target="modeler", score=9, approved=True, stage="basic"))
    s.critic_reports.append(CriticReport(target="modeler", score=5, stage="improved"))
    # basic 阶段的最新 critic 是 score=9 那条，不是 improved 的 5 分
    assert s.latest_critic_for_stage("modeler", "basic").score == 9
    assert s.latest_critic_for_stage("modeler", "improved").score == 5
    assert s.latest_critic_for_stage("modeler", "final") is None


def test_paper_sections_defaults_empty():
    p = PaperSections()
    assert p.abstract == ""
    assert p.conclusion == ""
    # sensitivity 字段在 MVP 已移除，避免 Plan B 引入后字段名漂移
    assert not hasattr(p, "sensitivity")
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/test_state.py -v`
Expected: ImportError。

---

### Task 2.2：实现 State

**Files:**
- Create: `src/math_agent/state.py`

- [ ] **Step 1：写 `state.py`**

```python
"""MathModelingState：整张图共享的状态，**同时充当 LangGraph 的 state schema**。

设计要点：
- 直接把 Pydantic 模型作为 LangGraph 的 state（langgraph>=0.2 支持）。
  list 字段用 `Annotated[..., operator.add]` 标记追加语义，节点返回增量字典时由
  reducer 自动合并；标量字段用"最新覆盖"语义。
- 节点签名统一为 `(state: MathModelingState) -> dict`，返回 **增量**。
- 不在 state 里放大文件，code/figure 路径只存路径字符串。
- 不再维护一份单独的 GraphState TypedDict，避免双重维护。
"""
from __future__ import annotations

from operator import add
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field

ModelStage = Literal["basic", "improved", "final"]


class Assumption(BaseModel):
    statement: str
    rationale: str = ""
    sensitivity_relevant: bool = False  # MVP 不消费，预留给 Plan B 的 Sensitivity 节点


class ModelVersion(BaseModel):
    stage: ModelStage
    description: str
    equations: list[str] = Field(default_factory=list)
    variables: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class CodeArtifact(BaseModel):
    purpose: str
    code: str
    stdout: str = ""
    stderr: str = ""
    success: bool = False
    artifact_paths: list[str] = Field(default_factory=list)  # 生成的图、数据等


class CriticReport(BaseModel):
    target: Literal["analyst", "modeler", "coder", "writer", "paper"]
    score: int  # 0-10
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    approved: bool = False
    # stage 标记 critic 是针对哪个建模阶段产生的；analyst/coder/writer/paper 类型可为 None
    stage: Optional[ModelStage] = None


class PaperSections(BaseModel):
    abstract: str = ""
    problem_restatement: str = ""
    assumptions: str = ""
    notation: str = ""
    model_section: str = ""
    solution: str = ""
    # sensitivity 章节延后到 Plan B 引入 Sensitivity 节点时再加回 schema 与模板
    conclusion: str = ""
    references: str = ""


class MathModelingState(BaseModel):
    # 输入
    problem: str
    background: str = ""
    questions: list[str] = Field(default_factory=list)

    # 中间产物（list 字段都是 append 语义）
    assumptions: Annotated[list[Assumption], add] = Field(default_factory=list)
    model_versions: Annotated[list[ModelVersion], add] = Field(default_factory=list)
    code_artifacts: Annotated[list[CodeArtifact], add] = Field(default_factory=list)
    critic_reports: Annotated[list[CriticReport], add] = Field(default_factory=list)

    # 论文（覆盖语义）
    paper: PaperSections = Field(default_factory=PaperSections)

    # 流程控制（覆盖语义）
    iteration: int = 0
    stage_target: ModelStage = "basic"  # 当前要产出的阶段
    errors: Annotated[list[str], add] = Field(default_factory=list)

    # 输出
    output_dir: Optional[str] = None

    # ---- 便利方法 ----
    def latest_model(self) -> ModelVersion | None:
        return self.model_versions[-1] if self.model_versions else None

    def latest_critic(self, target: str) -> CriticReport | None:
        for r in reversed(self.critic_reports):
            if r.target == target:
                return r
        return None

    def latest_critic_for_stage(self, target: str, stage: ModelStage) -> CriticReport | None:
        """按 (target, stage) 过滤，避免上一阶段未通过的反馈污染下一阶段。"""
        for r in reversed(self.critic_reports):
            if r.target == target and r.stage == stage:
                return r
        return None
```

- [ ] **Step 2：跑测试**

Run: `pytest tests/test_state.py -v`
Expected: 5 passed.

- [ ] **Step 3：提交**

```bash
git add src/math_agent/state.py tests/test_state.py
git commit -m "feat: pydantic state schema for math modeling pipeline"
```

---

## Phase 3：Prompt 与 Analyst 节点（首个节点 + 图骨架）

### Task 3.1：Analyst prompt

**Files:**
- Create: `src/math_agent/prompts/analyst.py`

- [ ] **Step 1：写 prompt 模板**

```python
"""Analyst：把题目分解为结构化假设清单。"""

SYSTEM = (
    "你是数学建模竞赛队的首席分析师。你要把题目拆解为"
    "（1）核心问题列表 （2）建模所需的假设清单（每条说明依据，并指出是否需要做敏感性分析）。"
    "禁止编造未给出的数据。"
)


def build_prompt(problem: str, background: str, questions: list[str]) -> str:
    qs = "\n".join(f"- {q}" for q in questions) or "（题目本身未列出独立小问）"
    return (
        f"# 题目\n{problem}\n\n"
        f"# 背景\n{background or '（无）'}\n\n"
        f"# 小问\n{qs}\n\n"
        f"请输出 JSON：{{\n"
        f"  \"assumptions\": [{{\"statement\": str, \"rationale\": str, \"sensitivity_relevant\": bool}}, ...]\n"
        f"}}，至少 5 条假设。"
        f"（sensitivity_relevant 字段允许 LLM 标注是否需要做敏感性分析，MVP 不强制；"
        f"Plan B 的 Sensitivity 节点会消费该字段。）"
    )
```

- [ ] **Step 2：提交**

```bash
git add src/math_agent/prompts/analyst.py
git commit -m "feat: analyst prompt"
```

---

### Task 3.2：Analyst 节点（先写测试）

**Files:**
- Create: `tests/nodes/test_analyst.py`

- [ ] **Step 1：写失败的测试**

```python
from pydantic import BaseModel
from math_agent.state import MathModelingState
from math_agent.nodes.analyst import analyst_node, AnalystOutput


def test_analyst_appends_assumptions(mocker):
    fake = AnalystOutput(
        assumptions=[
            {"statement": "需求服从泊松", "rationale": "短时间高频独立到达", "sensitivity_relevant": True},
            {"statement": "调度车恒速", "rationale": "市区均速近似", "sensitivity_relevant": False},
        ]
    )
    mocker.patch("math_agent.nodes.analyst.complete", return_value=fake)

    state = MathModelingState(problem="共享单车调度", questions=["预测", "调度"])
    delta = analyst_node(state)

    assert "assumptions" in delta
    assert len(delta["assumptions"]) == 2
    assert delta["assumptions"][0].sensitivity_relevant is True
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/nodes/test_analyst.py -v`
Expected: ImportError。

---

### Task 3.3：实现 Analyst 节点

**Files:**
- Create: `src/math_agent/nodes/analyst.py`

- [ ] **Step 1：写节点**

```python
from pydantic import BaseModel
from math_agent.llm import complete
from math_agent.config import MODEL_ROUTING
from math_agent.prompts.analyst import SYSTEM, build_prompt
from math_agent.state import Assumption, MathModelingState


class AnalystOutput(BaseModel):
    assumptions: list[Assumption]


def analyst_node(state: MathModelingState) -> dict:
    prompt = build_prompt(state.problem, state.background, state.questions)
    out: AnalystOutput = complete(
        prompt,
        schema=AnalystOutput,
        system=SYSTEM,
        model=MODEL_ROUTING["analyst"],
    )
    return {"assumptions": out.assumptions}
```

- [ ] **Step 2：跑测试**

Run: `pytest tests/nodes/test_analyst.py -v`
Expected: 1 passed.

- [ ] **Step 3：提交**

```bash
git add src/math_agent/nodes/analyst.py tests/nodes/test_analyst.py
git commit -m "feat: analyst node"
```

---

### Task 3.4：图骨架与 reducer

**Files:**
- Create: `src/math_agent/routing.py`
- Create: `src/math_agent/graph.py`
- Create: `tests/test_routing.py`

- [ ] **Step 1：先写 routing 的测试**

```python
from math_agent.state import MathModelingState, ModelVersion, CriticReport
from math_agent.routing import after_model_critic


def _state_with(stage, score, iteration):
    s = MathModelingState(problem="x", iteration=iteration, stage_target=stage)
    s.model_versions.append(ModelVersion(stage=stage, description="d"))
    s.critic_reports.append(
        CriticReport(target="modeler", score=score, approved=score >= 8, stage=stage)
    )
    return s


def test_routing_approved_basic_goes_to_improved():
    assert after_model_critic(_state_with("basic", 9, 0)) == "advance"


def test_routing_low_score_retries():
    assert after_model_critic(_state_with("basic", 4, 0)) == "retry"


def test_routing_caps_retries():
    # 即使分数低，达到迭代上限也必须前进
    assert after_model_critic(_state_with("basic", 4, 3)) == "advance"


def test_routing_after_final_goes_to_coder():
    assert after_model_critic(_state_with("final", 9, 0)) == "to_coder"
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/test_routing.py -v`
Expected: ImportError。

- [ ] **Step 3：实现 `routing.py`**

```python
"""集中放图的条件边路由函数，便于单元测试。

约定的返回值是字符串字面量，graph.py 会把它映射到具体节点名。
"""
from math_agent.config import MAX_MODEL_ITERATIONS
from math_agent.state import MathModelingState

_STAGE_NEXT = {"basic": "improved", "improved": "final"}


def after_model_critic(state: MathModelingState) -> str:
    """basic/improved/final 任一阶段评审完后的去向。"""
    critic = state.latest_critic("modeler")
    if critic is None:
        return "retry"

    if state.stage_target == "final":
        if critic.approved or state.iteration >= MAX_MODEL_ITERATIONS:
            return "to_coder"
        return "retry"

    # basic / improved
    if critic.approved or state.iteration >= MAX_MODEL_ITERATIONS:
        return "advance"
    return "retry"


def next_stage(current: str) -> str:
    return _STAGE_NEXT[current]
```

- [ ] **Step 4：跑测试**

Run: `pytest tests/test_routing.py -v`
Expected: 4 passed.

- [ ] **Step 5：实现 `graph.py`（先放 Analyst + END，后续阶段再追加）**

```python
"""LangGraph 图构建。

约定：
- 直接把 `MathModelingState`（Pydantic）当 LangGraph 的 state schema，
  list 字段的 `Annotated[..., add]` 注解由 LangGraph 识别为追加 reducer。
- 节点函数签名统一 `(state: MathModelingState) -> dict`，返回 **增量** 字段。
- 没有独立的 GraphState TypedDict，避免双重维护。
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from math_agent.state import MathModelingState
from math_agent.nodes.analyst import analyst_node


def build_graph():
    g = StateGraph(MathModelingState)
    g.add_node("analyst", analyst_node)
    g.set_entry_point("analyst")
    g.add_edge("analyst", END)
    return g.compile()
```

- [ ] **Step 6：写 smoke 测试**

Create `tests/test_graph_smoke.py`:

```python
from math_agent.graph import build_graph
from math_agent.state import Assumption
from math_agent.nodes.analyst import AnalystOutput


def test_graph_runs_analyst(mocker):
    fake = AnalystOutput(
        assumptions=[Assumption(statement="A", rationale="r", sensitivity_relevant=True)]
    )
    mocker.patch("math_agent.nodes.analyst.complete", return_value=fake)

    g = build_graph()
    final = g.invoke({"problem": "p", "background": "", "questions": []})
    assert len(final["assumptions"]) == 1
```

- [ ] **Step 7：跑测试**

Run: `pytest tests/test_graph_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 8：提交**

```bash
git add src/math_agent/routing.py src/math_agent/graph.py tests/test_routing.py tests/test_graph_smoke.py
git commit -m "feat: graph skeleton with analyst entrypoint and routing"
```

---

## Phase 4：Modeler + ModelCritic 循环

### Task 4.1：Modeler prompt 与节点

**Files:**
- Create: `src/math_agent/prompts/modeler.py`
- Create: `src/math_agent/nodes/modeler.py`
- Create: `tests/nodes/test_modeler.py`

- [ ] **Step 1：写 modeler prompt**

```python
"""Modeler：依据当前 stage 产出对应版本的模型。"""

SYSTEM = (
    "你是数学建模队的主建模手。请在给定假设下构建数学模型。"
    "你必须按照 stage 渐进：basic（最简可解模型）→ improved（加入更多现实因素）"
    "→ final（综合性最强、可被敏感性分析的最终模型）。"
)


def build_prompt(problem, assumptions, prev_model, stage, critic_feedback=None):
    asum = "\n".join(f"- {a.statement}（依据：{a.rationale}）" for a in assumptions) or "（暂无）"
    prev = "（无前一版本）"
    if prev_model is not None:
        prev = f"[{prev_model.stage}] {prev_model.description}\n方程：" + " ; ".join(prev_model.equations)
    fb = ""
    if critic_feedback:
        fb = "\n# 上一版 Critic 反馈\n" + "\n".join(
            f"- 问题: {i}" for i in critic_feedback.issues
        ) + "\n" + "\n".join(f"- 建议: {s}" for s in critic_feedback.suggestions)

    return (
        f"# 题目\n{problem}\n\n# 当前阶段\n{stage}\n\n"
        f"# 已确认假设\n{asum}\n\n# 上一版模型\n{prev}\n{fb}\n\n"
        f"请输出 JSON：{{\n"
        f"  \"stage\": \"{stage}\",\n"
        f"  \"description\": str,        # 模型定位与核心思路，>= 200 字\n"
        f"  \"equations\": [str, ...],   # LaTeX 字符串\n"
        f"  \"variables\": {{name: meaning}},\n"
        f"  \"notes\": str               # 与上一版的区别（basic 阶段可为空）\n"
        f"}}"
    )
```

- [ ] **Step 2：写 modeler 测试**

```python
from math_agent.state import MathModelingState, Assumption, ModelVersion, CriticReport
from math_agent.nodes.modeler import modeler_node


def test_modeler_produces_basic_first(mocker):
    fake = ModelVersion(stage="basic", description="d" * 200, equations=["x=1"], variables={"x": "ok"})
    mocker.patch("math_agent.nodes.modeler.complete", return_value=fake)

    s = MathModelingState(problem="p", stage_target="basic")
    s.assumptions.append(Assumption(statement="a", rationale="r"))
    delta = modeler_node(s)
    assert delta["model_versions"][0].stage == "basic"
    assert delta["iteration"] == 1


def test_modeler_passes_critic_feedback(mocker):
    spy = mocker.patch(
        "math_agent.nodes.modeler.complete",
        return_value=ModelVersion(stage="basic", description="d"*200, equations=["y=2"]),
    )
    s = MathModelingState(problem="p", stage_target="basic", iteration=1)
    s.assumptions.append(Assumption(statement="a", rationale="r"))
    s.model_versions.append(ModelVersion(stage="basic", description="old"))
    s.critic_reports.append(
        CriticReport(target="modeler", score=4, issues=["弱"], suggestions=["改"], stage="basic")
    )
    modeler_node(s)
    prompt_arg = spy.call_args.args[0]
    assert "弱" in prompt_arg and "改" in prompt_arg


def test_modeler_ignores_other_stage_critic(mocker):
    """basic 阶段未通过的 critic 不应污染 improved 阶段的 prompt。"""
    spy = mocker.patch(
        "math_agent.nodes.modeler.complete",
        return_value=ModelVersion(stage="improved", description="d"*200),
    )
    s = MathModelingState(problem="p", stage_target="improved", iteration=0)
    s.assumptions.append(Assumption(statement="a", rationale="r"))
    s.model_versions.append(ModelVersion(stage="basic", description="basic-final"))
    s.critic_reports.append(
        CriticReport(target="modeler", score=4, issues=["basic-issue"], suggestions=["basic-fix"], stage="basic")
    )
    modeler_node(s)
    prompt_arg = spy.call_args.args[0]
    assert "basic-issue" not in prompt_arg
    assert "basic-fix" not in prompt_arg
```

- [ ] **Step 3：实现 modeler 节点**

```python
from math_agent.llm import complete
from math_agent.config import MODEL_ROUTING
from math_agent.prompts.modeler import SYSTEM, build_prompt
from math_agent.state import MathModelingState, ModelVersion


def modeler_node(state: MathModelingState) -> dict:
    # 只关心针对**当前阶段**的上一版模型与上一份 critic，避免跨阶段污染。
    same_stage_prev = next(
        (m for m in reversed(state.model_versions) if m.stage == state.stage_target),
        None,
    )
    # 没有同阶段的上一版时，把上一阶段的最终版作为参考（用于 improved 起步）
    prev_for_stage = same_stage_prev or (state.model_versions[-1] if state.model_versions else None)

    critic_fb = state.latest_critic_for_stage("modeler", state.stage_target)
    # 当前阶段已 approved，不再回灌反馈
    if critic_fb and critic_fb.approved:
        critic_fb = None

    prompt = build_prompt(
        state.problem, state.assumptions, prev_for_stage, state.stage_target, critic_fb
    )
    out: ModelVersion = complete(
        prompt, schema=ModelVersion, system=SYSTEM, model=MODEL_ROUTING["modeler"]
    )
    # 保证 stage 与请求一致（防 LLM 篡改）
    out.stage = state.stage_target
    return {"model_versions": [out], "iteration": state.iteration + 1}
```

- [ ] **Step 4：跑测试**

Run: `pytest tests/nodes/test_modeler.py -v`
Expected: 3 passed.

- [ ] **Step 5：提交**

```bash
git add src/math_agent/prompts/modeler.py src/math_agent/nodes/modeler.py tests/nodes/test_modeler.py
git commit -m "feat: modeler node with stage progression and critic feedback loop"
```

---

### Task 4.2：ModelCritic prompt 与节点

**Files:**
- Create: `src/math_agent/prompts/model_critic.py`
- Create: `src/math_agent/nodes/model_critic.py`
- Create: `tests/nodes/test_model_critic.py`

- [ ] **Step 1：写 prompt**

```python
"""ModelCritic：从假设合理性、数学严密性、与题目相关性、可计算性 4 维度评分。"""

SYSTEM = (
    "你是国赛评委。请就给定模型给出 0-10 的整数总评分（>=8 视为通过），"
    "并列出至多 5 个 issues 与至多 5 个 suggestions。"
    "重点检查：假设是否被显式承接、方程量纲是否一致、是否存在更优经典模型。"
)


def build_prompt(problem, assumptions, model):
    asum = "\n".join(f"- {a.statement}" for a in assumptions)
    eqs = "\n".join(f"  - $$ {e} $$" for e in model.equations)
    return (
        f"# 题目\n{problem}\n\n# 假设\n{asum}\n\n# 模型（{model.stage}）\n"
        f"{model.description}\n方程：\n{eqs}\n\n"
        f"请输出 JSON：{{\"target\":\"modeler\",\"score\":int,\"issues\":[str],\"suggestions\":[str],\"approved\":bool}}"
    )
```

- [ ] **Step 2：写测试**

```python
from math_agent.state import MathModelingState, ModelVersion, CriticReport
from math_agent.nodes.model_critic import model_critic_node


def test_model_critic_appends_report(mocker):
    fake = CriticReport(target="modeler", score=8, issues=[], suggestions=[], approved=True)
    mocker.patch("math_agent.nodes.model_critic.complete", return_value=fake)
    s = MathModelingState(problem="p")
    s.model_versions.append(ModelVersion(stage="basic", description="d"))
    delta = model_critic_node(s)
    assert delta["critic_reports"][0].approved is True
    # critic 必须记录所针对的阶段，下游 modeler_node 据此过滤反馈
    assert delta["critic_reports"][0].stage == "basic"


def test_model_critic_tags_stage_from_latest_model(mocker):
    """即便 LLM 未返回 stage 字段，节点也要按当前最新 model 的 stage 强制覆写。"""
    fake = CriticReport(target="modeler", score=6, approved=False)  # stage 默认 None
    mocker.patch("math_agent.nodes.model_critic.complete", return_value=fake)
    s = MathModelingState(problem="p")
    s.model_versions.append(ModelVersion(stage="improved", description="d"))
    delta = model_critic_node(s)
    assert delta["critic_reports"][0].stage == "improved"


def test_model_critic_no_model_records_error(mocker):
    s = MathModelingState(problem="p")
    delta = model_critic_node(s)
    assert delta["errors"]
    assert delta.get("critic_reports", []) == []
```

- [ ] **Step 3：实现节点**

```python
from math_agent.llm import complete
from math_agent.config import MODEL_ROUTING
from math_agent.prompts.model_critic import SYSTEM, build_prompt
from math_agent.state import MathModelingState, CriticReport


def model_critic_node(state: MathModelingState) -> dict:
    model = state.latest_model()
    if model is None:
        return {"errors": ["ModelCritic: 当前 state 中没有 model_versions"]}

    prompt = build_prompt(state.problem, state.assumptions, model)
    out: CriticReport = complete(
        prompt, schema=CriticReport, system=SYSTEM, model=MODEL_ROUTING["model_critic"]
    )
    out.target = "modeler"        # 防篡改
    out.stage = model.stage       # 标记 critic 针对的阶段，避免跨阶段污染（见 modeler_node）
    return {"critic_reports": [out]}
```

- [ ] **Step 4：跑测试**

Run: `pytest tests/nodes/test_model_critic.py -v`
Expected: 3 passed.

- [ ] **Step 5：提交**

```bash
git add src/math_agent/prompts/model_critic.py src/math_agent/nodes/model_critic.py tests/nodes/test_model_critic.py
git commit -m "feat: model critic node"
```

---

### Task 4.3：把 Modeler/Critic 接入图

**Files:**
- Modify: `src/math_agent/graph.py`
- Modify: `tests/test_graph_smoke.py`

- [ ] **Step 1：替换 `build_graph`，加入 modeler/critic/stage 推进**

```python
"""LangGraph 图构建。"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from math_agent.state import MathModelingState
from math_agent.nodes.analyst import analyst_node
from math_agent.nodes.modeler import modeler_node
from math_agent.nodes.model_critic import model_critic_node
from math_agent.routing import after_model_critic, next_stage


def _advance_stage(state: MathModelingState) -> dict:
    return {"stage_target": next_stage(state.stage_target), "iteration": 0}


def build_graph():
    g = StateGraph(MathModelingState)
    g.add_node("analyst", analyst_node)
    g.add_node("modeler", modeler_node)
    g.add_node("model_critic", model_critic_node)
    g.add_node("advance_stage", _advance_stage)

    g.set_entry_point("analyst")
    g.add_edge("analyst", "modeler")
    g.add_edge("modeler", "model_critic")
    g.add_conditional_edges(
        "model_critic",
        after_model_critic,
        {"retry": "modeler", "advance": "advance_stage", "to_coder": END},  # to_coder 在 Phase 5 接入
    )
    g.add_edge("advance_stage", "modeler")
    return g.compile()
```

- [ ] **Step 2：扩展 smoke 测试**

把 `tests/test_graph_smoke.py` 改写为：

```python
from math_agent.graph import build_graph
from math_agent.state import Assumption, ModelVersion, CriticReport
from math_agent.nodes.analyst import AnalystOutput


def test_graph_runs_full_modeling_loop(mocker):
    mocker.patch(
        "math_agent.nodes.analyst.complete",
        return_value=AnalystOutput(
            assumptions=[Assumption(statement="A", rationale="r", sensitivity_relevant=True)]
        ),
    )

    stage_iter = iter(["basic", "improved", "final"])

    def fake_modeler(prompt, **kw):
        return ModelVersion(stage=next(stage_iter), description="d"*200, equations=["x=1"])

    def fake_critic(prompt, **kw):
        # stage 由 model_critic_node 强制覆写，这里随便填
        return CriticReport(target="modeler", score=9, approved=True)

    mocker.patch("math_agent.nodes.modeler.complete", side_effect=fake_modeler)
    mocker.patch("math_agent.nodes.model_critic.complete", side_effect=fake_critic)

    g = build_graph()
    final = g.invoke({"problem": "p", "stage_target": "basic", "iteration": 0})
    stages = [m.stage for m in final["model_versions"]]
    assert stages == ["basic", "improved", "final"]


def test_graph_retries_modeler_on_low_score(mocker):
    """basic 阶段前两轮 critic 不通过、第三轮通过：modeler 应在 basic 阶段被调 3 次；
    随后 improved/final 各一次性通过，覆盖 routing 的 `retry` 分支与 `latest_critic_for_stage` 过滤。"""
    mocker.patch(
        "math_agent.nodes.analyst.complete",
        return_value=AnalystOutput(assumptions=[Assumption(statement="A", rationale="r")]),
    )

    stage_iter = iter(["basic", "basic", "basic", "improved", "final"])
    mocker.patch(
        "math_agent.nodes.modeler.complete",
        side_effect=lambda *a, **k: ModelVersion(stage=next(stage_iter), description="d"*200),
    )

    critic_iter = iter([
        CriticReport(target="modeler", score=4, approved=False),
        CriticReport(target="modeler", score=5, approved=False),
        CriticReport(target="modeler", score=9, approved=True),   # basic done
        CriticReport(target="modeler", score=9, approved=True),   # improved done
        CriticReport(target="modeler", score=9, approved=True),   # final done
    ])
    mocker.patch(
        "math_agent.nodes.model_critic.complete",
        side_effect=lambda *a, **k: next(critic_iter),
    )

    g = build_graph()
    final = g.invoke({"problem": "p", "stage_target": "basic", "iteration": 0})

    basic_versions = [m for m in final["model_versions"] if m.stage == "basic"]
    basic_critics = [c for c in final["critic_reports"] if c.stage == "basic"]
    assert len(basic_versions) == 3
    assert len(basic_critics) == 3
    assert basic_critics[-1].approved is True
    # 最终 final 阶段也走完
    assert any(m.stage == "final" for m in final["model_versions"])
```

- [ ] **Step 3：跑测试**

Run: `pytest tests/test_graph_smoke.py -v`
Expected: 2 passed.

- [ ] **Step 4：提交**

```bash
git add src/math_agent/graph.py tests/test_graph_smoke.py
git commit -m "feat: wire modeler/critic loop with stage progression"
```

---

## Phase 5：代码执行器 + Coder

### Task 5.1：subprocess 代码执行器（runner）

> **不是隔离沙箱。** LLM 写的代码会以当前用户权限运行。本任务只做受限超时 + 临时工作目录 +
> 清空环境变量的最小防护，挡掉"顺手读 `~/.aws/credentials`"这类无心泄漏；不挡恶意输入。
> 真隔离（docker/firejail/nsjail）见 Plan C。本机可信使用前提下放心用。

**Files:**
- Create: `tests/test_runner.py`
- Create: `src/math_agent/tools/runner.py`

- [ ] **Step 1：写失败的测试**

```python
from math_agent.tools.runner import run_python, RunResult


def test_runner_runs_simple_code(workdir):
    res = run_python("print(1 + 1)", workdir=workdir)
    assert res.success
    assert "2" in res.stdout


def test_runner_captures_stderr_on_error(workdir):
    res = run_python("raise ValueError('boom')", workdir=workdir)
    assert not res.success
    assert "boom" in res.stderr


def test_runner_times_out(workdir):
    res = run_python("import time; time.sleep(30)", workdir=workdir, timeout=1)
    assert not res.success
    assert "timeout" in res.stderr.lower() or "killed" in res.stderr.lower()


def test_runner_lists_produced_files(workdir):
    code = "open('out.txt','w').write('x')"
    res = run_python(code, workdir=workdir)
    assert res.success
    assert any(p.endswith("out.txt") for p in res.artifact_paths)


def test_runner_strips_host_env(workdir, monkeypatch):
    """父进程的敏感环境变量不应继承到子进程。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leaked")
    res = run_python(
        "import os; print(os.environ.get('OPENAI_API_KEY', 'MISSING'))",
        workdir=workdir,
    )
    assert res.success
    assert "MISSING" in res.stdout
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/test_runner.py -v`
Expected: ImportError。

- [ ] **Step 3：实现 runner**

```python
"""subprocess 代码执行器。

**不是隔离沙箱**：LLM 写的代码以当前用户权限运行。本模块只做：
- 工作目录隔离（每次新建临时目录）
- 超时强杀
- 清空环境变量（仅传递最小 PATH / PYTHONPATH / SystemRoot），避免顺手读到 OPENAI_API_KEY 之类。

真正的隔离要靠 docker/firejail/nsjail，放 Plan C。
仅在本机可信使用前提下使用本模块。
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    artifact_paths: list[str] = field(default_factory=list)


def _minimal_env() -> dict[str, str]:
    """只透传子进程跑 Python 必需的变量。"""
    keys = ["PATH", "PYTHONPATH", "PYTHONHOME", "SystemRoot", "TEMP", "TMP", "LANG", "LC_ALL"]
    return {k: os.environ[k] for k in keys if k in os.environ}


def run_python(code: str, *, workdir: Path, timeout: int = 60) -> RunResult:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / "_run.py"
    script.write_text(code, encoding="utf-8")

    before = {p.name for p in workdir.iterdir()}

    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_minimal_env(),
        )
    except subprocess.TimeoutExpired as e:
        return RunResult(success=False, stdout=e.stdout or "", stderr=f"timeout after {timeout}s")

    after = {p.name for p in workdir.iterdir()}
    new_files = sorted(after - before - {"_run.py"})
    return RunResult(
        success=proc.returncode == 0,
        stdout=proc.stdout,
        stderr=proc.stderr,
        artifact_paths=[str(workdir / n) for n in new_files],
    )
```

- [ ] **Step 4：跑测试**

Run: `pytest tests/test_runner.py -v`
Expected: 5 passed.

- [ ] **Step 5：提交**

```bash
git add src/math_agent/tools/runner.py tests/test_runner.py
git commit -m "feat: subprocess python runner with timeout, tempdir, env stripping"
```

---

### Task 5.2：Coder 节点

**Files:**
- Create: `src/math_agent/prompts/coder.py`
- Create: `src/math_agent/nodes/coder.py`
- Create: `tests/nodes/test_coder.py`

- [ ] **Step 1：写 prompt**

```python
"""Coder：把 final 模型转成可执行 Python 代码。"""

SYSTEM = (
    "你是建模队的工程师。把给定的最终模型实现为一段**独立可运行**的 Python 脚本。"
    "约束：只用 numpy / scipy / matplotlib；不联网；不读取本地未声明的文件；"
    "需 print 关键结果，并把图保存到当前目录的 *.png。"
)


def build_prompt(model, prev_failure=None):
    eqs = "\n".join(f"- {e}" for e in model.equations)
    vars_ = "\n".join(f"- {k}: {v}" for k, v in model.variables.items())
    fb = ""
    if prev_failure:
        fb = f"\n# 上次运行失败\nstderr 节选：\n{prev_failure[:1000]}\n请修正后重试。"
    return (
        f"# 模型描述\n{model.description}\n\n# 方程\n{eqs}\n\n# 变量\n{vars_}\n{fb}\n\n"
        f"请输出 JSON：{{\"purpose\": str, \"code\": str}}，code 字段是完整的 Python 源码。"
    )
```

- [ ] **Step 2：写测试**

```python
from math_agent.state import MathModelingState, ModelVersion
from math_agent.nodes.coder import coder_node, CoderDraft


def test_coder_runs_code_and_records_artifact(mocker, workdir):
    mocker.patch(
        "math_agent.nodes.coder.complete",
        return_value=CoderDraft(purpose="solve", code="print('hello')"),
    )
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.model_versions.append(ModelVersion(stage="final", description="d", equations=["x=1"]))
    delta = coder_node(s)
    assert delta["code_artifacts"][0].success
    assert "hello" in delta["code_artifacts"][0].stdout


def test_coder_retries_once_on_failure(mocker, workdir):
    drafts = [
        CoderDraft(purpose="solve", code="raise RuntimeError('x')"),
        CoderDraft(purpose="solve", code="print('ok')"),
    ]
    mocker.patch("math_agent.nodes.coder.complete", side_effect=drafts)
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.model_versions.append(ModelVersion(stage="final", description="d"))
    delta = coder_node(s)
    # 应当保留两个 artifact：第一次失败、第二次成功
    arts = delta["code_artifacts"]
    assert len(arts) == 2
    assert arts[0].success is False
    assert arts[1].success is True
```

- [ ] **Step 3：实现节点**

```python
from __future__ import annotations
from pathlib import Path
import tempfile

from pydantic import BaseModel
from math_agent.llm import complete
from math_agent.config import MODEL_ROUTING
from math_agent.prompts.coder import SYSTEM, build_prompt
from math_agent.state import MathModelingState, CodeArtifact
from math_agent.tools.runner import run_python


class CoderDraft(BaseModel):
    purpose: str
    code: str


MAX_CODE_RETRIES = 1  # 一次失败后再给一次机会，避免成本失控


def coder_node(state: MathModelingState) -> dict:
    model = state.latest_model()
    workdir = Path(state.output_dir) if state.output_dir else Path(tempfile.mkdtemp(prefix="math_agent_"))
    workdir.mkdir(parents=True, exist_ok=True)

    artifacts: list[CodeArtifact] = []
    prev_err: str | None = None
    for attempt in range(MAX_CODE_RETRIES + 1):
        draft: CoderDraft = complete(
            build_prompt(model, prev_err),
            schema=CoderDraft,
            system=SYSTEM,
            model=MODEL_ROUTING["coder"],
        )
        result = run_python(draft.code, workdir=workdir / f"attempt_{attempt}")
        artifacts.append(
            CodeArtifact(
                purpose=draft.purpose,
                code=draft.code,
                stdout=result.stdout,
                stderr=result.stderr,
                success=result.success,
                artifact_paths=result.artifact_paths,
            )
        )
        if result.success:
            break
        prev_err = result.stderr

    return {"code_artifacts": artifacts}
```

- [ ] **Step 4：跑测试**

Run: `pytest tests/nodes/test_coder.py -v`
Expected: 2 passed.

- [ ] **Step 5：提交**

```bash
git add src/math_agent/prompts/coder.py src/math_agent/nodes/coder.py tests/nodes/test_coder.py
git commit -m "feat: coder node with retry-on-failure"
```

---

### Task 5.3：把 Coder 接进图

**Files:**
- Modify: `src/math_agent/graph.py`

- [ ] **Step 1：把 `to_coder` 从 END 改为 coder 节点，再连到 END（writer 在 Phase 6 接入）**

把 `build_graph()` 中的：

```python
from math_agent.nodes.model_critic import model_critic_node
```

下面加入：

```python
from math_agent.nodes.coder import coder_node
```

并把：

```python
        {"retry": "modeler", "advance": "advance_stage", "to_coder": END},
```

替换为：

```python
        {"retry": "modeler", "advance": "advance_stage", "to_coder": "coder"},
    )
    g.add_node("coder", coder_node)
    g.add_edge("coder", END)
```

- [ ] **Step 2：扩展 smoke 测试**

在 `tests/test_graph_smoke.py` 末尾追加：

```python
def test_graph_includes_coder(mocker, workdir):
    from math_agent.nodes.coder import CoderDraft
    mocker.patch(
        "math_agent.nodes.analyst.complete",
        return_value=AnalystOutput(assumptions=[Assumption(statement="A", rationale="r")]),
    )
    stage_iter = iter(["basic", "improved", "final"])
    mocker.patch(
        "math_agent.nodes.modeler.complete",
        side_effect=lambda *a, **k: ModelVersion(stage=next(stage_iter), description="d"*200),
    )
    mocker.patch(
        "math_agent.nodes.model_critic.complete",
        return_value=CriticReport(target="modeler", score=9, approved=True),
    )
    mocker.patch(
        "math_agent.nodes.coder.complete",
        return_value=CoderDraft(purpose="ok", code="print('done')"),
    )

    g = build_graph()
    final = g.invoke({
        "problem": "p", "stage_target": "basic", "iteration": 0,
        "output_dir": str(workdir),
    })
    assert final["code_artifacts"][-1].success
```

- [ ] **Step 3：跑测试**

Run: `pytest tests/test_graph_smoke.py -v`
Expected: 3 passed.

- [ ] **Step 4：提交**

```bash
git add src/math_agent/graph.py tests/test_graph_smoke.py
git commit -m "feat: wire coder node after final modeling stage"
```

---

## Phase 6：Writer（Markdown MVP）+ CLI

### Task 6.1：Writer prompt 与节点

**Files:**
- Create: `src/math_agent/prompts/writer.py`
- Create: `src/math_agent/nodes/writer.py`
- Create: `tests/nodes/test_writer.py`

- [ ] **Step 1：写 prompt**

```python
"""Writer：把 state 内的素材组装成论文各章节文本（Markdown）。"""

SYSTEM = (
    "你是负责撰写国赛论文的主笔。请把给定素材组织成正式论文章节。"
    "禁止编造数据；引用代码结果时使用'根据计算（见附录代码 X）'句式。"
)


def build_prompt(state):
    asum = "\n".join(f"- {a.statement}" for a in state.assumptions)
    models = "\n\n".join(
        f"### {m.stage}\n{m.description}\n方程：" + "; ".join(m.equations)
        for m in state.model_versions
    )
    code_stdout = "\n".join(a.stdout for a in state.code_artifacts if a.success)[:2000]
    return (
        f"# 题目\n{state.problem}\n\n# 假设\n{asum}\n\n"
        f"# 模型演化\n{models}\n\n# 代码运行关键输出（截断）\n{code_stdout}\n\n"
        f"请输出 JSON：{{\"abstract\":str,\"problem_restatement\":str,\"assumptions\":str,"
        f"\"notation\":str,\"model_section\":str,\"solution\":str,"
        f"\"conclusion\":str,\"references\":str}}。每段不少于 150 字。"
        f"（敏感性分析章节在 MVP 阶段不生成，Plan B 引入 Sensitivity 节点后会单独追加。）"
    )
```

- [ ] **Step 2：写测试**

```python
from math_agent.state import MathModelingState, ModelVersion, CodeArtifact, PaperSections
from math_agent.nodes.writer import writer_node


def test_writer_fills_paper(mocker):
    fake = PaperSections(
        abstract="a"*200, problem_restatement="b"*200, assumptions="c"*200,
        notation="d"*200, model_section="e"*200, solution="f"*200,
        conclusion="g"*200, references="h",
    )
    mocker.patch("math_agent.nodes.writer.complete", return_value=fake)
    s = MathModelingState(problem="p")
    s.model_versions.append(ModelVersion(stage="final", description="d"))
    s.code_artifacts.append(CodeArtifact(purpose="x", code="c", success=True, stdout="42"))
    delta = writer_node(s)
    assert isinstance(delta["paper"], PaperSections)
    assert delta["paper"].abstract.startswith("a")
```

- [ ] **Step 3：实现节点**

```python
from math_agent.llm import complete
from math_agent.config import MODEL_ROUTING
from math_agent.prompts.writer import SYSTEM, build_prompt
from math_agent.state import MathModelingState, PaperSections


def writer_node(state: MathModelingState) -> dict:
    out: PaperSections = complete(
        build_prompt(state),
        schema=PaperSections,
        system=SYSTEM,
        model=MODEL_ROUTING["writer"],
    )
    return {"paper": out}
```

- [ ] **Step 4：跑测试**

Run: `pytest tests/nodes/test_writer.py -v`
Expected: 1 passed.

- [ ] **Step 5：提交**

```bash
git add src/math_agent/prompts/writer.py src/math_agent/nodes/writer.py tests/nodes/test_writer.py
git commit -m "feat: writer node producing markdown paper sections"
```

---

### Task 6.2：Markdown 模板 + 渲染

**Files:**
- Create: `src/math_agent/templates/paper.md.j2`
- Create: `src/math_agent/nodes/writer.py` 中追加 `render_markdown`
- Modify: `tests/nodes/test_writer.py`

- [ ] **Step 1：写模板**

```jinja
# {{ problem }}

## 摘要
{{ paper.abstract }}

## 1. 问题重述
{{ paper.problem_restatement }}

## 2. 模型假设
{{ paper.assumptions }}

## 3. 符号说明
{{ paper.notation }}

## 4. 模型的建立与演化
{{ paper.model_section }}

## 5. 模型的求解
{{ paper.solution }}

## 6. 模型评价与结论
{{ paper.conclusion }}

## 参考文献
{{ paper.references }}

---

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

> 敏感性分析章节（原 "## 6. 敏感性分析"）在 MVP 模板中不出现，Plan B 引入 Sensitivity
> 节点后再加回章节、Schema 字段、Writer prompt 的对应输出键。

- [ ] **Step 2：在 `writer.py` 中追加 `render_markdown`**

```python
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=select_autoescape([]))


def render_markdown(state: MathModelingState) -> str:
    tmpl = _env.get_template("paper.md.j2")
    return tmpl.render(
        problem=state.problem,
        paper=state.paper,
        code_artifacts=state.code_artifacts,
    )
```

- [ ] **Step 3：在 `tests/nodes/test_writer.py` 追加测试**

```python
from math_agent.nodes.writer import render_markdown


def test_render_markdown_contains_sections():
    s = MathModelingState(problem="P")
    s.paper = PaperSections(abstract="A", problem_restatement="B", assumptions="C",
                            notation="D", model_section="E", solution="F",
                            conclusion="H", references="I")
    s.code_artifacts.append(CodeArtifact(purpose="x", code="print(1)", success=True, stdout="1"))
    md = render_markdown(s)
    assert "## 摘要" in md and "## 6. 模型评价" in md
    assert "print(1)" in md
```

- [ ] **Step 4：跑测试**

Run: `pytest tests/nodes/test_writer.py -v`
Expected: 2 passed.

- [ ] **Step 5：提交**

```bash
git add src/math_agent/templates/paper.md.j2 src/math_agent/nodes/writer.py tests/nodes/test_writer.py
git commit -m "feat: jinja2 markdown rendering for paper output"
```

---

### Task 6.3：把 Writer 接进图，并写 Markdown 到磁盘

**Files:**
- Modify: `src/math_agent/graph.py`

- [ ] **Step 1：加入 writer 与 finalize 节点**

在 `build_graph()` 顶部 import：

```python
from math_agent.nodes.writer import writer_node, render_markdown
```

替换最后几行：

```python
    g.add_node("coder", coder_node)
    g.add_node("writer", writer_node)

    def _finalize(state: MathModelingState) -> dict:
        from pathlib import Path
        if state.output_dir:
            out = Path(state.output_dir) / "paper.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(render_markdown(state), encoding="utf-8")
        return {}

    g.add_node("finalize", _finalize)
    g.add_edge("coder", "writer")
    g.add_edge("writer", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
```

- [ ] **Step 2：扩展 smoke 测试**

在 `tests/test_graph_smoke.py` 末尾追加：

```python
def test_graph_writes_paper_md(mocker, workdir):
    from math_agent.nodes.coder import CoderDraft
    from math_agent.state import PaperSections
    mocker.patch(
        "math_agent.nodes.analyst.complete",
        return_value=AnalystOutput(assumptions=[Assumption(statement="A", rationale="r")]),
    )
    stage_iter = iter(["basic", "improved", "final"])
    mocker.patch(
        "math_agent.nodes.modeler.complete",
        side_effect=lambda *a, **k: ModelVersion(stage=next(stage_iter), description="d"*200),
    )
    mocker.patch(
        "math_agent.nodes.model_critic.complete",
        return_value=CriticReport(target="modeler", score=9, approved=True),
    )
    mocker.patch(
        "math_agent.nodes.coder.complete",
        return_value=CoderDraft(purpose="ok", code="print('done')"),
    )
    mocker.patch(
        "math_agent.nodes.writer.complete",
        return_value=PaperSections(
            abstract="x"*200, problem_restatement="x"*200, assumptions="x"*200,
            notation="x"*200, model_section="x"*200, solution="x"*200,
            conclusion="x"*200, references="-",
        ),
    )
    g = build_graph()
    g.invoke({
        "problem": "single bike", "stage_target": "basic", "iteration": 0,
        "output_dir": str(workdir),
    })
    assert (workdir / "paper.md").exists()
    assert "## 摘要" in (workdir / "paper.md").read_text(encoding="utf-8")
```

- [ ] **Step 3：跑测试**

Run: `pytest tests/test_graph_smoke.py -v`
Expected: 4 passed.

- [ ] **Step 4：提交**

```bash
git add src/math_agent/graph.py tests/test_graph_smoke.py
git commit -m "feat: end-to-end pipeline writes paper.md to output_dir"
```

---

### Task 6.4：CLI 入口

**Files:**
- Create: `src/math_agent/cli.py`

- [ ] **Step 1：写 CLI**

```python
"""math-agent CLI。

用法:
  math-agent run --problem path/to/problem.json --out runs/2026-06-28
"""
import json
from pathlib import Path
import typer

from math_agent.graph import build_graph

app = typer.Typer(help="Math modeling multi-agent system (MVP).")


@app.command()
def run(
    problem: Path = typer.Option(..., exists=True, readable=True),
    out: Path = typer.Option(Path("runs/latest")),
):
    spec = json.loads(problem.read_text(encoding="utf-8"))
    initial = {
        "problem": spec.get("title", "") + "\n" + "\n".join(spec.get("questions", [])),
        "background": spec.get("background", ""),
        "questions": spec.get("questions", []),
        "stage_target": "basic",
        "iteration": 0,
        "output_dir": str(out),
    }
    g = build_graph()
    final = g.invoke(initial)
    typer.echo(f"done. paper at {out / 'paper.md'}")
    typer.echo(f"models: {[m.stage for m in final.get('model_versions', [])]}")
    typer.echo(f"iterations: {final.get('iteration')}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 2：手测**（需要真实 API key；在 CI 中跳过）

Run:
```bash
cp .env.example .env  # 填入 OPENAI_API_KEY
math-agent run --problem tests/fixtures/sample_problem.json --out runs/mvp-smoke
```
Expected: 终端输出 `done. paper at runs/mvp-smoke/paper.md`，且该文件存在并包含 9 个章节标题。

> 如果你不想真调 API，可以跳过手测，依赖前面 mock 化的 smoke 测试。

- [ ] **Step 3：提交**

```bash
git add src/math_agent/cli.py
git commit -m "feat: typer CLI entrypoint"
```

---

## Phase 7：自我审查与清理

### Task 7.1：跑全量测试 + 改 README

**Files:**
- Modify: `README.md`

- [ ] **Step 1：跑全测**

Run: `pytest -q`
Expected: 全绿，无 warning（除 deprecation 之外）。

- [ ] **Step 2：把 README 补成可用的快速上手指南**

```markdown
# math-agent (MVP)

LangGraph + LiteLLM 多智能体数学建模助手。本 MVP 实现：
**Analyst → Modeler(basic→improved→final) ⇄ ModelCritic → Coder(沙箱) → Writer → paper.md**

## 安装

```bash
pip install -e ".[dev]"
cp .env.example .env  # 填入 LLM API key
```

## 跑示例

```bash
math-agent run --problem tests/fixtures/sample_problem.json --out runs/demo
cat runs/demo/paper.md
```

## 跑测试

```bash
pytest -q
```

## 下一步（Plan B / Plan C）

- 强制 Sensitivity 节点 + Figure 流水线
- PaperCritic + Evaluation Module
- LaTeX 生成与编译
- HITL + Checkpointer
- 错误处理与重试统一化
- RAG（历年国一论文）
```

- [ ] **Step 3：提交**

```bash
git add README.md
git commit -m "docs: MVP usage and roadmap"
```

---

## 后续路线图（不在本计划范围内）

**Plan B：质量与产出层**
- `nodes/sensitivity.py`：强制运行参数扫并把图表 + 解读注入 `paper.sensitivity`
- `nodes/figure.py`：Coder 产出图 → FigureCritic（清晰度/标签/配色）→ FigureAnalyst（生成图说）→ 重生成循环
- `nodes/paper_critic.py` + `nodes/evaluation.py`：对齐国一四大标准 + 加分项打分
- `nodes/human.py` + Checkpointer：在 final 模型确认、终稿评审两个断点用 `interrupt_before`
- `tools/latex_compile.py` + `templates/paper.tex.j2`：xelatex 编译 + 失败时 Markdown 降级

**Plan C：鲁棒性与扩展**
- LLM 调用统一加 tenacity 重试 / 限流
- 沙箱失败的多轮自修复（区分语法错误、依赖错误、超时三类降级路径）
- 评分基准回归集（历年国一题）
- RAG：历年论文 / 经典模型库的向量检索注入到 prompt
- LangSmith / OpenTelemetry 链路追踪

---

## 自我审查

**Spec 覆盖**：
- 多智能体协作 → Analyst/Modeler/ModelCritic/Coder/Writer 节点已就位 ✓
- 状态机 + 条件边 + 循环 → graph.py 中通过 `add_conditional_edges` + 阶段推进实现 ✓
- 渐进式建模（basic→improved→final）→ `stage_target` + `next_stage` + `advance_stage` 节点 ✓
- 强制质量机制 → ModelCritic 循环 + 迭代上限 + critic 按 stage 过滤反馈（避免跨阶段污染）✓
- 代码执行 → `tools/runner.py`（非隔离）+ Coder retry ✓
- LaTeX/PDF → 显式标注落到 Plan B，MVP 用 Markdown 兜底 ✓
- Human-in-the-loop → 落到 Plan B（与 Checkpointer 一起）✓
- 敏感性分析强制节点 → 落到 Plan B（MVP schema/模板/prompt 均不含 sensitivity 章节，
  Plan B 接入 Sensitivity 节点时再加回 `PaperSections.sensitivity`、模板章节、Writer prompt 输出键）✓
- 图表 + 专业分析 → 落到 Plan B ✓
- RAG → 落到 Plan C ✓
- 鲁棒性 → 部分（LLM 重试、Coder 重试已做）；统一化落到 Plan C ✓

**Placeholder 扫描**：
- 全文无 TBD / TODO / 编造占位。CLI 输出真实 stage 列表与 iteration。

**已知非安全点**（接受 + 文档化，不修）：
- `tools/runner.py` **不是隔离沙箱**：LLM 写的代码以当前用户权限运行；只做超时 + 临时
  工作目录 + 清环境变量。文件 docstring 和文件结构注释均有红字提示。真隔离见 Plan C。

**类型一致性**：
- `MathModelingState` 同时充当 LangGraph state schema，list 字段用 `Annotated[..., add]`
  标记追加语义。节点签名统一 `(state: MathModelingState) -> dict`，无 `_wrap` 适配层。
- `analyst_node` 返回 `{"assumptions": [...]}` → list reducer `add` ✓
- `modeler_node` 返回 `{"model_versions": [ModelVersion], "iteration": int}`，
  `iteration` 是覆盖语义（每次 modeler 调用后 +1，`advance_stage` 重置为 0）✓
- `model_critic_node` 返回 `critic_reports: list[CriticReport]`，**强制写入 `stage` 字段**
  以匹配 `latest_critic_for_stage` 过滤 ✓
- `coder_node` 返回 `code_artifacts: list[CodeArtifact]` ✓
- `writer_node` 返回 `paper: PaperSections`（覆盖语义）✓
- routing 字面量 `retry/advance/to_coder` 与 graph 里的 `conditional_edges` 映射 key 一致 ✓

**测试覆盖关键控制流**：
- `test_complete_raises_after_all_retries_exhausted`：LLM 重试用尽抛 LLMError ✓
- `test_latest_critic_for_stage_filters_by_stage`：state 层的 stage 过滤 ✓
- `test_modeler_ignores_other_stage_critic`：节点层的跨阶段反馈隔离 ✓
- `test_model_critic_tags_stage_from_latest_model`：critic 写入时强制带 stage ✓
- `test_graph_retries_modeler_on_low_score`：图层面的 `retry` 分支 + 阶段推进端到端 ✓
- `test_runner_strips_host_env`：subprocess 子进程不继承 `OPENAI_API_KEY` ✓
