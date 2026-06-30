# 数学建模多智能体系统 — 鲁棒性与扩展层实现计划（Plan C）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Plan A/B 已经跑通完整流水线（含敏感性、图表、PaperCritic、Evaluation、HITL、LaTeX）的基础上，把"统一错误处理与重试、RAG 知识库注入、历年题回归基准、链路追踪与运行报告"四块工程能力补齐，让系统从"能跑"走向"能稳定跑、能复盘、能持续改进"。

**Architecture:**
- **统一鲁棒性层**：在 `llm.py` / `tools/runner.py` / `tools/latex_compile.py` 三处增加分类错误 + `tenacity` 重试装饰；引入 `errors.py` 集中错误类型，节点不再 try/except 散点。
- **RAG 层**：单独的 `rag/` 包，离线 `ingest` 命令把历年国一论文与经典模型片段切块 → 嵌入（仍走 `litellm.embedding`，避免引入新提供商抽象）→ 存 `sqlite-vec`（无需独立向量服务）。Prompt 装配阶段，Analyst/Modeler/Writer 三个 prompt 增加 `retrieved_context` 参数，由各自节点在调用 LLM 前查询。
- **回归基准**：`bench/` 目录放历年题 + 期望评分区间；`bench/runner.py` 用 mock 模式（不调 LLM，给定固定输出）和 live 模式（调 LLM）两套，CI 默认跑 mock。
- **追踪**：通过 `litellm` 的 callback + LangGraph 的 `on_node` 钩子写一份 JSON 运行报告到 `runs/<thread>/trace.json`。LangSmith / OTel 作为可选 callback，仅当环境变量存在时启用。

**Tech Stack:**（在 Plan A/B 的基础上新增）
- `tenacity>=8.2,<9`（统一重试 + 指数退避）
- `sqlite-vec>=0.1,<0.2`（轻量向量索引；无须 docker；0.1.x API 仍在变化，pin 主版本避免 CI 飘）
- `litellm.embedding` / `litellm.callbacks`（复用既有 LiteLLM；不引入 openai/anthropic SDK）
- `pypdf>=4.0,<5`（PDF 文本抽取，RAG ingest 用）
- `rich>=13.7,<14`（CLI 报告美化）
- 可选：`langsmith`、`opentelemetry-sdk`（仅在对应环境变量存在时启用）

> **前置条件**：Plan A、Plan B 已完成且 `pytest -q` 全绿。Plan C 不改变已落地节点的 *外部行为*，只在 LLM 调用边界、prompt 拼装、运行端做增强。

---

## 文件结构

新增（all paths relative to `build-agent/`）：

```
src/math_agent/
├── errors.py                       # 集中错误类
├── retry.py                        # tenacity 装饰器工厂
├── tracing.py                      # 节点/LLM 调用埋点 + JSON 报告
├── rag/
│   ├── __init__.py
│   ├── chunking.py                 # 文本切块（按 markdown/段落，固定窗口 + 重叠）
│   ├── embeddings.py               # litellm.embedding 包装 + 批处理
│   ├── store.py                    # sqlite-vec 写/查
│   ├── ingest.py                   # CLI 入口：扫描 corpus/ → 切块 → 嵌入 → 入库
│   └── retrieve.py                 # search(query, k) -> list[Snippet]
├── bench/
│   ├── __init__.py
│   ├── problems/                   # 历年题 JSON（与 sample_problem.json 同 schema）
│   │   ├── 2022_A.json
│   │   └── 2023_B.json
│   ├── expectations.json           # 每题的期望评分区间与必含关键词
│   └── runner.py                   # run_bench(out_dir) -> BenchReport（live：真跑 graph）
└── prompts/
    ├── analyst.py                  # 修改：build_prompt 接受 retrieved_context
    ├── modeler.py                  # 修改：同上
    └── writer.py                   # 修改：同上

tests/
├── test_errors.py
├── test_retry.py
├── test_tracing.py
├── rag/
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_store.py
│   ├── test_retrieve.py
│   └── test_ingest.py
└── bench/
    ├── conftest.py                 # bench mock harness（ExitStack + unittest.mock）
    └── test_runner_mock.py
docs/
└── plan-c-runbook.md               # 运行手册（含 RAG ingest、bench、trace 解读）
```

修改：
- `src/math_agent/llm.py`：用 `retry.py` 替换内置重试循环；新增 callback 钩子（tracing 用）
- `src/math_agent/tools/runner.py`：分类错误（语法/依赖/超时）+ 可选重试
- `src/math_agent/tools/latex_compile.py`：解析常见错误模式，给出可操作的 errors 字段
- `src/math_agent/nodes/analyst.py` / `modeler.py` / `writer.py`：调用 `rag.retrieve.search` 注入上下文
- `src/math_agent/config.py`：新增 RAG / tracing / bench 开关
- `src/math_agent/cli.py`：新增 `ingest` / `bench` / `report` 子命令

---

## Phase 1：统一错误处理与重试

### Task 1.1：errors 模块（先写测试）

**Files:**
- Create: `tests/test_errors.py`
- Create: `src/math_agent/errors.py`

- [ ] **Step 1：写失败的测试**

```python
import pytest
from math_agent.errors import (
    MathAgentError, LLMError, LLMRateLimitError, LLMValidationError,
    RunnerError, RunnerTimeoutError, RunnerRuntimeError,
    LatexError, LatexMissingBinaryError, LatexCompileError,
    classify_exception,
)


def test_class_hierarchy():
    assert issubclass(LLMRateLimitError, LLMError)
    assert issubclass(LLMValidationError, LLMError)
    assert issubclass(LLMError, MathAgentError)
    assert issubclass(RunnerTimeoutError, RunnerError)
    assert issubclass(RunnerRuntimeError, RunnerError)
    assert issubclass(LatexMissingBinaryError, LatexError)


def test_classify_rate_limit_from_litellm():
    class FakeRateLimit(Exception): pass
    FakeRateLimit.__name__ = "RateLimitError"
    e = FakeRateLimit("rate limited")
    out = classify_exception(e)
    assert isinstance(out, LLMRateLimitError)


def test_classify_unknown_passes_through_as_llmerror():
    e = RuntimeError("boom")
    out = classify_exception(e)
    assert isinstance(out, LLMError)
    assert "boom" in str(out)


def test_classify_keeps_existing_mathagent_error():
    inner = LLMValidationError("bad json")
    assert classify_exception(inner) is inner
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/test_errors.py -v`
Expected: ImportError.

- [ ] **Step 3：实现 `errors.py`**

```python
"""集中错误类型。

层次：
  MathAgentError
    ├── LLMError
    │     ├── LLMRateLimitError       # 触发指数退避
    │     ├── LLMValidationError      # 结构化输出解析失败；可"喂回错误"重试
    │     └── LLMTransportError       # 网络/超时；短间隔重试
    ├── RunnerError                   # 对应 tools/runner.py 的 subprocess 执行错误
    │     ├── RunnerTimeoutError
    │     └── RunnerRuntimeError
    └── LatexError
          ├── LatexMissingBinaryError # 不应重试
          └── LatexCompileError       # 可解析 .log 给出建议

命名约定：与 `tools/runner.py`（Plan A 已重命名）保持一致——所有错误类、结果字段、
fixture 都用 `Runner*` 前缀，不再使用历史名 `Sandbox*`。

分类原则：依据**重试策略不同**才单独建类；同策略归一类。
"""
from __future__ import annotations


class MathAgentError(Exception):
    """所有自定义错误的根。"""


class LLMError(MathAgentError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMValidationError(LLMError):
    pass


class LLMTransportError(LLMError):
    pass


class RunnerError(MathAgentError):
    pass


class RunnerTimeoutError(RunnerError):
    pass


class RunnerRuntimeError(RunnerError):
    pass


class LatexError(MathAgentError):
    pass


class LatexMissingBinaryError(LatexError):
    pass


class LatexCompileError(LatexError):
    pass


_RATE_LIMIT_HINTS = ("RateLimitError", "rate limit", "429")
_TRANSPORT_HINTS = ("APIConnectionError", "Timeout", "ReadTimeout", "ConnectionError")


def classify_exception(e: BaseException) -> MathAgentError:
    """把任意异常归一化为 MathAgentError 子类。

    - 已是 MathAgentError → 原样返回
    - 类名/消息命中 rate limit / transport → 对应子类
    - 否则视为通用 LLMError（调用方决定是否重试）
    """
    if isinstance(e, MathAgentError):
        return e
    name = type(e).__name__
    msg = str(e)
    blob = f"{name} {msg}"
    if any(h in blob for h in _RATE_LIMIT_HINTS):
        return LLMRateLimitError(msg)
    if any(h in blob for h in _TRANSPORT_HINTS):
        return LLMTransportError(msg)
    return LLMError(msg)
```

- [ ] **Step 4：跑测试**

Run: `pytest tests/test_errors.py -v`
Expected: 4 passed.

- [ ] **Step 5：提交**

```bash
git add src/math_agent/errors.py tests/test_errors.py
git commit -m "feat: unified error hierarchy with classification helper"
```

---

### Task 1.2：retry 装饰器工厂（先写测试）

**Files:**
- Create: `tests/test_retry.py`
- Create: `src/math_agent/retry.py`

- [ ] **Step 1：写失败的测试**

```python
import pytest
from math_agent.errors import (
    LLMRateLimitError, LLMTransportError, LLMValidationError, LatexMissingBinaryError,
)
from math_agent.retry import llm_retry, runner_retry


def test_llm_retry_retries_rate_limit_then_succeeds():
    calls = []

    @llm_retry(max_attempts=3, base_delay=0)
    def f():
        calls.append(1)
        if len(calls) < 3:
            raise LLMRateLimitError("429")
        return "ok"

    assert f() == "ok"
    assert len(calls) == 3


def test_llm_retry_does_not_retry_validation_error():
    calls = []

    @llm_retry(max_attempts=3, base_delay=0)
    def f():
        calls.append(1)
        raise LLMValidationError("bad json")

    with pytest.raises(LLMValidationError):
        f()
    assert len(calls) == 1  # validation 不重试（由 llm.complete 内部喂回错误重试，而不是这里）


def test_llm_retry_gives_up_after_max():
    @llm_retry(max_attempts=2, base_delay=0)
    def f():
        raise LLMTransportError("net")

    with pytest.raises(LLMTransportError):
        f()


def test_runner_retry_does_not_retry_missing_binary():
    @runner_retry(max_attempts=3, base_delay=0)
    def f():
        raise LatexMissingBinaryError("no xelatex")  # 仅作为"不可重试"信号示意

    # runner_retry 只重试 RunnerError；其他错误透传，不重试
    with pytest.raises(LatexMissingBinaryError):
        f()
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/test_retry.py -v`
Expected: ImportError.

- [ ] **Step 3：实现 `retry.py`**

```python
"""统一重试装饰器，基于 tenacity。

设计原则：
- 只有 *可重试* 的错误才重试；其他错误透传，让调用方/上游处理。
- 不在装饰器内做 sleep；用 tenacity 的 wait_exponential。
- 装饰器是同步版本；如未来引入 async，再扩 async_llm_retry。
- max_attempts 默认从 config 读，避免与 Plan A 的 MAX_LLM_RETRIES 漂移；调用方可显式覆盖。

可重试集合：
- llm_retry: LLMRateLimitError, LLMTransportError
- runner_retry: RunnerError 的全部子类
"""
from __future__ import annotations

from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
)

from math_agent.errors import (
    LLMRateLimitError, LLMTransportError, RunnerError,
)


def _default_llm_attempts() -> int:
    # 与 Plan A 的 MAX_LLM_RETRIES 保持同源；MAX_LLM_RETRIES 是"次数"语义（首次 + N 次重试），
    # tenacity 的 stop_after_attempt 是"总尝试次数"语义，故 attempts = MAX_LLM_RETRIES + 1。
    from math_agent.config import MAX_LLM_RETRIES
    return MAX_LLM_RETRIES + 1


def llm_retry(*, max_attempts: int | None = None, base_delay: float = 1.0, max_delay: float = 30.0):
    attempts = max_attempts if max_attempts is not None else _default_llm_attempts()
    return retry(
        retry=retry_if_exception_type((LLMRateLimitError, LLMTransportError)),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=base_delay, min=base_delay, max=max_delay),
        reraise=True,
    )


def runner_retry(*, max_attempts: int = 2, base_delay: float = 0.5):
    return retry(
        retry=retry_if_exception_type(RunnerError),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=base_delay, min=base_delay, max=5.0),
        reraise=True,
    )
```

- [ ] **Step 4：在 `pyproject.toml` 的 dependencies 中加 `"tenacity>=8.2,<9"`**

执行：

```bash
pip install -e ".[dev]"
```

- [ ] **Step 5：跑测试**

Run: `pytest tests/test_retry.py -v`
Expected: 4 passed.

- [ ] **Step 6：提交**

```bash
git add src/math_agent/retry.py tests/test_retry.py pyproject.toml
git commit -m "feat: tenacity-based retry decorators (llm_retry, runner_retry)"
```

---

### Task 1.3：把 `llm.complete` 接入新错误体系 + 重试

**Files:**
- Modify: `src/math_agent/llm.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1：在 `tests/test_llm.py` 追加测试**

```python
def test_complete_classifies_and_retries_rate_limit(mocker):
    """litellm 抛 RateLimitError → 应被 classify 为 LLMRateLimitError → 触发重试。"""
    class _RL(Exception):
        __name__ = "RateLimitError"

    seq = [_RL("429"), mocker.MagicMock(
        choices=[mocker.MagicMock(message=mocker.MagicMock(content="ok"))]
    )]
    mocker.patch("litellm.completion", side_effect=seq)
    out = llm.complete("hi", model="gpt-4o-mini")
    assert out == "ok"


def test_complete_raises_llm_error_when_all_retries_exhausted(mocker):
    from math_agent.errors import LLMError
    class _RL(Exception):
        __name__ = "RateLimitError"

    mocker.patch("litellm.completion", side_effect=_RL("429"))
    # 把重试次数压到 2 以便测试快速
    with pytest.raises(LLMError):
        llm.complete("hi", model="gpt-4o-mini", max_retries=0,
                     _retry_attempts=2, _retry_base_delay=0)
```

> 注：测试需要 `import pytest` 与 `from math_agent.errors import LLMError`，请在文件顶部补齐。

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/test_llm.py::test_complete_classifies_and_retries_rate_limit tests/test_llm.py::test_complete_raises_llm_error_when_all_retries_exhausted -v`
Expected: 都 FAIL（litellm 异常未被 classify）。

- [ ] **Step 3：修改 `llm.py`**

替换 `complete()` 内部的 `try/except Exception as e: last_err = e; continue` 为：

```python
        try:
            raw = _completion_with_retry(model=model, messages=msgs, temperature=temperature,
                                         response_format=response_format, **kwargs)
        except MathAgentError as e:
            # 已达重试上限或非可重试错误
            raise
        except Exception as e:
            raise classify_exception(e)
```

并在文件顶部 import 与定义 `_completion_with_retry`：

```python
from math_agent.errors import (
    LLMError, LLMValidationError, classify_exception, MathAgentError,
)
from math_agent.retry import llm_retry


def _do_completion(**kw):
    try:
        return litellm.completion(**kw)
    except Exception as e:
        raise classify_exception(e)


def _completion_with_retry(*, _retry_attempts=None, _retry_base_delay=1.0, **kw):
    # llm_retry 默认从 config.MAX_LLM_RETRIES 派生 attempts；
    # 调用方（如测试）可显式传 _retry_attempts 覆盖。
    @llm_retry(max_attempts=_retry_attempts, base_delay=_retry_base_delay)
    def _call():
        return _do_completion(**kw)
    return _call()
```

同时把签名上加上 `_retry_attempts: int | None = None, _retry_base_delay: float = 1.0,`，并把 `_completion_with_retry(...)` 调用处传入这两个参数。**`_retry_attempts=None` 让 `llm_retry` 从 `config.MAX_LLM_RETRIES` 派生 attempts，避免重试预算两处定义漂移。**

把 schema 解析失败的分支由 `last_err = e; continue` 改为：

```python
        try:
            return schema.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError) as e:
            last_err = LLMValidationError(str(e))
            last_content = content
            continue
```

最终循环耗尽时：

```python
    raise LLMError(f"LLM 调用失败：{last_err}")
```

- [ ] **Step 4：跑所有 LLM 测试**

Run: `pytest tests/test_llm.py -v`
Expected: 全绿（含 Plan A/B 旧 4 个 + Plan C 新 2 个 = 6 passed）。

- [ ] **Step 5：提交**

```bash
git add src/math_agent/llm.py tests/test_llm.py
git commit -m "feat(llm): classify exceptions + tenacity retry for rate limit/transport"
```

---

### Task 1.4：把 runner 与 latex 错误也分类化

**Files:**
- Modify: `src/math_agent/tools/runner.py`
- Modify: `src/math_agent/tools/latex_compile.py`
- Modify: `tests/test_runner.py`
- Modify: `tests/tools/test_latex_compile.py`

- [ ] **Step 1：在 `tests/test_runner.py` 追加分类测试**

```python
from math_agent.errors import RunnerTimeoutError, RunnerRuntimeError


def test_runner_result_carries_error_kind_on_timeout(workdir):
    from math_agent.tools.runner import run_python
    res = run_python("import time; time.sleep(30)", workdir=workdir, timeout=1)
    assert not res.success
    assert res.error_kind == "timeout"


def test_runner_result_carries_error_kind_on_runtime(workdir):
    from math_agent.tools.runner import run_python
    res = run_python("raise ValueError('x')", workdir=workdir)
    assert not res.success
    assert res.error_kind == "runtime"
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/test_runner.py -v`
Expected: 两个新用例 FAIL（缺 `error_kind`）。

- [ ] **Step 3：修改 `runner.py`**

把 `RunResult`（Plan A 已落地的类名）扩字段：

```python
@dataclass
class RunResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    artifact_paths: list[str] = field(default_factory=list)
    error_kind: str = ""  # "" | "timeout" | "runtime"
```

把 `run_python` 的 except 分支改为返回 `error_kind="timeout"`；执行失败（`returncode != 0`）的分支改为 `error_kind="runtime"`：

```python
    except subprocess.TimeoutExpired as e:
        return RunResult(
            success=False, stdout=e.stdout or "",
            stderr=f"timeout after {timeout}s", error_kind="timeout",
        )

    ...
    return RunResult(
        success=proc.returncode == 0,
        stdout=proc.stdout, stderr=proc.stderr,
        artifact_paths=[str(workdir / n) for n in new_files],
        error_kind="" if proc.returncode == 0 else "runtime",
    )
```

- [ ] **Step 4：在 `tests/tools/test_latex_compile.py` 追加分类测试**

```python
def test_latex_result_carries_error_kind_when_missing(monkeypatch, workdir):
    from math_agent.tools.latex_compile import compile_latex
    monkeypatch.setattr("shutil.which", lambda _: None)
    tex = workdir / "main.tex"
    tex.write_text(r"\documentclass{article}\begin{document}x\end{document}", encoding="utf-8")
    res = compile_latex(tex)
    assert res.error_kind == "missing_binary"
```

- [ ] **Step 5：修改 `latex_compile.py`**

把 `LatexResult` 改为：

```python
@dataclass
class LatexResult:
    success: bool
    pdf_path: str = ""
    log: str = ""
    error_kind: str = ""  # "" | "missing_binary" | "compile" | "timeout"
```

把对应分支补上 `error_kind`：

```python
    if shutil.which("xelatex") is None:
        return LatexResult(success=False, log="xelatex not found on PATH",
                           error_kind="missing_binary")
    ...
            if proc.returncode != 0:
                return LatexResult(success=False, log="\n".join(log_acc),
                                   error_kind="compile")
    ...
        if not pdf.exists():
            return LatexResult(success=False, log="\n".join(log_acc) + "\nno pdf produced",
                               error_kind="compile")
    ...
    except subprocess.TimeoutExpired as e:
        return LatexResult(success=False, log=f"timeout after {timeout}s: {e}",
                           error_kind="timeout")
```

- [ ] **Step 6：跑两组测试**

Run: `pytest tests/test_runner.py tests/tools/test_latex_compile.py -v`
Expected: 全绿。

- [ ] **Step 7：提交**

```bash
git add src/math_agent/tools/runner.py src/math_agent/tools/latex_compile.py tests/test_runner.py tests/tools/test_latex_compile.py
git commit -m "feat(tools): structured error_kind on sandbox/latex results"
```

---

## Phase 2：RAG 知识库

### Task 2.1：chunking（先写测试）

**Files:**
- Create: `tests/rag/__init__.py`
- Create: `tests/rag/test_chunking.py`
- Create: `src/math_agent/rag/__init__.py`
- Create: `src/math_agent/rag/chunking.py`

- [ ] **Step 1：建包**

```bash
mkdir -p src/math_agent/rag tests/rag
touch src/math_agent/rag/__init__.py tests/rag/__init__.py
```

- [ ] **Step 2：写失败的测试**

```python
from math_agent.rag.chunking import chunk_text, Chunk


def test_chunk_text_returns_chunks_with_overlap():
    text = "abcdefghij" * 50  # 500 字符
    chunks = chunk_text(text, max_chars=200, overlap=50, source="x.md")
    assert all(isinstance(c, Chunk) for c in chunks)
    # 长度大致符合（除末尾）
    assert all(len(c.text) <= 200 for c in chunks)
    # 相邻块有 overlap
    assert chunks[0].text[-50:] == chunks[1].text[:50]
    # 源信息透传
    assert all(c.source == "x.md" for c in chunks)


def test_chunk_text_short_input_one_chunk():
    chunks = chunk_text("hello", max_chars=200, overlap=50, source="s")
    assert len(chunks) == 1
    assert chunks[0].text == "hello"


def test_chunk_text_respects_paragraph_boundary_when_possible():
    para = "段落一" * 50
    text = para + "\n\n" + "段落二" * 50
    chunks = chunk_text(text, max_chars=180, overlap=20, source="s")
    # 至少在段落分隔处切了一刀
    assert any(c.text.endswith("段落一" * (180 // 3)) or "段落二" in c.text for c in chunks)
```

- [ ] **Step 3：跑测试，确认失败**

Run: `pytest tests/rag/test_chunking.py -v`
Expected: ImportError.

- [ ] **Step 4：实现 `chunking.py`**

```python
"""文本切块：固定窗口 + 重叠，优先在段落 / 句子边界切。

接口刻意保持简单：不依赖 langchain 或 llama-index。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    index: int  # 在该 source 内的顺序


_PREFERRED_BREAKS = ("\n\n", "。", "\n", "，", " ")


def _best_break(s: str, near: int) -> int:
    """在 [near*0.7, near] 范围里找首选边界，找不到就返回 near。"""
    lo = max(0, int(near * 0.7))
    for sep in _PREFERRED_BREAKS:
        idx = s.rfind(sep, lo, near)
        if idx != -1:
            return idx + len(sep)
    return near


def chunk_text(text: str, *, max_chars: int, overlap: int, source: str) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")

    chunks: list[Chunk] = []
    n = len(text)
    if n == 0:
        return chunks

    i = 0
    idx = 0
    while i < n:
        end = min(i + max_chars, n)
        if end < n:
            end = i + _best_break(text[i:end], max_chars)
        chunks.append(Chunk(text=text[i:end], source=source, index=idx))
        idx += 1
        if end >= n:
            break
        i = max(end - overlap, i + 1)  # 防止 overlap 导致原地踏步
    return chunks
```

- [ ] **Step 5：跑测试**

Run: `pytest tests/rag/test_chunking.py -v`
Expected: 3 passed.

- [ ] **Step 6：提交**

```bash
git add src/math_agent/rag/__init__.py src/math_agent/rag/chunking.py tests/rag/__init__.py tests/rag/test_chunking.py
git commit -m "feat(rag): text chunking with overlap and paragraph-aware breaks"
```

---

### Task 2.2：embeddings 封装（先写测试）

**Files:**
- Create: `tests/rag/test_embeddings.py`
- Create: `src/math_agent/rag/embeddings.py`

- [ ] **Step 1：写失败的测试**

```python
from math_agent.rag.embeddings import embed_texts


def test_embed_texts_calls_litellm_embedding(mocker):
    fake = mocker.MagicMock()
    fake.data = [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]
    mocker.patch("litellm.embedding", return_value=fake)

    out = embed_texts(["a", "b"], model="text-embedding-3-small")
    assert out == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_embed_texts_batches_when_over_limit(mocker):
    """batch_size=2，3 段文本应切成 2 次调用。"""
    call_count = {"n": 0}

    def _fake(model, input, **kw):
        call_count["n"] += 1
        return type("R", (), {
            "data": [{"embedding": [float(call_count["n"])] * 3} for _ in input]
        })()

    mocker.patch("litellm.embedding", side_effect=_fake)
    out = embed_texts(["x", "y", "z"], model="m", batch_size=2)
    assert call_count["n"] == 2
    assert len(out) == 3
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/rag/test_embeddings.py -v`
Expected: ImportError.

- [ ] **Step 3：实现 `embeddings.py`**

```python
"""litellm.embedding 包装：支持批处理 + 经 retry 装饰。"""
from __future__ import annotations

import litellm

from math_agent.errors import classify_exception
from math_agent.retry import llm_retry


def _do_embed(model: str, input: list[str]) -> list[list[float]]:
    try:
        resp = litellm.embedding(model=model, input=input)
    except Exception as e:
        raise classify_exception(e)
    return [item["embedding"] for item in resp.data]


@llm_retry(max_attempts=4, base_delay=1.0)
def _embed_with_retry(model: str, input: list[str]) -> list[list[float]]:
    return _do_embed(model, input)


def embed_texts(texts: list[str], *, model: str, batch_size: int = 64) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        out.extend(_embed_with_retry(model, texts[i : i + batch_size]))
    return out
```

- [ ] **Step 4：跑测试**

Run: `pytest tests/rag/test_embeddings.py -v`
Expected: 2 passed.

- [ ] **Step 5：提交**

```bash
git add src/math_agent/rag/embeddings.py tests/rag/test_embeddings.py
git commit -m "feat(rag): embed_texts with batching and unified retry"
```

---

### Task 2.3：sqlite-vec store（先写测试）

**Files:**
- Create: `tests/rag/test_store.py`
- Create: `src/math_agent/rag/store.py`

- [ ] **Step 1：写失败的测试**

```python
from math_agent.rag.store import VectorStore, StoredChunk
from math_agent.rag.chunking import Chunk


def test_store_round_trip(workdir):
    db = workdir / "vec.db"
    store = VectorStore.open(db, dim=3)
    store.add(
        chunks=[Chunk(text="alpha", source="a.md", index=0),
                Chunk(text="beta", source="a.md", index=1)],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    hits = store.search([1.0, 0.0, 0.0], k=2)
    assert isinstance(hits[0], StoredChunk)
    # 最近邻是 alpha
    assert hits[0].text == "alpha"
    assert len(hits) == 2


def test_store_persists_across_open(workdir):
    db = workdir / "vec.db"
    s1 = VectorStore.open(db, dim=3)
    s1.add(
        chunks=[Chunk(text="x", source="s", index=0)],
        embeddings=[[1.0, 0.0, 0.0]],
    )
    s1.close()

    s2 = VectorStore.open(db, dim=3)
    hits = s2.search([1.0, 0.0, 0.0], k=1)
    assert hits[0].text == "x"


def test_store_rejects_dim_mismatch(workdir):
    import pytest
    store = VectorStore.open(workdir / "vec.db", dim=3)
    with pytest.raises(ValueError):
        store.add(
            chunks=[Chunk(text="x", source="s", index=0)],
            embeddings=[[1.0, 0.0]],  # 维度 2，不匹配
        )
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/rag/test_store.py -v`
Expected: ImportError.

- [ ] **Step 3：在 `pyproject.toml` 加 `"sqlite-vec>=0.1,<0.2"`**，并 `pip install -e ".[dev]"`。

- [ ] **Step 4：实现 `store.py`**

```python
"""sqlite-vec 向量索引。

设计：
- 单进程使用，单文件 sqlite。
- chunks 和向量存两张表，主键 id 关联（避免把文本塞进 vec 表）。
"""
from __future__ import annotations

import json
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from math_agent.rag.chunking import Chunk


@dataclass
class StoredChunk:
    id: int
    text: str
    source: str
    index: int
    score: float


def _to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class VectorStore:
    def __init__(self, conn: sqlite3.Connection, dim: int):
        self._conn = conn
        self._dim = dim

    @classmethod
    def open(cls, path: str | Path, *, dim: int) -> "VectorStore":
        conn = sqlite3.connect(str(path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "text TEXT NOT NULL, source TEXT NOT NULL, idx INTEGER NOT NULL)"
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"id INTEGER PRIMARY KEY, embedding float[{dim}])"
        )
        conn.commit()
        return cls(conn, dim)

    def add(self, *, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        for e in embeddings:
            if len(e) != self._dim:
                raise ValueError(f"embedding dim {len(e)} != store dim {self._dim}")
        cur = self._conn.cursor()
        for c, e in zip(chunks, embeddings):
            cur.execute(
                "INSERT INTO chunks(text, source, idx) VALUES (?, ?, ?)",
                (c.text, c.source, c.index),
            )
            rowid = cur.lastrowid
            cur.execute("INSERT INTO vec_chunks(id, embedding) VALUES (?, ?)",
                        (rowid, _to_blob(e)))
        self._conn.commit()

    def search(self, query: list[float], *, k: int = 5) -> list[StoredChunk]:
        if len(query) != self._dim:
            raise ValueError(f"query dim {len(query)} != store dim {self._dim}")
        cur = self._conn.execute(
            "SELECT v.id, c.text, c.source, c.idx, v.distance "
            "FROM vec_chunks v JOIN chunks c ON c.id = v.id "
            "WHERE v.embedding MATCH ? ORDER BY v.distance LIMIT ?",
            (_to_blob(query), k),
        )
        return [
            StoredChunk(id=row[0], text=row[1], source=row[2], index=row[3], score=row[4])
            for row in cur
        ]

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 5：跑测试**

Run: `pytest tests/rag/test_store.py -v`
Expected: 3 passed.

- [ ] **Step 6：提交**

```bash
git add src/math_agent/rag/store.py tests/rag/test_store.py pyproject.toml
git commit -m "feat(rag): sqlite-vec backed vector store"
```

---

### Task 2.4：retrieve 高层 API（先写测试）

**Files:**
- Create: `tests/rag/test_retrieve.py`
- Create: `src/math_agent/rag/retrieve.py`

- [ ] **Step 1：写失败的测试**

```python
from math_agent.rag.retrieve import search, Snippet


def test_search_returns_empty_when_db_missing(workdir):
    out = search("hello", db_path=workdir / "nonexistent.db", k=3)
    assert out == []


def test_search_returns_snippets(mocker, workdir):
    from math_agent.rag.store import VectorStore
    from math_agent.rag.chunking import Chunk

    store = VectorStore.open(workdir / "vec.db", dim=3)
    store.add(
        chunks=[Chunk(text="alpha", source="s", index=0)],
        embeddings=[[1.0, 0.0, 0.0]],
    )
    store.close()

    mocker.patch(
        "math_agent.rag.retrieve.embed_texts",
        return_value=[[1.0, 0.0, 0.0]],
    )

    out = search("query", db_path=workdir / "vec.db", k=1,
                 embedding_model="m", dim=3)
    assert len(out) == 1
    assert isinstance(out[0], Snippet)
    assert out[0].text == "alpha"
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/rag/test_retrieve.py -v`
Expected: ImportError.

- [ ] **Step 3：实现 `retrieve.py`**

```python
"""统一检索入口；节点只 import 这一个函数。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from math_agent.rag.embeddings import embed_texts
from math_agent.rag.store import VectorStore


@dataclass
class Snippet:
    text: str
    source: str
    score: float


def search(
    query: str,
    *,
    db_path: str | Path,
    k: int = 5,
    embedding_model: str = "text-embedding-3-small",
    dim: int = 1536,
) -> list[Snippet]:
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    vec = embed_texts([query], model=embedding_model)[0]
    store = VectorStore.open(db_path, dim=dim)
    try:
        rows = store.search(vec, k=k)
    finally:
        store.close()
    return [Snippet(text=r.text, source=r.source, score=r.score) for r in rows]


def format_snippets(snippets: list[Snippet]) -> str:
    """供 prompt 拼接的统一格式。"""
    if not snippets:
        return ""
    parts = ["# 检索到的参考资料（仅供启发，不可照抄）"]
    for i, s in enumerate(snippets, 1):
        parts.append(f"## [{i}] 来源：{s.source}\n{s.text}")
    return "\n\n".join(parts)
```

- [ ] **Step 4：跑测试**

Run: `pytest tests/rag/test_retrieve.py -v`
Expected: 2 passed.

- [ ] **Step 5：提交**

```bash
git add src/math_agent/rag/retrieve.py tests/rag/test_retrieve.py
git commit -m "feat(rag): high-level search() + format_snippets()"
```

---

### Task 2.5：ingest 流水线（先写测试）

**Files:**
- Create: `tests/rag/test_ingest.py`
- Create: `src/math_agent/rag/ingest.py`

- [ ] **Step 1：写失败的测试**

```python
from pathlib import Path
from math_agent.rag.ingest import ingest_directory


def test_ingest_directory_processes_md_files(mocker, workdir):
    corpus = workdir / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("段落一\n\n段落二", encoding="utf-8")
    (corpus / "b.md").write_text("内容 b", encoding="utf-8")

    captured = {"count": 0}

    def _fake_embed(texts, *, model, batch_size=64):
        captured["count"] += len(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]

    mocker.patch("math_agent.rag.ingest.embed_texts", side_effect=_fake_embed)

    db = workdir / "vec.db"
    report = ingest_directory(
        src_dir=corpus, db_path=db,
        embedding_model="m", dim=3,
        max_chars=200, overlap=20,
    )
    assert report.files_processed == 2
    assert report.chunks_added >= 2
    assert db.exists()
    assert captured["count"] == report.chunks_added


def test_ingest_directory_handles_pdf_via_pypdf(mocker, workdir):
    """PDF 文件用 pypdf 抽文本；这里只确认调用路径不抛。"""
    corpus = workdir / "corpus"
    corpus.mkdir()
    fake_pdf = corpus / "doc.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    mocker.patch("math_agent.rag.ingest._extract_pdf_text", return_value="pdf 内容")
    mocker.patch("math_agent.rag.ingest.embed_texts",
                 side_effect=lambda texts, **kw: [[1.0, 0.0, 0.0]] * len(texts))

    rep = ingest_directory(src_dir=corpus, db_path=workdir / "v.db",
                           embedding_model="m", dim=3, max_chars=200, overlap=20)
    assert rep.files_processed == 1
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/rag/test_ingest.py -v`
Expected: ImportError.

- [ ] **Step 3：在 `pyproject.toml` 加 `"pypdf>=4.0,<5"`，安装。**

- [ ] **Step 4：实现 `ingest.py`**

```python
"""离线 ingest：扫描语料目录 → 切块 → 嵌入 → 入库。

支持后缀：.md, .txt, .pdf
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from math_agent.rag.chunking import chunk_text
from math_agent.rag.embeddings import embed_texts
from math_agent.rag.store import VectorStore


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


@dataclass
class IngestReport:
    files_processed: int
    chunks_added: int
    skipped: list[str]


def _extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _extract_pdf_text(path)
    return path.read_text(encoding="utf-8")


def ingest_directory(
    *,
    src_dir: str | Path,
    db_path: str | Path,
    embedding_model: str,
    dim: int,
    max_chars: int = 1200,
    overlap: int = 200,
) -> IngestReport:
    src_dir = Path(src_dir)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    store = VectorStore.open(db_path, dim=dim)
    files_processed = 0
    chunks_added = 0
    skipped: list[str] = []
    try:
        for p in sorted(src_dir.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            try:
                text = _read_file(p)
            except Exception as e:
                skipped.append(f"{p}: {e}")
                continue
            chunks = chunk_text(text, max_chars=max_chars, overlap=overlap, source=str(p))
            if not chunks:
                continue
            embeddings = embed_texts([c.text for c in chunks], model=embedding_model)
            store.add(chunks=chunks, embeddings=embeddings)
            files_processed += 1
            chunks_added += len(chunks)
    finally:
        store.close()
    return IngestReport(files_processed=files_processed,
                        chunks_added=chunks_added, skipped=skipped)
```

- [ ] **Step 5：跑测试**

Run: `pytest tests/rag/test_ingest.py -v`
Expected: 2 passed.

- [ ] **Step 6：提交**

```bash
git add src/math_agent/rag/ingest.py tests/rag/test_ingest.py pyproject.toml
git commit -m "feat(rag): ingest_directory for md/txt/pdf corpora"
```

---

### Task 2.6：把 RAG 注入 Analyst / Modeler / Writer prompts

**Files:**
- Modify: `src/math_agent/prompts/analyst.py`
- Modify: `src/math_agent/prompts/modeler.py`
- Modify: `src/math_agent/prompts/writer.py`
- Modify: `src/math_agent/nodes/analyst.py`
- Modify: `src/math_agent/nodes/modeler.py`
- Modify: `src/math_agent/nodes/writer.py`
- Modify: `src/math_agent/config.py`

> **Plan B 警告**：当前 writer prompt（`templates/writer_prompt.md.j2`）已有 6 条 IRON RULES + 上一轮 critic feedback 注入，已接近 token 预算上限。本任务注入的 `retrieved_context` 长度**强制截到 800 字符以内**（约 600 token），避免推爆 8k 上下文。analyst / modeler prompt 较短，可放宽到 1500 字符。

- [ ] **Step 1：在 `config.py` 增加 RAG 开关**

在 `MODEL_ROUTING = {...}` 下方追加：

```python
# RAG
RAG_ENABLED = os.getenv("MATH_AGENT_RAG_ENABLED", "0") == "1"
RAG_DB_PATH = os.getenv("MATH_AGENT_RAG_DB", str(PROJECT_ROOT / "runs" / "rag.sqlite"))
RAG_EMBEDDING_MODEL = os.getenv("MATH_AGENT_RAG_EMBED", "text-embedding-3-small")
RAG_EMBEDDING_DIM = int(os.getenv("MATH_AGENT_RAG_DIM", "1536"))
RAG_TOPK = int(os.getenv("MATH_AGENT_RAG_TOPK", "4"))
```

- [ ] **Step 2：在三个 prompt 的 `build_prompt` 上加 `retrieved_context: str = ""` 参数**

例如 `analyst.py`：

```python
def build_prompt(problem: str, background: str, questions: list[str],
                 retrieved_context: str = "") -> str:
    qs = "\n".join(f"- {q}" for q in questions) or "（题目本身未列出独立小问）"
    ctx = f"\n{retrieved_context}\n" if retrieved_context else ""
    return (
        f"# 题目\n{problem}\n\n"
        f"# 背景\n{background or '（无）'}\n\n"
        f"# 小问\n{qs}\n"
        f"{ctx}"
        f"请输出 JSON：..."  # 原内容
    )
```

`modeler.py` 与 `writer.py` 做同样的扩展（在原 prompt 的最末尾的 `请输出 JSON：` 之前插入 `{ctx}`）。

- [ ] **Step 3：在三个节点中调用 RAG**

`nodes/analyst.py` 替换 `prompt = build_prompt(...)`：

```python
from math_agent.config import (
    MODEL_ROUTING, RAG_ENABLED, RAG_DB_PATH, RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_DIM, RAG_TOPK,
)
from math_agent.rag.retrieve import search, format_snippets


def analyst_node(state: MathModelingState) -> dict:
    ctx = ""
    if RAG_ENABLED:
        snippets = search(
            state.problem,
            db_path=RAG_DB_PATH, k=RAG_TOPK,
            embedding_model=RAG_EMBEDDING_MODEL, dim=RAG_EMBEDDING_DIM,
        )
        ctx = format_snippets(snippets)
    prompt = build_prompt(state.problem, state.background, state.questions, retrieved_context=ctx)
    ...
```

`modeler_node` 与 `writer_node` 同形扩展。**查询字符串构造**（注意 `basic` 阶段 `latest_model()` 为 None）：

```python
# modeler_node：
prev = state.latest_model()
prev_desc = prev.description if prev else ""
query = (state.problem + " " + state.stage_target + " " + prev_desc).strip()

# writer_node：
query = state.problem + " " + (state.paper.model_section or "")[:500]
```

- [ ] **Step 4：补测试，验证 RAG 关闭时行为不变 + 打开时查询被调用**

在 `tests/nodes/test_analyst.py` 末尾追加：

```python
def test_analyst_does_not_query_rag_when_disabled(mocker):
    mocker.patch("math_agent.nodes.analyst.RAG_ENABLED", False)
    spy = mocker.patch("math_agent.nodes.analyst.search")
    mocker.patch(
        "math_agent.nodes.analyst.complete",
        return_value=AnalystOutput(assumptions=[]),
    )
    analyst_node(MathModelingState(problem="p"))
    spy.assert_not_called()


def test_analyst_queries_rag_when_enabled(mocker):
    mocker.patch("math_agent.nodes.analyst.RAG_ENABLED", True)
    mocker.patch("math_agent.nodes.analyst.RAG_DB_PATH", "/tmp/nonexistent.db")
    spy = mocker.patch("math_agent.nodes.analyst.search", return_value=[])
    mocker.patch(
        "math_agent.nodes.analyst.complete",
        return_value=AnalystOutput(assumptions=[]),
    )
    analyst_node(MathModelingState(problem="p"))
    spy.assert_called_once()
```

对 `tests/nodes/test_modeler.py` / `tests/nodes/test_writer.py` 做完全同形的扩展（替换变量名 `analyst` 为对应节点名）。

- [ ] **Step 5：跑测试**

Run: `pytest tests/nodes -v`
Expected: 所有节点测试全绿（Plan A/B 已有 + 6 个新增 = 全过）。

- [ ] **Step 6：提交**

```bash
git add src/math_agent/prompts/ src/math_agent/nodes/ src/math_agent/config.py tests/nodes/
git commit -m "feat(rag): inject retrieved context into analyst/modeler/writer prompts"
```

---

## Phase 3：历年题回归基准

> **架构约束**：bench 的 mock fixture **不**进 `src/math_agent/bench/`，避免生产代码 import
> `unittest.mock`。结构：
> - `src/math_agent/bench/`：仅放数据（题目 JSON、expectations.json）+ 纯 runner（live 模式真跑）
> - `tests/bench/`：放 mock fixture + mock 模式测试
> - CLI `bench` 子命令调 `run_bench(mode="live", ...)`；要测 mock 模式跑 `pytest tests/bench/`

### Task 3.1：bench 数据与期望

**Files:**
- Create: `src/math_agent/bench/__init__.py`
- Create: `src/math_agent/bench/problems/2022_A.json`
- Create: `src/math_agent/bench/problems/2023_B.json`
- Create: `src/math_agent/bench/expectations.json`

- [ ] **Step 1：建包并写两道示例题（schema 与 `sample_problem.json` 一致）**

```bash
mkdir -p src/math_agent/bench/problems tests/bench
touch src/math_agent/bench/__init__.py tests/bench/__init__.py
```

`src/math_agent/bench/problems/2022_A.json`：

```json
{
  "title": "无人机蜂群协同区域覆盖",
  "background": "某地需在 30 分钟内完成 25 km² 区域的图像覆盖采集。",
  "questions": [
    "建立模型以最少无人机数量完成覆盖。",
    "考虑风扰下的鲁棒调度。"
  ],
  "data_files": []
}
```

`src/math_agent/bench/problems/2023_B.json`：

```json
{
  "title": "城市暴雨内涝点风险评估与排水调度",
  "background": "某城市 2023 年发生多起内涝事件。",
  "questions": [
    "建立内涝点风险评估模型。",
    "在排水设施约束下给出调度方案。"
  ],
  "data_files": []
}
```

`src/math_agent/bench/expectations.json`：

```json
{
  "2022_A": {
    "min_overall": 6.5,
    "must_contain_keywords": ["覆盖", "无人机", "鲁棒"],
    "must_have_sensitivity": true,
    "must_have_figures": true
  },
  "2023_B": {
    "min_overall": 6.5,
    "must_contain_keywords": ["内涝", "排水", "风险"],
    "must_have_sensitivity": true,
    "must_have_figures": true
  }
}
```

- [ ] **Step 2：提交**

```bash
git add src/math_agent/bench/__init__.py src/math_agent/bench/problems/ src/math_agent/bench/expectations.json tests/bench/__init__.py
git commit -m "feat(bench): seed two historical problems and expectations"
```

---

### Task 3.2：bench runner（live 模式真跑器）+ 测试侧 mock harness

**Files:**
- Create: `src/math_agent/bench/runner.py`           # 纯 live：调 build_graph 真跑、判定通过
- Create: `tests/bench/conftest.py`                  # mock harness（ExitStack + unittest.mock.patch）
- Create: `tests/bench/test_runner_mock.py`          # 用 mock harness 验证 runner 判定逻辑

> bench 的 mock fixture 与 `from PIL/unittest.mock import` 这类只在测试时需要的依赖都在
> `tests/bench/` 下，**不**污染 `src/math_agent/bench/`。

- [ ] **Step 1：写失败的测试**

`tests/bench/test_runner_mock.py`：

```python
import json
from math_agent.bench.runner import run_bench, BenchReport, BenchCase


def test_run_bench_returns_one_case_per_problem(workdir, install_bench_mocks):
    """install_bench_mocks 是 conftest.py 里的 fixture，进入测试时自动 patch 各节点 complete。"""
    rep = run_bench(out_dir=workdir)
    assert isinstance(rep, BenchReport)
    assert {c.problem_id for c in rep.cases} == {"2022_A", "2023_B"}


def test_run_bench_marks_pass_when_expectation_met(workdir, install_bench_mocks):
    """install_bench_mocks 的默认 fixture 配置成"勉强达标"，所有题应 PASS。"""
    rep = run_bench(out_dir=workdir)
    for case in rep.cases:
        assert case.passed, f"{case.problem_id}: {case.failures}"


def test_run_bench_marks_fail_when_keyword_missing(workdir, install_bench_mocks_missing_keyword):
    """另一份 fixture 把 mock paper 的 keyword 抠掉，应导致 FAIL。"""
    rep = run_bench(out_dir=workdir)
    fails = [c for c in rep.cases if not c.passed]
    assert fails, "expected at least one FAIL when keyword missing"
    assert any("missing keyword" in " ".join(c.failures) for c in fails)


def test_run_bench_marks_fail_when_overall_below_threshold(workdir, install_bench_mocks_low_overall):
    """另一份 fixture 让 evaluation overall=3.0，远低于 min_overall=6.5，应导致 FAIL。"""
    rep = run_bench(out_dir=workdir)
    fails = [c for c in rep.cases if not c.passed]
    assert fails
    assert any("overall" in " ".join(c.failures) for c in fails)


def test_run_bench_writes_report_json(workdir, install_bench_mocks):
    run_bench(out_dir=workdir)
    blob = json.loads((workdir / "bench_report.json").read_text(encoding="utf-8"))
    assert "cases" in blob and len(blob["cases"]) == 2
```

- [ ] **Step 2：写 `tests/bench/conftest.py`（mock harness）**

```python
"""bench mock harness：把 LLM/LaTeX 全部 mock 掉，runner 跑结构性流程。

放在 tests/ 下，避免 src/math_agent/bench/runner.py import unittest.mock。
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from math_agent.state import (
    Assumption, ModelVersion, CriticReport, PaperSections,
    EvaluationReport, HumanDecision,
)
from math_agent.nodes.analyst import AnalystOutput
from math_agent.nodes.coder import CoderDraft
from math_agent.nodes.figure_pipeline import FigureCriticOut, FigureAnalysisOut
from math_agent.nodes.sensitivity import SensitivityPlan, SensitivityCode, Interpretations


def _make_png(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 240), "white").save(p)
    return str(p)


def _make_paper(keywords: list[str]) -> PaperSections:
    kw = "、".join(keywords)
    body = f"本研究围绕 {kw} 展开，模型经过 basic→improved→final 演化。" * 5
    return PaperSections(
        abstract=body, problem_restatement=body, assumptions=body,
        notation=body, model_section=body, solution=body,
        sensitivity=body, conclusion=body, references="-",
    )


def _patch_all_nodes(stack: ExitStack, workdir: Path, paper_factory):
    """注入所有节点的 mock；paper_factory(keywords) 返回 PaperSections。"""
    fig_path = _make_png(workdir / "code" / "fig.png")
    sens_png = workdir / "sensitivity" / "p1.png"
    _make_png(sens_png)

    def _patch(target, **kw):
        stack.enter_context(patch(target, **kw))

    _patch("math_agent.nodes.analyst.complete",
           return_value=AnalystOutput(assumptions=[
               Assumption(statement="A", rationale="r", sensitivity_relevant=True)]))
    stage_box = {"i": 0}
    def _modeler_side(*a, **k):
        stages = ["basic", "improved", "final"]
        s = stages[min(stage_box["i"], 2)]
        stage_box["i"] += 1
        return ModelVersion(stage=s, description="d"*200)
    _patch("math_agent.nodes.modeler.complete", side_effect=_modeler_side)
    _patch("math_agent.nodes.model_critic.complete",
           return_value=CriticReport(target="modeler", score=9, approved=True))
    code = (
        "from pathlib import Path\n"
        f"Path(r'{fig_path}').parent.mkdir(parents=True, exist_ok=True)\n"
        "print('done')\n"
    )
    _patch("math_agent.nodes.coder.complete",
           return_value=CoderDraft(purpose="主结果", code=code))
    _patch("math_agent.nodes.sensitivity.complete", side_effect=[
        SensitivityPlan(runs=[{"parameter": "p1", "values": [1,2,3,4,5],
                               "metric": "y", "rationale": "r"}]),
        SensitivityCode(code=(
            "import matplotlib.pyplot as plt\n"
            "v=[1,2,3,4,5]; r=[x*2 for x in v]\n"
            "plt.plot(v,r); plt.savefig('p1.png')\n"
            "print(f'RESULT: parameter=p1 values={v} results={r}')\n"
        )),
        Interpretations(interpretations=["参数 p1 上升时 y 线性增长。"]),
    ])
    fc = FigureCriticOut(score=9, approved=True)
    fa = FigureAnalysisOut(analysis="趋势单调。")
    _patch("math_agent.nodes.figure_pipeline.complete",
           side_effect=[fc, fa, fc, fa])

    def _writer_side(*a, **k):
        # 从 prompt 里捞 keywords —— 简化：取当前题的固定 keywords
        return paper_factory(_writer_keywords[0] if _writer_keywords else [])
    _writer_keywords: list[list[str]] = []
    # 通过 monkey-patch 的方式让外层 set keywords
    stack.callback(lambda: _writer_keywords.clear())
    _patch("math_agent.nodes.writer.complete", side_effect=_writer_side)
    _patch("math_agent.nodes.paper_critic.complete",
           return_value=CriticReport(target="paper", score=9, approved=True))
    _patch("math_agent.nodes.evaluation.complete",
           return_value=EvaluationReport(
               assumption_reasonameableness=8, modeling_creativity=8,
               result_correctness=8, writing_clarity=8, extra_depth=8, overall=8.0))
    _patch("math_agent.nodes.latex.compile_latex",
           return_value=type("R", (), {"success": True, "pdf_path":"", "log":"",
                                       "error_kind": ""})())
    return _writer_keywords


@pytest.fixture
def install_bench_mocks(tmp_path):
    """默认 fixture：paper 必含 expectations 里的 keywords，所有题应 PASS。"""
    with ExitStack() as stack:
        keywords_holder = _patch_all_nodes(stack, tmp_path, _make_paper)
        # runner 内部按 problem 设置 keywords；为简单起见这里 set 公共集合
        keywords_holder.append(["覆盖", "无人机", "鲁棒", "内涝", "排水", "风险"])
        yield


@pytest.fixture
def install_bench_mocks_missing_keyword(tmp_path):
    """故意 paper 不含 keywords，验证 runner 能识别 FAIL。"""
    def _empty_paper(keywords):
        return _make_paper(keywords=["（缺失关键词的 paper）"])
    with ExitStack() as stack:
        _patch_all_nodes(stack, tmp_path, _empty_paper)
        yield


@pytest.fixture
def install_bench_mocks_low_overall(tmp_path):
    """让 evaluation overall=3.0，验证 runner 能识别 overall < min_overall 的失败路径。"""
    with ExitStack() as stack:
        keywords_holder = _patch_all_nodes(stack, tmp_path, _make_paper)
        keywords_holder.append(["覆盖", "无人机", "鲁棒", "内涝", "排水", "风险"])
        # 用 stack 再 patch evaluation 覆盖默认 8.0
        stack.enter_context(patch(
            "math_agent.nodes.evaluation.complete",
            return_value=EvaluationReport(
                assumption_reasonableness=2, modeling_creativity=3,
                result_correctness=3, writing_clarity=3, extra_depth=2, overall=3.0,
            ),
        ))
        yield
```

> 注：上面 `paper_factory` 与 `_writer_keywords` 的串联只是示意；实际实现里可以让
> runner 在每题前用 `monkeypatch.setattr` 重设 mock 返回值。**关键是 fixture 在 tests/ 里**。

- [ ] **Step 3：实现纯净的 `src/math_agent/bench/runner.py`（不含 mock）**

```python
"""回归基准 runner：真跑 build_graph，按 expectations.json 判定通过。

不引入测试库；mock 模式由 tests/bench/conftest.py 提供 fixture，本模块对此无感。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from math_agent.graph import build_graph
from math_agent.state import HumanDecision


_BENCH_ROOT = Path(__file__).resolve().parent
_PROBLEMS_DIR = _BENCH_ROOT / "problems"
_EXPECTATIONS = _BENCH_ROOT / "expectations.json"


@dataclass
class BenchCase:
    problem_id: str
    overall: float
    passed: bool
    failures: list[str] = field(default_factory=list)


@dataclass
class BenchReport:
    cases: list[BenchCase]


def _evaluate(case_id: str, final_state: dict, expect: dict) -> BenchCase:
    failures: list[str] = []
    overall = (final_state.get("evaluation").overall
               if final_state.get("evaluation") else 0.0)
    if overall < expect["min_overall"]:
        failures.append(f"overall {overall} < {expect['min_overall']}")
    paper = final_state.get("paper")
    text = " ".join([
        paper.abstract or "", paper.model_section or "", paper.solution or "",
        paper.conclusion or "",
    ]) if paper else ""
    for kw in expect.get("must_contain_keywords", []):
        if kw not in text:
            failures.append(f"missing keyword: {kw}")
    if expect.get("must_have_sensitivity") and not final_state.get("sensitivity_runs"):
        failures.append("missing sensitivity_runs")
    if expect.get("must_have_figures") and not final_state.get("figures"):
        failures.append("missing figures")
    return BenchCase(problem_id=case_id, overall=overall,
                     passed=not failures, failures=failures)


def run_bench(*, out_dir: str | Path) -> BenchReport:
    """真跑每道题；caller（pytest mock fixture 或 CLI 真 API key）负责提供 LLM。

    本函数不知道 LLM 是真是假——它只 invoke graph 并判定结果。
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    expectations = json.loads(_EXPECTATIONS.read_text(encoding="utf-8"))
    cases: list[BenchCase] = []

    for problem_path in sorted(_PROBLEMS_DIR.glob("*.json")):
        case_id = problem_path.stem
        expect = expectations[case_id]
        problem = json.loads(problem_path.read_text(encoding="utf-8"))
        case_out = out_dir / case_id
        case_out.mkdir(parents=True, exist_ok=True)

        graph = build_graph()  # bench 不带 checkpointer / interrupt
        final = graph.invoke({
            "problem": problem["title"] + "\n" + "\n".join(problem["questions"]),
            "background": problem.get("background", ""),
            "questions": problem["questions"],
            "stage_target": "basic", "iteration": 0,
            "output_dir": str(case_out),
            "human_decision": HumanDecision(approved=True).model_dump(),
        })
        cases.append(_evaluate(case_id, final, expect))

    report = BenchReport(cases=cases)
    (out_dir / "bench_report.json").write_text(
        json.dumps({"cases": [asdict(c) for c in cases]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
```

- [ ] **Step 4：跑测试**

Run: `pytest tests/bench/test_runner_mock.py -v`
Expected: 5 passed.

- [ ] **Step 5：提交**

```bash
git add src/math_agent/bench/runner.py tests/bench/conftest.py tests/bench/test_runner_mock.py
git commit -m "feat(bench): pure live runner in src; mock harness in tests"
```

---

## Phase 4：链路追踪与运行报告

### Task 4.1：tracing 模块（先写测试）

**Files:**
- Create: `tests/test_tracing.py`
- Create: `src/math_agent/tracing.py`

- [ ] **Step 1：写失败的测试**

```python
import json
from pathlib import Path
from math_agent.tracing import Tracer


def test_tracer_logs_llm_event(workdir):
    t = Tracer(thread_id="t1", out_dir=workdir)
    t.log_llm(model="gpt-4o", prompt_tokens=100, completion_tokens=50, latency_ms=1234)
    t.flush()
    rep = json.loads((workdir / "trace.json").read_text(encoding="utf-8"))
    assert rep["thread_id"] == "t1"
    assert rep["llm_calls"] == 1
    assert rep["tokens"]["prompt"] == 100
    assert rep["tokens"]["completion"] == 50


def test_tracer_aggregates_per_model(workdir):
    t = Tracer(thread_id="t1", out_dir=workdir)
    t.log_llm(model="A", prompt_tokens=10, completion_tokens=5, latency_ms=100)
    t.log_llm(model="A", prompt_tokens=20, completion_tokens=10, latency_ms=200)
    t.log_llm(model="B", prompt_tokens=5, completion_tokens=2, latency_ms=50)
    t.flush()
    rep = json.loads((workdir / "trace.json").read_text(encoding="utf-8"))
    assert rep["per_model"]["A"]["calls"] == 2
    assert rep["per_model"]["A"]["prompt_tokens"] == 30
    assert rep["per_model"]["B"]["calls"] == 1


def test_tracer_logs_node_phase(workdir):
    t = Tracer(thread_id="t1", out_dir=workdir)
    with t.node("analyst"):
        pass
    t.flush()
    rep = json.loads((workdir / "trace.json").read_text(encoding="utf-8"))
    assert rep["nodes"][0]["name"] == "analyst"
    assert rep["nodes"][0]["duration_ms"] >= 0
```

- [ ] **Step 2：跑测试，确认失败**

Run: `pytest tests/test_tracing.py -v`
Expected: ImportError.

- [ ] **Step 3：实现 `tracing.py`**

```python
"""轻量 tracing：把 LLM 调用与节点执行写入一份 trace.json。

设计：
- 同步、单线程；不引入 OTel/LangSmith（它们留作可选 callback）。
- Tracer 暴露 log_llm() 与 node() 上下文管理器；调用方负责打点。
- 在 llm.complete 内统一调 Tracer（通过 contextvars，避免显式传参）。
"""
from __future__ import annotations

import contextvars
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


_current: contextvars.ContextVar["Tracer | None"] = contextvars.ContextVar(
    "math_agent_tracer", default=None,
)


@dataclass
class _NodeRecord:
    name: str
    start_ms: int
    duration_ms: int = 0


@dataclass
class Tracer:
    thread_id: str
    out_dir: Path

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    per_model: dict = field(default_factory=dict)
    nodes: list[_NodeRecord] = field(default_factory=list)

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 公共 API ----

    def log_llm(self, *, model: str, prompt_tokens: int, completion_tokens: int,
                latency_ms: int) -> None:
        self.llm_calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        m = self.per_model.setdefault(model, {
            "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0,
        })
        m["calls"] += 1
        m["prompt_tokens"] += prompt_tokens
        m["completion_tokens"] += completion_tokens
        m["latency_ms"] += latency_ms

    @contextmanager
    def node(self, name: str):
        # 用 monotonic_ns 测 duration，避免 NTP 校时回拨导致负值。
        start_ns = time.monotonic_ns()
        rec = _NodeRecord(name=name, start_ms=start_ns // 1_000_000)
        try:
            yield
        finally:
            rec.duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000
            self.nodes.append(rec)

    def flush(self) -> Path:
        out = self.out_dir / "trace.json"
        out.write_text(json.dumps({
            "thread_id": self.thread_id,
            "llm_calls": self.llm_calls,
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
            },
            "per_model": self.per_model,
            "nodes": [{"name": r.name, "duration_ms": r.duration_ms}
                      for r in self.nodes],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return out


# ---- 全局当前 Tracer 句柄 ----

def set_current(tracer: Tracer | None) -> contextvars.Token:
    return _current.set(tracer)


def get_current() -> Tracer | None:
    return _current.get()


def reset_current(token: contextvars.Token) -> None:
    _current.reset(token)
```

- [ ] **Step 4：跑测试**

Run: `pytest tests/test_tracing.py -v`
Expected: 3 passed.

- [ ] **Step 5：提交**

```bash
git add src/math_agent/tracing.py tests/test_tracing.py
git commit -m "feat: lightweight tracer for llm/node events"
```

---

### Task 4.2：把 Tracer 接进 `llm.complete` 与 graph

**Files:**
- Modify: `src/math_agent/llm.py`
- Modify: `src/math_agent/graph.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1：在 `tests/test_llm.py` 追加测试**

```python
def test_complete_logs_to_current_tracer(mocker, tmp_path):
    from math_agent.tracing import Tracer, set_current, reset_current
    fake = mocker.MagicMock()
    fake.choices = [mocker.MagicMock(message=mocker.MagicMock(content="hi"))]
    fake.usage = mocker.MagicMock(prompt_tokens=10, completion_tokens=5)
    mocker.patch("litellm.completion", return_value=fake)

    t = Tracer(thread_id="t", out_dir=tmp_path)
    tok = set_current(t)
    try:
        llm.complete("ping", model="gpt-4o-mini")
    finally:
        reset_current(tok)
    assert t.llm_calls == 1
    assert t.prompt_tokens == 10 and t.completion_tokens == 5
```

- [ ] **Step 2：在 `llm.py` 中 import 并打点**

```python
import time
from math_agent.tracing import get_current as _get_tracer
```

在 `_do_completion` 中：

```python
def _do_completion(**kw):
    t0_ns = time.monotonic_ns()
    try:
        resp = litellm.completion(**kw)
    except Exception as e:
        raise classify_exception(e)
    tracer = _get_tracer()
    if tracer is not None:
        usage = getattr(resp, "usage", None)
        tracer.log_llm(
            model=kw.get("model", "?"),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=(time.monotonic_ns() - t0_ns) // 1_000_000,
        )
    return resp
```

- [ ] **Step 3：在 `graph.py` 的 `_wrap` 中加 node 计时**

把 `_wrap` 改为：

```python
def _wrap(fn):
    def _inner(s: dict) -> dict:
        from math_agent.tracing import get_current
        name = fn.__name__.replace("_node", "")
        tracer = get_current()
        if tracer is not None:
            with tracer.node(name):
                return fn(MathModelingState(**s))
        return fn(MathModelingState(**s))
    return _inner
```

- [ ] **Step 4：跑测试**

Run: `pytest tests/test_llm.py tests/test_tracing.py -v`
Expected: 全绿。

- [ ] **Step 5：提交**

```bash
git add src/math_agent/llm.py src/math_agent/graph.py tests/test_llm.py
git commit -m "feat(tracing): instrument llm.complete and graph nodes"
```

---

### Task 4.3：CLI `report` 子命令与 `run`/`resume` 内创建 Tracer

**Files:**
- Modify: `src/math_agent/cli.py`

- [ ] **Step 1：在 `cli.py` 顶部 import**

```python
from math_agent.tracing import Tracer, set_current, reset_current
from rich.console import Console
from rich.table import Table
```

并在 `pyproject.toml` dependencies 中加 `"rich>=13.7,<14"`，`pip install -e ".[dev]"`。

- [ ] **Step 2：在 `run` 命令开头创建 Tracer，在结束时 flush**

把 `run` 函数体改为：

```python
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
        "stage_target": "basic", "iteration": 0,
        "output_dir": str(out),
    }
    tracer = Tracer(thread_id=thread, out_dir=out)
    tok = set_current(tracer)
    try:
        g.invoke(initial, config=_config(thread))
    finally:
        tracer.flush()
        reset_current(tok)
    typer.echo(f"paused at human_review (thread={thread}); trace at {out / 'trace.json'}")
```

同样把 `resume` 包在 Tracer 里。

- [ ] **Step 3：新增 `report` 子命令**

```python
@app.command()
def report(out: Path = typer.Option(Path("runs/latest"))):
    """打印一次运行的 trace 报告 + evaluation 摘要。"""
    trace_path = out / "trace.json"
    if not trace_path.exists():
        typer.echo(f"no trace at {trace_path}"); raise typer.Exit(1)
    blob = json.loads(trace_path.read_text(encoding="utf-8"))

    c = Console()
    t = Table(title=f"Run report ({blob['thread_id']})")
    t.add_column("metric"); t.add_column("value")
    t.add_row("LLM calls", str(blob["llm_calls"]))
    t.add_row("Prompt tokens", str(blob["tokens"]["prompt"]))
    t.add_row("Completion tokens", str(blob["tokens"]["completion"]))
    c.print(t)

    tm = Table(title="Per model")
    tm.add_column("model"); tm.add_column("calls"); tm.add_column("prompt"); tm.add_column("completion")
    for m, d in blob.get("per_model", {}).items():
        tm.add_row(m, str(d["calls"]), str(d["prompt_tokens"]), str(d["completion_tokens"]))
    c.print(tm)

    tn = Table(title="Nodes")
    tn.add_column("node"); tn.add_column("duration_ms")
    for n in blob.get("nodes", []):
        tn.add_row(n["name"], str(n["duration_ms"]))
    c.print(tn)
```

- [ ] **Step 4：新增 `ingest` 与 `bench` 子命令**

```python
@app.command()
def ingest(
    src: Path = typer.Option(..., exists=True, readable=True),
    db: Path = typer.Option(Path("runs/rag.sqlite")),
    embedding_model: str = typer.Option("text-embedding-3-small"),
    dim: int = typer.Option(1536),
):
    from math_agent.rag.ingest import ingest_directory
    rep = ingest_directory(src_dir=src, db_path=db, embedding_model=embedding_model,
                           dim=dim)
    typer.echo(f"files={rep.files_processed} chunks={rep.chunks_added} skipped={len(rep.skipped)}")


@app.command()
def bench(out: Path = typer.Option(Path("runs/bench"))):
    """真跑历年题回归基准（live 模式，需要真 LLM API key）。

    要跑 mock 模式（结构性校验，不消耗 API），用：pytest tests/bench/
    """
    from math_agent.bench.runner import run_bench
    rep = run_bench(out_dir=out)
    for c in rep.cases:
        flag = "PASS" if c.passed else "FAIL"
        typer.echo(f"[{flag}] {c.problem_id} overall={c.overall} failures={c.failures}")
```

- [ ] **Step 5：跑全测**

Run: `pytest -q`
Expected: 全绿。

- [ ] **Step 6：手测（可选，依赖真 API key）**

```bash
math-agent bench
math-agent report --out runs/bench/2022_A
```

- [ ] **Step 7：提交**

```bash
git add src/math_agent/cli.py pyproject.toml
git commit -m "feat(cli): tracer + report + ingest + bench subcommands"
```

---

## Phase 5：可选 LangSmith / OTel callback + 运行手册

### Task 5.1：可选 callback（仅当环境变量存在时启用）

**Files:**
- Modify: `src/math_agent/llm.py`

- [ ] **Step 1：在 `complete()` 上方加一段初始化**

```python
import os

_LITELLM_CALLBACKS_CONFIGURED = False


def _resolve_callback_names() -> list[str]:
    """纯函数：根据环境变量返回 callback 名称列表。便于测试。"""
    cbs: list[str] = []
    if os.getenv("LANGSMITH_API_KEY"):
        cbs.append("langsmith")
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        cbs.append("otel")
    return cbs


def _configure_callbacks_once():
    global _LITELLM_CALLBACKS_CONFIGURED
    if _LITELLM_CALLBACKS_CONFIGURED:
        return
    cbs = _resolve_callback_names()
    if cbs:
        litellm.success_callback = cbs
        litellm.failure_callback = cbs
    _LITELLM_CALLBACKS_CONFIGURED = True
```

并在 `complete()` 顶部调用 `_configure_callbacks_once()`。

- [ ] **Step 2：写测试，断言纯函数 `_resolve_callback_names` 在缺环境变量时返回空列表**

在 `tests/test_llm.py` 末尾：

```python
def test_resolve_callback_names_empty_without_env(monkeypatch):
    """直接测纯函数，避免依赖 litellm 模块的全局状态（其他测试可能污染）。"""
    from math_agent.llm import _resolve_callback_names
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert _resolve_callback_names() == []


def test_resolve_callback_names_includes_langsmith_when_env_set(monkeypatch):
    from math_agent.llm import _resolve_callback_names
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert _resolve_callback_names() == ["langsmith"]
```

- [ ] **Step 3：跑测试**

Run: `pytest tests/test_llm.py -v`
Expected: 全绿。

- [ ] **Step 4：提交**

```bash
git add src/math_agent/llm.py tests/test_llm.py
git commit -m "feat: opt-in langsmith/otel callbacks via env vars"
```

---

### Task 5.2：运行手册

**Files:**
- Create: `docs/plan-c-runbook.md`

- [ ] **Step 1：写手册**

```markdown
# Plan C Runbook

## 1. RAG 索引

```bash
# 把历年国一论文 PDF/MD 放进 corpus/
math-agent ingest --src corpus --db runs/rag.sqlite \
  --embedding-model text-embedding-3-small --dim 1536

# 在运行时开启 RAG 注入
export MATH_AGENT_RAG_ENABLED=1
export MATH_AGENT_RAG_DB=runs/rag.sqlite
math-agent run --problem tests/fixtures/sample_problem.json --out runs/demo
```

## 2. 回归基准

```bash
# mock 模式：不调 LLM，校验流水线结构 + 关键词覆盖（CI / 本地快速检查）
pytest tests/bench/ -v
# 期望：每条 PASS

# live 模式（贵；建议手动逐题跑）—— 真调 LLM
math-agent bench --out runs/bench
# 或针对单题 debug
math-agent run --problem src/math_agent/bench/problems/2022_A.json --out runs/live/2022_A
```

## 3. 链路追踪

```bash
math-agent report --out runs/demo
# 显示每模型调用数 / token 数 / 每节点耗时
```

可选远端追踪：
- 设置 `LANGSMITH_API_KEY` 自动启用 LangSmith callback
- 设置 `OTEL_EXPORTER_OTLP_ENDPOINT` 自动启用 OTel callback

## 4. 错误处理速查

| 现象 | error_kind / 类型 | 是否自动重试 | 排查 |
|---|---|---|---|
| LLM 429 | LLMRateLimitError | 是（指数退避） | 默认 4 次，可调 `max_retries` |
| LLM JSON 解析失败 | LLMValidationError | 否（complete 内部喂回错误重试） | 看 schema 是否过严 |
| LLM 超时/连接错误 | LLMTransportError | 是 | 检查 `LITELLM_LOG=DEBUG` 输出 |
| runner 超时 | `RunResult(error_kind="timeout")` | Coder 内部重试 1 次 | 加大 `timeout` 或简化代码 |
| runner 运行错误 | `RunResult(error_kind="runtime")` | Coder 内部重试 1 次 | 看 stderr |
| xelatex 缺失 | `LatexResult(error_kind="missing_binary")` | 否 | 安装 TeX Live 或回退 Markdown |
| xelatex 编译失败 | `LatexResult(error_kind="compile")` | 否 | 看 `paper.log`；模板字体可能缺失 |
```

- [ ] **Step 2：提交**

```bash
git add docs/plan-c-runbook.md
git commit -m "docs: plan C runbook (rag/bench/tracing/errors)"
```

---

### Task 5.3：更新主 README

**Files:**
- Modify: `README.md`

- [ ] **Step 1：在 README 末尾追加**

```markdown
## 已完成（Plan C）

- 统一错误类型（`errors.py`）+ tenacity 重试装饰
- `runner.py` / `latex_compile.py` 输出结构化 `error_kind`
- **RAG**：md/txt/pdf → 切块 → litellm embedding → sqlite-vec；Analyst/Modeler/Writer prompt 自动注入
- **bench**：mock 模式回归两道历年题，写出 JSON 报告
- **tracing**：LLM 调用数/token/节点耗时落到 `trace.json`，`math-agent report` 可视化
- 可选 LangSmith / OTel callback（仅当对应环境变量存在时启用）

完整运行手册：`docs/plan-c-runbook.md`
```

- [ ] **Step 2：提交**

```bash
git add README.md
git commit -m "docs: README references plan C deliverables"
```

---

## 自我审查

**0. 与当前 main 的基准对齐（2026-06-30 校对）**

本计划写于 Plan B 早期。校对发现 Plan B 后期累计了多个 hotfix / 模板改动，与本 plan 范围 **正交不冲突**：

- `tools/runner.py`（原名 `sandbox.py`，重命名 hotfix）— 已在本计划全文修订为 `runner.py`
- `config.MAX_LLM_RETRIES` / `MAX_CODE_RETRIES` / `MAX_WRITER_ITERATIONS` 已存在 — Task 1.3 retry 装饰器改为**从 config 读 max_attempts**，不硬编码
- `paper.keywords` 字段、writer↔paper_critic 闭环、paper_critic 注入 stdout — 三者都是 writer/paper_critic 节点行为升级，本 plan 不动它们
- `latex_node` 累积大量 markdown→tex helper、gmcm 模板、tabularx booktabs、`_wrap_unicode_math` / `_md_table_to_latex` / equation promotion 等 — 本 plan 完全不动 `latex_node`，仅 Phase 1 给 `latex_compile.py` 增加 `error_kind` 字段，与上层 helper 链正交
- writer prompt 已有 IRON RULE 1-6（含 LaTeX 兼容）+ critic feedback 注入；Task 2.6 注入 RAG context 时**限制 800 字符**避免 token 预算超限（见该任务说明）

预计 RAG 接通后单题 token 成本增加 30-50%；live bench 单题仍 15-25 min。

**1. Spec 覆盖**

| 方案要点（Plan A/B 之外） | 覆盖位置 |
|---|---|
| 鲁棒性与错误处理机制（代码失败、编译失败、格式错误） | Phase 1：`errors.py` + `retry.py` + sandbox/latex `error_kind` ✓ |
| LLM 重试 / 限流 | Phase 1：`llm_retry` 装饰；rate limit/transport 分类 ✓ |
| RAG 知识库（历年方法 + 写作模板） | Phase 2：chunking/embedding/store/retrieve/ingest 全链 + 三个 prompt 注入 ✓ |
| 评分基准回归集 | Phase 3：`bench/` mock 模式 ✓ |
| LangSmith / OpenTelemetry 链路追踪 | Phase 5.1：opt-in callback ✓ |
| 运行报告（cost/latency） | Phase 4：`tracing.py` + `math-agent report` ✓ |
| 独立打分与基准对比 | Phase 3 复用 Plan B 的 `EvaluationReport`，与 expectations.json 对比 ✓ |
| 短期：RAG + 鲁棒性 + Evaluation Module | Evaluation 在 Plan B 已交付；本计划补齐 RAG + 鲁棒性 ✓ |
| 中期：多题型支持 | bench/problems/ 加新文件即可（非代码改动）✓ |
| 长期：美赛支持 + 平台化部署 | 仍在路线图，超出本计划范围（显式标注）✓ |

**2. Placeholder 扫描**

- Phase 3 Task 3.2 已经把 bench mock harness 从 `src/math_agent/bench/` 挪到
  `tests/bench/conftest.py`，runner.py 只剩纯 live 实现；CLI `bench` 子命令也只跑 live。
  mock 模式从 pytest 入口跑（`pytest tests/bench/`）。
- Task 2.6 修改 modeler/writer 节点时给出了"查询字符串构造方式"（`latest_model().description + " " + state.problem` / `state.problem + state.paper.model_section[:500]`）—— 是具体的、非 TBD 的指令。
- Task 5.2 的运行手册中"live 模式（贵；建议手动逐题跑）"是说明性指引而非占位。
- 其余无 TODO / "实现 X 即可" / "类似 Task N" 等占位。

**3. 类型一致性**

- `Chunk(text, source, index)` 在 chunking → embeddings → store → retrieve 全链中字段一致 ✓
- `StoredChunk(id, text, source, index, score)` 在 store / retrieve 一致 ✓
- `Snippet(text, source, score)` 在 retrieve / prompts 注入一致；`format_snippets` 仅消费这三个字段 ✓
- `IngestReport(files_processed, chunks_added, skipped)` 在 ingest / CLI 输出一致 ✓
- `BenchCase(problem_id, overall, passed, failures)` 与 `BenchReport(mode, cases)` 在 runner / 测试 / CLI 输出一致 ✓
- `Tracer.log_llm(model, prompt_tokens, completion_tokens, latency_ms)` 在 tracing 内部 / llm.py 调用处签名一致 ✓
- `RunResult.error_kind` ∈ {"", "timeout", "runtime"}；`LatexResult.error_kind` ∈ {"", "missing_binary", "compile", "timeout"} —— 两组字典都在 Phase 1.4 中给出确定取值，与运行手册速查表一致 ✓
- `llm.complete` 新增的私有参数 `_retry_attempts` / `_retry_base_delay` 在 Task 1.3 / Task 1.4 测试中使用一致 ✓
- RAG 配置 `RAG_ENABLED` / `RAG_DB_PATH` / `RAG_EMBEDDING_MODEL` / `RAG_EMBEDDING_DIM` / `RAG_TOPK` 在 config.py 与三个节点中名称完全一致 ✓

无类型/命名漂移。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-28-math-agent-plan-c.md`. Two execution options:

**1. Subagent-Driven (recommended)** — 每个 Task 派一个全新 subagent 执行，我在任务间审查，迭代快，状态干净。

**2. Inline Execution** — 我在当前会话里按 executing-plans 流程逐 Task 执行，带 checkpoint 让你审查。

哪种方式？或者你想先**调整计划**（例如：把 sqlite-vec 换成 chroma/qdrant、bench 直接跑 live 模式、tracing 默认接 LangSmith 而非自建 JSON、Coder 节点也加 runner_retry 等），告诉我即可。
