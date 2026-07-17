"""§10.4 真实 router 冒烟测试。

默认跳过；通过显式环境变量启用：
  MATH_AGENT_RUN_LIVE_LLM_TESTS=1

仅发送极短 prompt，验证普通成功、时限参数和 trace，不测试真实 300 秒 timeout。
"""
from __future__ import annotations

import os

import pytest

from math_agent.transport import LiteLLMWorkerTransport, CompletionRequest
from math_agent.errors import LLMError

_LIVE = os.getenv("MATH_AGENT_RUN_LIVE_LLM_TESTS") == "1"

pytestmark = pytest.mark.skipif(not _LIVE, reason="需要 MATH_AGENT_RUN_LIVE_LLM_TESTS=1 启用真实调用")


def _live_model() -> str:
    """从 .env 读取默认模型（格式与 config.DEFAULT_MODEL 一致）。"""
    from math_agent.config import DEFAULT_MODEL
    # litellm 需要 provider 前缀
    model = DEFAULT_MODEL
    if "/" not in model:
        model = f"openai/{model}"
    return model


def test_live_basic_completion():
    """真实 router：极短 prompt 应成功返回文本。"""
    t = LiteLLMWorkerTransport(kill_grace=2)
    req = CompletionRequest(
        model=_live_model(),
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        timeout_s=60.0,
    )
    resp = t.send(req, timeout_s=60.0)
    assert resp.content, f"空响应：{resp}"
    assert resp.prompt_tokens > 0, f"prompt_tokens 应 >0：{resp}"
    t.close()


def test_live_structured_completion():
    """真实 router：候选模型必须能返回 Beacon 所需的结构化 JSON。"""
    from pydantic import BaseModel
    from math_agent.llm import complete

    class _Probe(BaseModel):
        ok: bool
        answer: int

    result = complete(
        '只输出 JSON：计算 6*7，格式 {"ok":true,"answer":42}',
        schema=_Probe,
        model=_live_model(),
        max_retries=0,
        max_validation_repairs=0,
        fallback_models=[],
    )
    assert result.ok is True
    assert result.answer == 42


def test_live_timeout_parameter_respected():
    """真实 router：极短 timeout 应触发 LLMTimeoutError（不测试 300s，用 0.01s）。"""
    t = LiteLLMWorkerTransport(kill_grace=2)
    req = CompletionRequest(
        model=_live_model(),
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        timeout_s=0.01,  # 极短，必然超时
    )
    from math_agent.errors import LLMTimeoutError
    with pytest.raises(LLMTimeoutError):
        t.send(req, timeout_s=0.01)
    assert t._worker is None, "超时后 worker 应被回收"
    t.close()


def test_live_trace_records_attempt():
    """真实 router：成功调用后 trace 记录 attempt。"""
    from math_agent.tracing import Tracer, set_current, reset_current
    import tempfile
    from pathlib import Path

    t = LiteLLMWorkerTransport(kill_grace=2)
    req = CompletionRequest(
        model=_live_model(),
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        timeout_s=60.0,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tracer = Tracer(thread_id="live", out_dir=Path(tmp))
        tok = set_current(tracer)
        # 手动打点（complete() 路径的简化版）
        tracer.begin_attempt(model=req.model, profile="standard", attempt=1)
        resp = t.send(req, timeout_s=60.0)
        tracer.end_attempt(model=req.model, status="success", latency_ms=0)
        tracer.log_llm(model=req.model, prompt_tokens=resp.prompt_tokens,
                       completion_tokens=resp.completion_tokens, latency_ms=0)
        reset_current(tok)

    assert tracer.llm_attempts == 1
    assert tracer.llm_calls == 1
    assert tracer.prompt_tokens > 0
    t.close()
