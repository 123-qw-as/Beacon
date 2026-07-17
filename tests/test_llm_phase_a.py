from __future__ import annotations

import time

import pytest

from math_agent.errors import (
    LLMError, LLMTimeoutError, LLMRateLimitError, LLMConnectionError,
    LLMServerError, LLMBudgetExceededError, LLMValidationError,
    LLMAuthenticationError, LLMInvalidRequestError,
)
from math_agent.transport import (
    FakeCompletionTransport, FakeClock, CompletionRequest,
)
import math_agent.llm as llm_mod


@pytest.fixture
def fake_clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(llm_mod, "_get_clock", lambda: c)
    return c


@pytest.fixture
def fake_transport(fake_clock, monkeypatch):
    """doc"""
    t = FakeCompletionTransport(clock=fake_clock)
    monkeypatch.setattr(llm_mod, "_get_transport", lambda: t)
    return t
    assert fake_transport.send_count <= 4



# ---- 搂10.1.4: per-profile attempt timeout 鐙珛鐢熸晥 ----

def test_per_profile_attempt_timeout_is_applied(fake_transport, fake_clock, monkeypatch):
    """doc"""
    monkeypatch.setattr(llm_mod, "_PROFILE_ATTEMPT_TIMEOUT",
                        {"standard": 50.0, "long": 80.0, "vision": 50.0})

    fake_transport.enqueue_ok("ok")
    llm_mod.complete("hi", model="gpt-test", profile="standard")

    assert len(fake_transport.calls) == 1
    std_timeout = fake_transport.calls[0].timeout_s
    assert std_timeout == 50.0, f"standard 鐢诲儚 timeout 搴斾负 50锛屽疄闄?{std_timeout}"

    fake_transport.enqueue_ok("ok")
    llm_mod.complete("hi", model="gpt-test", profile="long")

    long_timeout = fake_transport.calls[1].timeout_s
    assert long_timeout == 80.0, f"long 鐢诲儚 timeout 搴斾负 80锛屽疄闄?{long_timeout}"


# ---- 搂10.1.5: JSON 淇鍏变韩鎬?deadline ----

def test_validation_repairs_share_total_budget(fake_transport, fake_clock, monkeypatch):
    """doc"""
    from pydantic import BaseModel

    class _Ans(BaseModel):
        x: int

    # 姣忔閮借繑鍥為潪娉?JSON锛屼笖姣忔娑堣€?0.3s锛坱otal=1.0 -> 绾?3 娆″悗棰勭畻鑰楀敖锛?    for _ in range(10):
        fake_transport.enqueue_ok("not json at all", delay_s=0.3)
    monkeypatch.setattr(llm_mod, "_PROFILE_ATTEMPT_TIMEOUT",
                        {"standard": 100.0, "long": 100.0, "vision": 100.0})
    monkeypatch.setattr(llm_mod, "_PROFILE_TOTAL_TIMEOUT",
                        {"standard": 1.0, "long": 1.0, "vision": 1.0})
    monkeypatch.setattr(llm_mod, "_DEFAULT_VALIDATION_REPAIRS", 5)

    with pytest.raises(LLMError):
        llm_mod.complete("hi", schema=_Ans, model="gpt-test")
    assert fake_transport.send_count <= 5


# ---- 搂10.1.6: 澶辫触 attempt 鍐欏叆 trace ----

def test_failed_attempt_is_written_to_trace(fake_transport, fake_clock, monkeypatch, tmp_path):
    """doc"""
    from math_agent.tracing import Tracer, set_current, reset_current

    fake_transport.enqueue_hang()
    monkeypatch.setattr(llm_mod, "_DEFAULT_ATTEMPT_TIMEOUT", 0.2)
    monkeypatch.setattr(llm_mod, "_DEFAULT_TOTAL_TIMEOUT", 0.3)

    t = Tracer(thread_id="t", out_dir=tmp_path)
    tok = set_current(t)
    try:
        with pytest.raises(LLMTimeoutError):
            llm_mod.complete("hi", model="gpt-test")
    finally:
        reset_current(tok)

    assert t.llm_attempts == 1, f"搴旇褰?1 娆＄墿鐞嗗皾璇曪紝瀹為檯 {t.llm_attempts}"
    assert t.llm_failures == 1, f"搴旇褰?1 娆″け璐ワ紝瀹為檯 {t.llm_failures}"
    assert t.llm_timeouts == 1, f"搴旇褰?1 娆¤秴鏃讹紝瀹為檯 {t.llm_timeouts}"
    assert len(t.attempt_records) == 1
    rec = t.attempt_records[0]
    assert rec["status"] == "timeout"


# ---- 搂10.1.7: worker timeout 鍚庨噸鍚?----

def test_worker_timeout_terminates_and_restarts_worker(fake_transport, fake_clock, monkeypatch):
    """doc"""
    fake_transport.enqueue_hang()
    fake_transport.enqueue_ok("recovered")
    monkeypatch.setattr(llm_mod, "_DEFAULT_ATTEMPT_TIMEOUT", 0.2)
    monkeypatch.setattr(llm_mod, "_DEFAULT_TOTAL_TIMEOUT", 0.3)

    with pytest.raises(LLMTimeoutError):
        llm_mod.complete("first", model="gpt-test")

    out = llm_mod.complete("second", model="gpt-test")
    assert out == "recovered"


# ---- 搂10.1.8: tracer 鐣欏湪鐖惰繘绋?----

def test_tracer_stays_in_parent_process(fake_transport, fake_clock, monkeypatch, tmp_path):
    """doc"""
    from math_agent.tracing import Tracer, set_current, reset_current

    fake_transport.enqueue_ok("hi", prompt_tokens=10, completion_tokens=5)

    t = Tracer(thread_id="t", out_dir=tmp_path)
    tok = set_current(t)
    try:
        out = llm_mod.complete("hi", model="gpt-test")
    finally:
        reset_current(tok)

    assert out == "hi"
    assert t.llm_calls == 1
    assert t.llm_attempts == 1
    assert t.prompt_tokens == 10
    assert t.completion_tokens == 5
def test_validation_failure_surfaces_as_llm_validation_error(fake_transport, fake_clock, monkeypatch):
    from pydantic import BaseModel
    class _Ans(BaseModel):
        x: int
    fake_transport.enqueue_ok("not json at all", delay_s=0.1)
    fake_transport.enqueue_ok("still not json", delay_s=0.1)
    fake_transport.enqueue_ok("again not json", delay_s=0.1)
    monkeypatch.setattr(llm_mod, "_PROFILE_ATTEMPT_TIMEOUT",
                        {"standard": 100.0, "long": 100.0, "vision": 100.0})
    monkeypatch.setattr(llm_mod, "_PROFILE_TOTAL_TIMEOUT",
                        {"standard": 2.0, "long": 2.0, "vision": 2.0})
    monkeypatch.setattr(llm_mod, "_DEFAULT_VALIDATION_REPAIRS", 2)
    with pytest.raises(LLMValidationError):
        llm_mod.complete("hi", schema=_Ans, model="gpt-test")
