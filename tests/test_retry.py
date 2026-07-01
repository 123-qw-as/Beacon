import pytest
from math_agent.errors import (
    LLMRateLimitError, LLMTransportError, LLMValidationError,
    RunnerError, RunnerTimeoutError, LatexMissingBinaryError,
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
    assert len(calls) == 1


def test_llm_retry_gives_up_after_max():
    calls = []

    @llm_retry(max_attempts=2, base_delay=0)
    def f():
        calls.append(1)
        raise LLMTransportError("net")

    with pytest.raises(LLMTransportError):
        f()
    assert len(calls) == 2


def test_runner_retry_retries_runner_error_then_succeeds():
    calls = []

    @runner_retry(max_attempts=3, base_delay=0)
    def f():
        calls.append(1)
        if len(calls) < 2:
            raise RunnerTimeoutError("timeout")
        return "ok"

    assert f() == "ok"
    assert len(calls) == 2


def test_runner_retry_does_not_retry_missing_binary():
    calls = []

    @runner_retry(max_attempts=3, base_delay=0)
    def f():
        calls.append(1)
        raise LatexMissingBinaryError("no xelatex")

    with pytest.raises(LatexMissingBinaryError):
        f()
    assert len(calls) == 1


def test_llm_retry_defaults_to_5_attempts_via_env(monkeypatch):
    """新默认 MAX_LLM_RETRIES + 3 = 5 次；可被 env override。"""
    from math_agent.retry import _default_llm_attempts
    # Plan A 的 MAX_LLM_RETRIES=2 -> 默认 5
    assert _default_llm_attempts() == 5
    monkeypatch.setenv("MATH_AGENT_LLM_RETRY_ATTEMPTS", "8")
    assert _default_llm_attempts() == 8


def test_llm_retry_defaults_to_2s_base_delay(monkeypatch):
    from math_agent.retry import _default_llm_base_delay
    assert _default_llm_base_delay() == 2.0
    monkeypatch.setenv("MATH_AGENT_LLM_RETRY_BASE_DELAY", "0.5")
    assert _default_llm_base_delay() == 0.5


