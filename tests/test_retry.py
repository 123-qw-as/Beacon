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


def test_llm_retry_attempts_can_be_overridden_by_env(monkeypatch):
    """env override takes precedence over default (currently 8 due to .env)."""
    from math_agent.retry import _default_llm_attempts
    monkeypatch.setenv("MATH_AGENT_LLM_RETRY_ATTEMPTS", "3")
    assert _default_llm_attempts() == 3

def test_llm_retry_base_delay_can_be_overridden_by_env(monkeypatch):
    from math_agent.retry import _default_llm_base_delay
    monkeypatch.setenv("MATH_AGENT_LLM_RETRY_BASE_DELAY", "0.5")
    assert _default_llm_base_delay() == 0.5

def test_llm_retry_defaults_to_2s_base_delay(monkeypatch):
    from math_agent.retry import _default_llm_base_delay
    monkeypatch.delenv("MATH_AGENT_LLM_RETRY_BASE_DELAY", raising=False)
    assert _default_llm_base_delay() == 2.0
    monkeypatch.setenv("MATH_AGENT_LLM_RETRY_BASE_DELAY", "0.5")
    assert _default_llm_base_delay() == 0.5


