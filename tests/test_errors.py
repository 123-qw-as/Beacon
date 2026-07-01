import pytest
from math_agent.errors import (
    MathAgentError, LLMError, LLMRateLimitError, LLMValidationError,
    LLMTransportError,
    RunnerError, RunnerTimeoutError, RunnerRuntimeError,
    LatexError, LatexMissingBinaryError, LatexCompileError,
    classify_exception,
)


def test_class_hierarchy():
    assert issubclass(LLMRateLimitError, LLMError)
    assert issubclass(LLMValidationError, LLMError)
    assert issubclass(LLMTransportError, LLMError)
    assert issubclass(LLMError, MathAgentError)
    assert issubclass(RunnerTimeoutError, RunnerError)
    assert issubclass(RunnerRuntimeError, RunnerError)
    assert issubclass(LatexMissingBinaryError, LatexError)
    assert issubclass(LatexCompileError, LatexError)


def test_classify_rate_limit_from_litellm():
    class FakeRateLimit(Exception):
        pass
    FakeRateLimit.__name__ = "RateLimitError"
    e = FakeRateLimit("rate limited")
    out = classify_exception(e)
    assert isinstance(out, LLMRateLimitError)


def test_classify_transport_error():
    class FakeTimeout(Exception):
        pass
    FakeTimeout.__name__ = "Timeout"
    e = FakeTimeout("read timeout after 30s")
    out = classify_exception(e)
    assert isinstance(out, LLMTransportError)


def test_classify_5xx_and_connection_error_as_transport():
    """本地 router 常见的 502/503/504 / Connection error / InternalServerError
    都应当被视为 transport（会被 llm_retry 重试）。"""
    cases = [
        RuntimeError("litellm.InternalServerError: OpenAIException - Connection error."),
        RuntimeError("Server error '502 Bad Gateway' for url 'http://x/embed'"),
        RuntimeError("HTTP 503 Service Unavailable"),
        RuntimeError("504 Gateway Timeout"),
    ]
    for e in cases:
        out = classify_exception(e)
        assert isinstance(out, LLMTransportError), f"{e} -> {type(out).__name__}"


def test_classify_unknown_passes_through_as_llmerror():
    e = RuntimeError("boom")
    out = classify_exception(e)
    assert isinstance(out, LLMError)
    assert "boom" in str(out)


def test_classify_keeps_existing_mathagent_error():
    inner = LLMValidationError("bad json")
    assert classify_exception(inner) is inner
