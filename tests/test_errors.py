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


def test_classify_unknown_passes_through_as_llmerror():
    e = RuntimeError("boom")
    out = classify_exception(e)
    assert isinstance(out, LLMError)
    assert "boom" in str(out)


def test_classify_keeps_existing_mathagent_error():
    inner = LLMValidationError("bad json")
    assert classify_exception(inner) is inner
