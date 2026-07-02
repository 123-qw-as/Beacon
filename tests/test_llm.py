import pytest
from pydantic import BaseModel
import math_agent.llm as llm
from math_agent.errors import LLMError


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
    assert isinstance(user_msg["content"], list)
    kinds = {p["type"] for p in user_msg["content"]}
    assert kinds == {"text", "image_url"}


def test_complete_classifies_and_retries_rate_limit(mocker):
    """litellm 抛 RateLimitError → 应被 classify 为 LLMRateLimitError → 触发重试。"""
    class _RL(Exception):
        pass
    _RL.__name__ = "RateLimitError"

    seq = [_RL("429"), mocker.MagicMock(
        choices=[mocker.MagicMock(message=mocker.MagicMock(content="ok"))]
    )]
    mocker.patch("litellm.completion", side_effect=seq)
    out = llm.complete("hi", model="gpt-4o-mini",
                      _retry_attempts=3, _retry_base_delay=0)
    assert out == "ok"


def test_complete_raises_llm_error_when_all_retries_exhausted(mocker):
    class _RL(Exception):
        pass
    _RL.__name__ = "RateLimitError"

    mocker.patch("litellm.completion", side_effect=_RL("429"))
    with pytest.raises(LLMError):
        llm.complete("hi", model="gpt-4o-mini", max_retries=0,
                     _retry_attempts=2, _retry_base_delay=0)


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


def test_resolve_callback_names_empty_without_env(monkeypatch):
    from math_agent.llm import _resolve_callback_names
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert _resolve_callback_names() == []


def test_resolve_callback_names_includes_langsmith_when_env_set(monkeypatch):
    from math_agent.llm import _resolve_callback_names
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert _resolve_callback_names() == ["langsmith"]


def test_complete_passes_timeout_to_litellm(mocker):
    """llm.complete 必须给 litellm.completion 传 timeout，防 httpx 无限阻塞。"""
    import httpx
    from math_agent.config import LLM_TIMEOUT

    captured = {}

    def _fake(*args, **kw):
        captured["kw"] = kw
        return mocker.MagicMock(
            choices=[mocker.MagicMock(message=mocker.MagicMock(content="ok"))]
        )

    mocker.patch("litellm.completion", side_effect=_fake)
    llm.complete("hi", model="gpt-4o-mini")
    assert captured["kw"].get("timeout") == LLM_TIMEOUT


def test_timeout_is_classified_as_transport_and_retried(mocker):
    """litellm 抛 httpx.TimeoutException → 应 classify 为 LLMTransportError → 走 tenacity 重试。"""
    import httpx

    timeout_err = httpx.TimeoutException("Read timed out")
    ok = mocker.MagicMock(
        choices=[mocker.MagicMock(message=mocker.MagicMock(content="ok"))]
    )
    mock_completion = mocker.patch(
        "litellm.completion", side_effect=[timeout_err, ok]
    )
    out = llm.complete(
        "hi", model="gpt-4o-mini",
        _retry_attempts=3, _retry_base_delay=0,
    )
    assert out == "ok"
    # 第一次超时 + 第二次成功 = 至少 2 次调用，证明 timeout 被纳入重试链
    assert mock_completion.call_count >= 2


def test_timeout_exhausts_retries_raises_transport_error(mocker):
    """连续超时超过重试预算 → 抛 LLMError（且底层是 LLMTransportError 语义）。"""
    import httpx
    from math_agent.errors import LLMError, LLMTransportError

    mocker.patch("litellm.completion", side_effect=httpx.TimeoutException("stalled"))
    with pytest.raises(LLMError):
        llm.complete(
            "hi", model="gpt-4o-mini", max_retries=0,
            _retry_attempts=2, _retry_base_delay=0,
        )
