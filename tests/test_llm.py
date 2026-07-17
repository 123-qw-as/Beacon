"""LLM 接口行为测试（迁移到 transport seam）。

旧测试 mock litellm.completion，阶段 B 后 complete() 走 worker transport，
mock 不再生效。改用 FakeCompletionTransport 注入。
"""
import os
from pathlib import Path

import pytest
from pydantic import BaseModel

import math_agent.llm as llm_mod
from math_agent.errors import LLMError
from math_agent.transport import FakeCompletionTransport, FakeClock


class _Answer(BaseModel):
    summary: str
    score: int


@pytest.fixture
def fake_clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(llm_mod, "_get_clock", lambda: c)
    return c


@pytest.fixture
def fake_transport(monkeypatch, fake_clock):
    t = FakeCompletionTransport(clock=fake_clock)
    monkeypatch.setattr(llm_mod, "_get_transport", lambda: t)
    return t


def test_complete_returns_text_when_no_schema(fake_transport):
    fake_transport.enqueue_ok("hello")
    out = llm_mod.complete("say hi", model="gpt-4o-mini")
    assert out == "hello"


def test_code_profile_has_independent_bounded_deadline(monkeypatch):
    monkeypatch.setattr(llm_mod, "_PROFILE_ATTEMPT_TIMEOUT", {"code": 90.0})
    monkeypatch.setattr(llm_mod, "_PROFILE_TOTAL_TIMEOUT", {"code": 240.0})

    assert llm_mod._profile_timeout("code") == (90.0, 240.0)


def test_complete_returns_pydantic_when_schema(fake_transport):
    fake_transport.enqueue_ok('{"summary": "ok", "score": 9}')
    out = llm_mod.complete("rate it", schema=_Answer, model="gpt-4o-mini")
    assert isinstance(out, _Answer)
    assert out.score == 9


def test_complete_retries_on_invalid_json(fake_transport, monkeypatch):
    monkeypatch.setattr(llm_mod, "_DEFAULT_VALIDATION_REPAIRS", 2)
    fake_transport.enqueue_ok("not json")
    fake_transport.enqueue_ok('{"summary":"x","score":1}')
    out = llm_mod.complete("x", schema=_Answer, model="gpt-4o-mini")
    assert out.score == 1


def test_complete_raises_after_all_retries_exhausted(fake_transport, monkeypatch):
    monkeypatch.setattr(llm_mod, "_DEFAULT_VALIDATION_REPAIRS", 1)
    fake_transport.enqueue_ok("nope")
    fake_transport.enqueue_ok("still nope")
    with pytest.raises(llm_mod.LLMError):
        llm_mod.complete("x", schema=_Answer, model="gpt-4o-mini")


def test_complete_with_images_packs_multimodal_content(fake_transport):
    fake_transport.enqueue_ok("ok")
    out = llm_mod.complete(
        "describe this", model="gpt-4o-mini",
        images=["data:image/png;base64,AAA="],
    )
    assert out == "ok"
    user_msg = fake_transport.calls[0].messages[-1]
    assert isinstance(user_msg["content"], list)
    kinds = {p["type"] for p in user_msg["content"]}
    assert kinds == {"text", "image_url"}


def test_complete_classifies_and_retries_rate_limit(fake_transport, monkeypatch):
    from math_agent.errors import LLMRateLimitError
    fake_transport.enqueue_error(LLMRateLimitError("429"))
    fake_transport.enqueue_ok("ok")
    monkeypatch.setattr(llm_mod, "_PROFILE_ATTEMPT_TIMEOUT",
                        {"standard": 100.0, "long": 100.0, "vision": 100.0})
    monkeypatch.setattr(llm_mod, "_PROFILE_TOTAL_TIMEOUT",
                        {"standard": 100.0, "long": 100.0, "vision": 100.0})
    out = llm_mod.complete("hi", model="gpt-4o-mini")
    assert out == "ok"


def test_complete_raises_llm_error_when_all_retries_exhausted(fake_transport, monkeypatch):
    from math_agent.errors import LLMRateLimitError
    for _ in range(5):
        fake_transport.enqueue_error(LLMRateLimitError("429"))
    monkeypatch.setattr(llm_mod, "_PROFILE_ATTEMPT_TIMEOUT",
                        {"standard": 100.0, "long": 100.0, "vision": 100.0})
    monkeypatch.setattr(llm_mod, "_PROFILE_TOTAL_TIMEOUT",
                        {"standard": 100.0, "long": 100.0, "vision": 100.0})
    with pytest.raises(LLMError):
        llm_mod.complete("hi", model="gpt-4o-mini")


def test_server_error_fails_over_to_next_model_without_replaying_primary(
    fake_transport, monkeypatch,
):
    from math_agent.errors import LLMServerError

    fake_transport.enqueue_error(LLMServerError("502"))
    fake_transport.enqueue_ok("recovered")
    monkeypatch.setattr(
        llm_mod, "_PROFILE_ATTEMPT_TIMEOUT",
        {"standard": 100.0, "long": 100.0, "vision": 100.0},
    )
    monkeypatch.setattr(
        llm_mod, "_PROFILE_TOTAL_TIMEOUT",
        {"standard": 200.0, "long": 200.0, "vision": 200.0},
    )

    out = llm_mod.complete(
        "hi", model="primary", fallback_models=["backup"]
    )

    assert out == "recovered"
    assert [call.model for call in fake_transport.calls] == ["primary", "backup"]


def test_timeout_does_not_replay_request_on_fallback_model(fake_transport, monkeypatch):
    from math_agent.errors import LLMTimeoutError

    fake_transport.enqueue_error(LLMTimeoutError("timed out"))
    monkeypatch.setattr(
        llm_mod, "_PROFILE_ATTEMPT_TIMEOUT",
        {"standard": 90.0, "long": 90.0, "vision": 90.0},
    )
    monkeypatch.setattr(
        llm_mod, "_PROFILE_TOTAL_TIMEOUT",
        {"standard": 100.0, "long": 100.0, "vision": 100.0},
    )
    monkeypatch.setattr(
        llm_mod, "_PROFILE_FALLBACK_RESERVE",
        {"standard": 30.0, "long": 30.0, "vision": 30.0},
    )

    with pytest.raises(LLMTimeoutError):
        llm_mod.complete("hi", model="primary", fallback_models=["backup"])

    assert len(fake_transport.calls) == 1
    assert fake_transport.calls[0].model == "primary"
    assert fake_transport.calls[0].timeout_s == pytest.approx(90.0)


def test_timeout_cooldown_survives_worker_recovery_and_promotes_fallback(
    fake_transport, fake_clock, monkeypatch, tmp_path,
):
    """超时请求本次不重放；新 graph worker 从 run 文件恢复后切备用模型。"""
    from math_agent.errors import LLMTimeoutError
    from math_agent.tracing import Tracer, reset_current, set_current

    monkeypatch.setattr(llm_mod, "_MODEL_FAILURE_COOLDOWN", 30.0)
    monkeypatch.setattr(llm_mod, "_MODEL_MAX_COOLDOWN", 300.0)
    monkeypatch.setattr(llm_mod.time, "time", lambda: 1000.0)
    fake_transport.enqueue_error(LLMTimeoutError("timed out"))

    first = Tracer(thread_id="run", out_dir=tmp_path)
    token = set_current(first)
    try:
        with pytest.raises(LLMTimeoutError):
            llm_mod.complete("first", model="primary", fallback_models=["backup"])
    finally:
        reset_current(token)

    assert [call.model for call in fake_transport.calls] == ["primary"]
    health_path = tmp_path / "llm_model_health.json"
    assert health_path.is_file()

    # 模拟 supervisor 启动一个全新的 graph worker：进程内状态消失，run 文件保留。
    llm_mod._MODEL_UNHEALTHY_UNTIL.clear()
    llm_mod._MODEL_FAILURE_STREAK.clear()
    llm_mod._MODEL_SUCCESS_STREAK.clear()
    fake_transport.enqueue_ok("recovered")
    second = Tracer(thread_id="run", out_dir=tmp_path, append_existing=True)
    token = set_current(second)
    try:
        assert llm_mod.complete(
            "second", model="primary", fallback_models=["backup"]
        ) == "recovered"
    finally:
        reset_current(token)

    assert [call.model for call in fake_transport.calls] == ["primary", "backup"]


def test_json_repair_stays_on_model_that_returned_the_response(
    fake_transport, monkeypatch,
):
    from math_agent.errors import LLMServerError

    monkeypatch.setattr(llm_mod, "_DEFAULT_VALIDATION_REPAIRS", 1)
    fake_transport.enqueue_error(LLMServerError("502"))
    fake_transport.enqueue_ok("not json")
    fake_transport.enqueue_ok('{"summary":"ok","score":8}')

    out = llm_mod.complete(
        "hi", schema=_Answer, model="primary", fallback_models=["backup"]
    )

    assert out.score == 8
    assert [call.model for call in fake_transport.calls] == [
        "primary", "backup", "backup",
    ]


def test_unhealthy_primary_is_bypassed_until_cooldown_expires(
    fake_transport, fake_clock, monkeypatch,
):
    from math_agent.errors import LLMServerError

    monkeypatch.setattr(llm_mod, "_MODEL_FAILURE_COOLDOWN", 30.0)
    monkeypatch.setattr(llm_mod, "_MODEL_MAX_COOLDOWN", 300.0)
    fake_transport.enqueue_error(LLMServerError("502"))
    fake_transport.enqueue_ok("first")
    fake_transport.enqueue_ok("second")
    fake_transport.enqueue_ok("third")
    fake_transport.enqueue_error(LLMServerError("502 again"))
    fake_transport.enqueue_ok("fourth")

    assert llm_mod.complete(
        "one", model="primary", fallback_models=["backup"]
    ) == "first"
    assert llm_mod.complete(
        "two", model="primary", fallback_models=["backup"]
    ) == "second"
    fake_clock.advance(31.0)
    assert llm_mod.complete(
        "three", model="primary", fallback_models=["backup"]
    ) == "third"
    assert llm_mod.complete(
        "four", model="primary", fallback_models=["backup"]
    ) == "fourth"

    assert [call.model for call in fake_transport.calls] == [
        "primary", "backup", "backup", "primary", "primary", "backup",
    ]
    assert llm_mod._MODEL_FAILURE_STREAK["primary"] == 2
    assert llm_mod._MODEL_UNHEALTHY_UNTIL["primary"] == pytest.approx(91.0)


def test_complete_logs_to_current_tracer(fake_transport, tmp_path):
    from math_agent.tracing import Tracer, set_current, reset_current
    fake_transport.enqueue_ok("hi", prompt_tokens=10, completion_tokens=5)

    t = Tracer(thread_id="t", out_dir=tmp_path)
    tok = set_current(t)
    try:
        llm_mod.complete("ping", model="gpt-4o-mini")
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


def test_json_escape_preserves_latex_backslash_b_t_f():
    """LLM 在 JSON 里输出 LaTeX 命令 \beta/\tau/\frac 时，
    \b \t \f 是合法 JSON escape，json.loads 会吃掉反斜杠。
    修复用 raw string r"\b" r"\t" r"\f" 双写反斜杠。"""
    raw = r'{"result": "use $\beta$ and $\tau$ and $\frac{1}{2}$"}'
    content = raw
    for esc in (r"\b", r"\t", r"\f"):
        content = content.replace(esc, r"\\" + esc[1:])
    import json
    parsed = json.loads(content)["result"]
    assert r"\beta" in parsed, f"\beta 被损坏: {parsed}"
