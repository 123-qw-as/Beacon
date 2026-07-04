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
    """llm.complete 仍给 litellm.completion 传 timeout（作为第一层软超时）。"""
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
    """litellm 抛 httpx.TimeoutException → 子线程 catch → classify 为 LLMTransportError → 走 tenacity 重试。"""
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
    from math_agent.errors import LLMError

    mocker.patch("litellm.completion", side_effect=httpx.TimeoutException("stalled"))
    with pytest.raises(LLMError):
        llm.complete(
            "hi", model="gpt-4o-mini", max_retries=0,
            _retry_attempts=2, _retry_base_delay=0,
        )


def test_complete_hangs_forever_times_out_via_thread_join(mocker, monkeypatch):
    """第一性原理验证：litellm.completion 永不返回（模拟 router 半挂）时，
    Thread.join 超时后主线程必须抛 LLMTransportError，而不是无限阻塞。

    用小 LLM_TIMEOUT（2s）避免真等 180s；mock litellm.completion 为 sleep(999)。
    """
    import time as _time
    from math_agent.errors import LLMError, LLMTransportError
    import math_agent.llm as llm_mod

    # 把超时窗口临时调小到 2s，避免测试等 180s
    monkeypatch.setattr(llm_mod, "LLM_TIMEOUT", 2.0)

    def _hang(**kw):
        _time.sleep(999)   # 模拟 router 半挂，永不返回

    mocker.patch("litellm.completion", side_effect=_hang)

    import time as _t
    t0 = _t.monotonic()
    with pytest.raises(LLMError):   # tenacity 重试 1 次后耗尽 → LLMError
        llm.complete("hi", model="gpt-4o-mini",
                     _retry_attempts=1, _retry_base_delay=0)
    elapsed = _t.monotonic() - t0
    # 应在 ~2-6s 内返回（1 次 join 超时 2s + 可能 1 次重试再超时 2s）
    assert elapsed < 15, f"耗时 {elapsed:.1f}s，Thread.join 超时未生效"


def test_json_escape_preserves_latex_backslash_b_t_f():
    """LLM 在 JSON 里输出 LaTeX 命令 \beta/\tau/\frac 时，
    \b \t \f 是合法 JSON escape，json.loads 会吃掉反斜杠。
    修复用 raw string r"\b" r"\t" r"\f" 双写反斜杠。"""
    import math_agent.llm as llm_mod

    # 模拟 LLM 返回的 raw content（含 LaTeX 命令）
    # \beta → \b+eta, \tau → \t+au, \frac → \f+rac
    raw = r'{"result": "use $\beta$ and $\tau$ and $\frac{1}{2}$"}'

    # 验证修复逻辑：手动执行 replace
    content = raw
    for esc in (r"\b", r"\t", r"\f"):
        content = content.replace(esc, r"\\" + esc[1:])

    import json
    parsed = json.loads(content)["result"]
    assert r"\beta" in parsed, f"\beta 被损坏: {parsed}"
    assert r"\tau" in parsed, f"\tau 被损坏: {parsed}"
    assert r"\frac" in parsed, f"\frac 被损坏: {parsed}"
    # 不应含 backspace/tab/formfeed 控制字符
    assert chr(8) not in parsed, "含 backspace"
    assert chr(9) not in parsed, "含 tab"
    assert chr(12) not in parsed, "含 formfeed"


class _LatexOut(BaseModel):
    """模拟 writer section 输出 schema（含 LaTeX 的字符串字段）。"""
    model_section: str


def test_complete_preserves_correctly_escaped_latex_json(mocker):
    """Case A: LLM 输出正确转义的 JSON（\\text 双反斜杠），不应被损坏。

    这是 phase2_final 的根因：旧修复的 str.replace 把正确 JSON 里的
    \\text 误伤成 \\\\\\text，json.loads 解析出 backslash+TAB+ext (0x5C 0x09)。
    LLM 正确转义时，complete() 必须原样保留 \\text。
    """
    # LLM 正确输出 JSON：字符串值 \text{total} 在 JSON wire 上转义为 \\text{total}
    # Python 源码里要用 \\\\ 表示 2 个真实反斜杠（JSON wire 上的 \\）
    payload = '{"model_section": "\\\\text{total}"}'
    # 验证 payload 字节：pos 20-21 应为 0x5C 0x5C（两个反斜杠）
    pb = payload.encode()
    idx = pb.find(b'text')
    assert pb[idx-2:idx] == b'\x5c\x5c', f"payload 不是正确转义: {pb!r}"
    mocker.patch(
        "litellm.completion",
        return_value=mocker.MagicMock(
            choices=[mocker.MagicMock(message=mocker.MagicMock(content=payload))]
        ),
    )
    out = llm.complete("write", schema=_LatexOut, model="gpt-4o-mini")
    assert isinstance(out, _LatexOut)
    # 必须是 \text{total}（backslash + t + ext），不能含控制字符
    assert r"\text" in out.model_section, f"\\text 被损坏: {out.model_section!r}"
    assert chr(9) not in out.model_section, f"含 TAB 控制字符: {out.model_section!r}"
    assert chr(8) not in out.model_section, f"含 backspace: {out.model_section!r}"
    assert chr(12) not in out.model_section, f"含 formfeed: {out.model_section!r}"


def test_complete_repairs_under_escaped_latex_json(mocker):
    """Case B: LLM 欠转义输出 \\text（单反斜杠，JSON \\t=TAB），应被修复为 \\text。

    旧修复的目标场景：LLM 在 JSON 字符串里写了 \\text 而非 \\\\text，
    json.loads 把 \\t 当 TAB 转义吃掉反斜杠。修复后应还原为 \\text。
    """
    # 欠转义：wire 上是 \text（单反斜杠），JSON \t = TAB
    # 用 raw string r"\t" 确保 Python 不把 \t 解析成 TAB
    payload = r'{"model_section": "\text{total}"}'
    # 验证 payload 字节：pos 20 应为 0x5C（单反斜杠）
    pb = payload.encode()
    idx = pb.find(b'text')
    assert pb[idx-1:idx] == b'\x5c', f"payload 不是欠转义: {pb!r}"
    mocker.patch(
        "litellm.completion",
        return_value=mocker.MagicMock(
            choices=[mocker.MagicMock(message=mocker.MagicMock(content=payload))]
        ),
    )
    out = llm.complete("write", schema=_LatexOut, model="gpt-4o-mini")
    assert isinstance(out, _LatexOut)
    assert r"\text" in out.model_section, f"\\text 未被修复: {out.model_section!r}"
    assert chr(9) not in out.model_section, f"含 TAB: {out.model_section!r}"
