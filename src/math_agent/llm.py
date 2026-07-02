"""LiteLLM 统一入口。

设计原则：
- 所有节点必须通过 complete() 调用 LLM，不直接 import litellm。
- 支持 Pydantic schema → 结构化输出 + JSON 解析重试。
- 错误统一抛 LLMError 体系（math_agent.errors），调用方决定降级策略。
- rate limit / transport 类错误由 tenacity 自动指数退避重试；
  其它错误（鉴权、参数错误等）透传为 LLMError，不重试。
- JSON schema 解析失败属于"喂回错误重试"语义，在本函数内部循环处理，不走 tenacity。
"""
from __future__ import annotations

import json
import time
from typing import Any, Type, TypeVar

# 模块级一次性抑制 LiteLLM banner 噪音（每次 completion/embedding 都打极吵）
import logging as _logging
_pl = _logging.getLogger("LiteLLM")
_pl.propagate = False
_pl.handlers.clear()
_pl.addHandler(_logging.NullHandler())

import litellm
# 关掉 litellm 自带的各种 print 调试输出（Provider List 横幅 / cost map fetch 警告）
litellm.suppress_debug_info = True
litellm.set_verbose = False
import os
os.environ.setdefault("LITELLM_LOG", "CRITICAL")

from pydantic import BaseModel, ValidationError

from math_agent.config import DEFAULT_MODEL, MAX_LLM_RETRIES, LLM_TIMEOUT
from math_agent.errors import (
    LLMError, LLMValidationError, MathAgentError, classify_exception,
)
from math_agent.retry import llm_retry
from math_agent.tracing import get_current as _get_tracer

T = TypeVar("T", bound=BaseModel)


def _do_completion(**kw):
    """单次 litellm 调用 + 错误分类 + tracing 打点。"""
    # 兜底 httpx 无限阻塞：litellm 默认无 timeout，本地 router 半挂连接时
    # 会让 complete() 永不返回、tenacity 也进不去重试。显式设 per-request
    # 超时，命中后被 classify 为 LLMTransportError → 走 llm_retry 退避链。
    kw.setdefault("timeout", LLM_TIMEOUT)
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


def _completion_with_retry(*, _retry_attempts=None, _retry_base_delay=1.0, **kw):
    """对 rate_limit / transport 错误执行 tenacity 指数退避重试。

    `_retry_attempts=None` → 从 config.MAX_LLM_RETRIES 派生，避免重试预算两处定义漂移。
    """
    @llm_retry(max_attempts=_retry_attempts, base_delay=_retry_base_delay)
    def _call():
        return _do_completion(**kw)
    return _call()


# ---- 可选 callback（仅当对应环境变量存在时启用） ----

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


def complete(
    prompt: str,
    *,
    schema: Type[T] | None = None,
    model: str | None = None,
    system: str | None = None,
    images: list[str] | None = None,
    temperature: float = 0.3,
    max_retries: int = MAX_LLM_RETRIES,
    _retry_attempts: int | None = None,
    _retry_base_delay: float = 1.0,
    **kwargs: Any,
) -> str | T:
    """统一 LLM 调用。

    - schema 为 None 时返回纯文本；
    - schema 为 Pydantic 模型时强制 JSON 输出，解析失败会自动重试，并把"上次输出 + 错误"喂回模型让它修正。
    - images 为图片 data URL（`data:image/png;base64,...`）列表时走多模态格式。

    错误分两层：
    - 网络/限流/超时（litellm 抛错）：由 tenacity 在 `_completion_with_retry` 内部退避重试；
      超过预算后抛 LLMRateLimitError / LLMTransportError（均是 LLMError 子类）。
    - JSON schema 解析失败：把上次内容 + 解析错误打包成 parse_feedback，下一轮注入。
    """
    _configure_callbacks_once()
    model = model or DEFAULT_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if images:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *[{"type": "image_url", "image_url": {"url": u}} for u in images],
            ],
        })
    else:
        messages.append({"role": "user", "content": prompt})

    response_format = None
    if schema is not None:
        response_format = {"type": "json_object"}

    last_err: Exception | None = None
    parse_feedback: tuple[str, Exception] | None = None  # (上次内容, 解析错误)
    for attempt in range(max_retries + 1):
        msgs = list(messages)
        if schema and parse_feedback is not None:
            prev_content, prev_err = parse_feedback
            msgs.append({"role": "assistant", "content": prev_content})
            msgs.append(
                {
                    "role": "user",
                    "content": f"上一次响应无法被解析为目标 schema：{prev_err}。请只输出严格合法的 JSON。",
                }
            )
        try:
            raw = _completion_with_retry(
                model=model,
                messages=msgs,
                temperature=temperature,
                response_format=response_format,
                _retry_attempts=_retry_attempts,
                _retry_base_delay=_retry_base_delay,
                **kwargs,
            )
        except MathAgentError:
            # 已达 tenacity 重试上限的可重试错误，或不可重试的 LLMError —— 透传
            raise
        except Exception as e:
            # 兜底：理论上 _do_completion 已经 classify 过，这里再护一层
            raise classify_exception(e)

        content = raw.choices[0].message.content or ""
        if schema is None:
            return content
        try:
            return schema.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError) as e:
            last_err = LLMValidationError(str(e))
            parse_feedback = (content, e)
            continue

    raise LLMError(f"LLM 调用失败：{last_err}")
