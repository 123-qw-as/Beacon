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
import threading
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
# 强制本地 LLM 调用不走系统 HTTP 代理（Windows clash/v2ray 常设系统代理，
# httpx 会读系统代理设置把 localhost:20128 的请求也走代理转发，导致
# 代理转发本地请求时 socket read 永久阻塞。NO_PROXY 让 localhost 直连。）
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
# 强制本地请求不走系统代理：Windows 用户常开 clash/v2ray 系统代理（7892 等），
# litellm/httpx 会读 Windows 注册表 ProxyServer 把 localhost:20128 的请求也走
# 代理转发，代理转发本地请求时 socket read 永久阻塞 → Thread.join 每次等满
# 180s 才超时 → tenacity 5 次重试 = 15min 看起来像僵死。设 NO_PROXY 让本地
# LLM router / ollama 直连。用 setdefault 不覆盖用户显式设的值。
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from pydantic import BaseModel, ValidationError

from math_agent.config import DEFAULT_MODEL, MAX_LLM_RETRIES, LLM_TIMEOUT
from math_agent.errors import (
    LLMError, LLMTransportError, LLMValidationError, MathAgentError, classify_exception,
)
from math_agent.retry import llm_retry
from math_agent.tracing import get_current as _get_tracer

T = TypeVar("T", bound=BaseModel)


def _do_completion(**kw):
    """单次 litellm 调用，用 Thread.join 强制超时窗口兜底。

    第一性原理：litellm 内部 timeout 参数实测不可靠（不同 provider handler
    透传不一致），不能依赖第三方库自觉超时。Thread.join(timeout) 是 OS 级
    可靠原语，超时后主线程必然继续，无论 litellm/httpx 内部在做什么。
    子线程设 daemon=True，超时后不等待它——它会在 router 端 TCP keepalive
    失败后自然退出，或随进程结束被 OS 回收。这与 tools/runner.py 用
    subprocess.run(timeout=N) 是同一种哲学：调用方强制裁决，不信被调用方。
    """
    kw.setdefault("timeout", LLM_TIMEOUT)   # 第一层软超时（litellm 可能不透传，但有总比没有好）
    t0_ns = time.monotonic_ns()
    box: dict = {}

    def _run():
        try:
            box["resp"] = litellm.completion(**kw)
        except BaseException as e:   # BaseException: 连 KeyboardInterrupt 也接住交给主线程 classify
            box["err"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(LLM_TIMEOUT)

    if t.is_alive():
        # 子线程仍在跑 litellm → router 半挂，强制超时
        raise LLMTransportError(
            f"LLM 调用 {LLM_TIMEOUT}s 未返回（router 半挂？），强制超时"
        )
    if "err" in box:
        raise classify_exception(box["err"])

    resp = box["resp"]
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
        # LLM 在 JSON 字符串里输出 LaTeX 命令 \beta/\big/\tau/\text/\top 时，
        # \b \t \f 都是合法 JSON 转义（backspace/tab/formfeed），json.loads 会
        # 把 "\beta" 解析成 0x08+eta，"\tau" 解析成 tab+au，丢了反斜杠。
        # 修复：在 raw JSON 里把 \b \t \f 的反斜杠双写，让 json.loads 保留字面反斜杠。
        # 注意：必须用 raw string r"\t" 匹配字面 backslash+t，普通 "\\t" 是 TAB 字符。
        # ponytail: \n 不动——\newcommand 罕见且 \n 确实是换行意图居多
        for esc in (r"\b", r"\t", r"\f"):
            content = content.replace(esc, r"\\" + esc[1:])
        try:
            return schema.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError) as e:
            last_err = LLMValidationError(str(e))
            parse_feedback = (content, e)
            continue

    raise LLMError(f"LLM 调用失败：{last_err}")
