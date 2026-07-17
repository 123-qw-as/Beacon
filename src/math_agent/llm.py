"""Beacon 的统一 LLM 入口。

业务节点只依赖 :func:`complete`。物理请求由可终止的 worker transport 执行；
传输重试、结构化输出修复与退避共享同一个单调时钟 deadline。
"""
from __future__ import annotations

import atexit
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from math_agent.config import DEFAULT_MODEL, LLM_FALLBACK_MODELS
from math_agent.errors import (
    LLMBudgetExceededError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    LLMValidationError,
    MathAgentError,
    classify_exception,
)
from math_agent.tracing import get_current as _get_tracer
from math_agent.transport import (
    CompletionRequest,
    CompletionResponse,
    LiteLLMWorkerTransport,
)

T = TypeVar("T", bound=BaseModel)


class _SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


_SYSTEM_CLOCK = _SystemClock()
_TRANSPORT: LiteLLMWorkerTransport | None = None
_MODEL_UNHEALTHY_UNTIL: dict[str, float] = {}
_MODEL_FAILURE_STREAK: dict[str, int] = {}
_MODEL_SUCCESS_STREAK: dict[str, int] = {}
_MODEL_HEALTH_FILENAME = "llm_model_health.json"


def _float_env(name: str, default: float) -> float:
    try:
        return max(0.01, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


_MODEL_FAILURE_COOLDOWN = _float_env("MATH_AGENT_LLM_MODEL_COOLDOWN", 300.0)
_MODEL_MAX_COOLDOWN = _float_env("MATH_AGENT_LLM_MODEL_MAX_COOLDOWN", 1800.0)


_LEGACY_TIMEOUT = _float_env("MATH_AGENT_LLM_TIMEOUT", 180.0)
_DEFAULT_ATTEMPT_TIMEOUT = _float_env(
    "MATH_AGENT_LLM_ATTEMPT_TIMEOUT", _LEGACY_TIMEOUT
)
_DEFAULT_TOTAL_TIMEOUT = _float_env(
    "MATH_AGENT_LLM_TOTAL_TIMEOUT",
    min(_DEFAULT_ATTEMPT_TIMEOUT * 1.5, _DEFAULT_ATTEMPT_TIMEOUT + 180.0),
)
# standard 故意使用上面的两个默认值，便于测试和运行时诊断覆盖；长文本和视觉
# 画像在此单独配置。
_PROFILE_ATTEMPT_TIMEOUT = {
    "code": _float_env("MATH_AGENT_LLM_CODE_ATTEMPT_TIMEOUT", 90.0),
    "long": _float_env("MATH_AGENT_LLM_LONG_ATTEMPT_TIMEOUT", 300.0),
    "vision": _float_env("MATH_AGENT_LLM_VISION_ATTEMPT_TIMEOUT", _DEFAULT_ATTEMPT_TIMEOUT),
}
_PROFILE_TOTAL_TIMEOUT = {
    "code": _float_env("MATH_AGENT_LLM_CODE_TOTAL_TIMEOUT", 240.0),
    "long": _float_env("MATH_AGENT_LLM_LONG_TOTAL_TIMEOUT", 420.0),
    "vision": _float_env("MATH_AGENT_LLM_VISION_TOTAL_TIMEOUT", _DEFAULT_TOTAL_TIMEOUT),
}
_PROFILE_FALLBACK_RESERVE = {
    "standard": _float_env("MATH_AGENT_LLM_FALLBACK_RESERVE", 120.0),
    "code": _float_env("MATH_AGENT_LLM_CODE_FALLBACK_RESERVE", 120.0),
    "long": _float_env("MATH_AGENT_LLM_LONG_FALLBACK_RESERVE", 180.0),
    "vision": _float_env("MATH_AGENT_LLM_VISION_FALLBACK_RESERVE", 120.0),
}
try:
    _DEFAULT_VALIDATION_REPAIRS = max(
        0, int(os.getenv("MATH_AGENT_LLM_VALIDATION_REPAIRS", "2"))
    )
except ValueError:
    _DEFAULT_VALIDATION_REPAIRS = 2


def _get_clock():
    """内部测试 seam；业务节点不应替换时钟。"""
    return _SYSTEM_CLOCK


def _get_transport():
    """返回进程内复用的可终止 LLM worker transport。"""
    global _TRANSPORT
    if _TRANSPORT is None:
        _TRANSPORT = LiteLLMWorkerTransport()
    return _TRANSPORT


def _close_transport() -> None:
    global _TRANSPORT
    transport, _TRANSPORT = _TRANSPORT, None
    if transport is not None:
        transport.close()


atexit.register(_close_transport)


def _profile_timeout(profile: str) -> tuple[float, float]:
    if profile not in {"standard", "code", "long", "vision"}:
        raise ValueError(f"未知 LLM 调用画像：{profile}")
    attempt = _PROFILE_ATTEMPT_TIMEOUT.get(profile, _DEFAULT_ATTEMPT_TIMEOUT)
    total = _PROFILE_TOTAL_TIMEOUT.get(profile, _DEFAULT_TOTAL_TIMEOUT)
    return attempt, total


def _sleep_with_clock(clock, seconds: float) -> None:
    if seconds <= 0:
        return
    if hasattr(clock, "sleep"):
        clock.sleep(seconds)
    elif hasattr(clock, "advance"):
        clock.advance(seconds)
    else:
        time.sleep(seconds)


def _attempt_limit(error: LLMError) -> int:
    if isinstance(error, LLMRateLimitError):
        return 3
    if isinstance(error, (LLMConnectionError, LLMServerError)):
        return 2
    # timeout 不自动重试；远端可能仍在执行原请求。
    return 1


def _backoff_seconds(error: LLMError, failure_index: int) -> float:
    if isinstance(error, LLMRateLimitError):
        return float(2 ** max(0, failure_index - 1))
    if isinstance(error, (LLMConnectionError, LLMServerError)):
        return float(max(1, failure_index))
    return 0.0


def _model_candidates(
    primary: str, fallback_models: list[str] | tuple[str, ...] | None
) -> tuple[str, ...]:
    """返回主模型和备用模型，去空、去重且不改变声明顺序。"""
    configured = LLM_FALLBACK_MODELS if fallback_models is None else fallback_models
    candidates: list[str] = []
    for item in (primary, *configured):
        candidate = str(item).strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def _can_fail_over(error: LLMError) -> bool:
    """仅对可确认未在上游继续执行的故障切模型。

    读超时发生时，上游可能仍在生成。此时切换模型会复制同一个昂贵请求，
    并与路由器内部的账号回退叠加，因此超时必须直接交给调用方处理。
    """
    return isinstance(
        error,
        (LLMConnectionError, LLMServerError, LLMRateLimitError),
    )


def _order_candidates_by_health(
    candidates: tuple[str, ...], *, now: float
) -> tuple[str, ...]:
    """把熔断中的模型移到末尾；全部熔断时先试最早恢复者。"""
    healthy = [m for m in candidates if _MODEL_UNHEALTHY_UNTIL.get(m, 0.0) <= now]
    unhealthy = [m for m in candidates if m not in healthy]
    if healthy:
        return tuple(healthy + unhealthy)
    return tuple(sorted(candidates, key=lambda m: _MODEL_UNHEALTHY_UNTIL.get(m, 0.0)))


def _model_health_path() -> Path | None:
    """把模型熔断状态限定在当前 run，不写入全局配置或 prompt。"""
    tracer = _get_tracer()
    if tracer is None:
        return None
    return Path(tracer.out_dir) / _MODEL_HEALTH_FILENAME


def _persist_model_health(*, now: float) -> None:
    """用 wall-clock expiry 持久化 cooldown，使新 worker 可继续选路。"""
    path = _model_health_path()
    if path is None:
        return
    models = {}
    wall_now = time.time()
    for model, until in _MODEL_UNHEALTHY_UNTIL.items():
        remaining = max(0.0, until - now)
        if remaining <= 0:
            continue
        models[model] = {
            "until_epoch": wall_now + remaining,
            "failure_streak": _MODEL_FAILURE_STREAK.get(model, 1),
        }
    if not models:
        path.unlink(missing_ok=True)
        return
    payload = {"version": 1, "models": models}
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _load_persisted_model_health(*, now: float) -> None:
    """把 run 级 wall-clock expiry 映射到本进程的 monotonic 时钟。"""
    path = _model_health_path()
    if path is None or not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        models = payload.get("models", {})
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, TypeError):
        return
    if not isinstance(models, dict):
        return
    wall_now = time.time()
    changed = False
    for model, record in models.items():
        if not isinstance(record, dict):
            changed = True
            continue
        try:
            remaining = float(record.get("until_epoch", 0.0)) - wall_now
            streak = max(1, int(record.get("failure_streak", 1)))
        except (TypeError, ValueError, OverflowError):
            changed = True
            continue
        if remaining <= 0:
            changed = True
            continue
        name = str(model)
        _MODEL_UNHEALTHY_UNTIL[name] = max(
            _MODEL_UNHEALTHY_UNTIL.get(name, 0.0), now + remaining
        )
        _MODEL_FAILURE_STREAK[name] = max(
            _MODEL_FAILURE_STREAK.get(name, 0), streak
        )
    if changed:
        _persist_model_health(now=now)


def _mark_model_unhealthy(model: str, *, now: float) -> None:
    streak = min(10, _MODEL_FAILURE_STREAK.get(model, 0) + 1)
    _MODEL_FAILURE_STREAK[model] = streak
    _MODEL_SUCCESS_STREAK.pop(model, None)
    cooldown = min(_MODEL_MAX_COOLDOWN, _MODEL_FAILURE_COOLDOWN * (2 ** (streak - 1)))
    _MODEL_UNHEALTHY_UNTIL[model] = max(
        _MODEL_UNHEALTHY_UNTIL.get(model, 0.0), now + cooldown
    )
    _persist_model_health(now=now)


def _mark_model_healthy(model: str, *, now: float | None = None) -> None:
    _MODEL_UNHEALTHY_UNTIL.pop(model, None)
    if model not in _MODEL_FAILURE_STREAK:
        return
    successes = _MODEL_SUCCESS_STREAK.get(model, 0) + 1
    if successes >= 2:
        _MODEL_FAILURE_STREAK.pop(model, None)
        _MODEL_SUCCESS_STREAK.pop(model, None)
    else:
        _MODEL_SUCCESS_STREAK[model] = successes
    _persist_model_health(now=_get_clock().monotonic() if now is None else now)


def _attempt_timeout_with_reserve(
    *, profile: str, attempt_timeout: float, remaining: float,
    candidates_after: int, total_fallbacks: int,
) -> float:
    """给尚未尝试的备用模型预留共享 deadline，避免主模型吃光预算。"""
    if candidates_after <= 0 or total_fallbacks <= 0:
        return min(attempt_timeout, remaining)
    total_reserve = min(
        _PROFILE_FALLBACK_RESERVE.get(profile, 60.0),
        max(0.0, remaining - 0.01),
    )
    reserve_per_candidate = total_reserve / total_fallbacks
    reserved = reserve_per_candidate * candidates_after
    return min(attempt_timeout, max(0.01, remaining - reserved))


# ---- 可选 callback（仅当对应环境变量存在时启用） ----

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
    # callback 在 worker 进程中依据相同环境变量配置；父进程这里只负责把该
    # 一次性检查固定下来，避免业务节点导入 LiteLLM。
    _resolve_callback_names()
    _LITELLM_CALLBACKS_CONFIGURED = True


def complete(
    prompt: str,
    *,
    schema: Type[T] | None = None,
    model: str | None = None,
    system: str | None = None,
    images: list[str] | None = None,
    profile: str = "standard",
    temperature: float = 0.3,
    max_validation_repairs: int | None = None,
    max_retries: int | None = None,
    fallback_models: list[str] | tuple[str, ...] | None = None,
    **kwargs: Any,
) -> str | T:
    """执行一次有总 deadline、可取消且可追踪的逻辑 LLM 调用。

    ``max_retries`` 仅作为旧调用方的传输重试上限兼容参数（0 表示不重试）；
    结构化输出修复请使用 ``max_validation_repairs``。主模型遇到超时、连接、
    限流或 5xx 时，会按 ``fallback_models``（默认读取集中配置）立即切换上游，
    但所有模型和 JSON 修复轮次仍共享同一个总 deadline。
    """
    _configure_callbacks_once()
    model = model or DEFAULT_MODEL
    messages: list[dict] = []
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

    repairs = (
        _DEFAULT_VALIDATION_REPAIRS
        if max_validation_repairs is None
        else max(0, int(max_validation_repairs))
    )
    attempt_timeout, total_timeout = _profile_timeout(profile)
    clock = _get_clock()
    started = clock.monotonic()
    deadline = started + total_timeout
    physical_attempt = 0
    parse_feedback: tuple[str, str] | None = None
    last_validation_error: LLMValidationError | None = None
    health_now = clock.monotonic()
    _load_persisted_model_health(now=health_now)
    candidates = _order_candidates_by_health(
        _model_candidates(model, fallback_models), now=health_now
    )
    preferred_model = candidates[0]
    successful_model = preferred_model

    for validation_round in range(repairs + 1):
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
        ordered_candidates = (
            preferred_model,
            *(candidate for candidate in candidates if candidate != preferred_model),
        )
        candidate_index = 0
        failure_counts: dict[str, int] = {}
        while True:
            remaining = deadline - clock.monotonic()
            if remaining <= 0:
                raise LLMBudgetExceededError(
                    f"LLM 逻辑调用总预算 {total_timeout:.1f}s 已耗尽"
                )
            active_model = ordered_candidates[candidate_index]
            # 超时不再跨模型重放，因此不能为了一个不会发生的超时回退而缩短
            # 主请求期限。502/429/断连通常会提前返回，仍可使用剩余总预算切换。
            timeout_s = min(attempt_timeout, remaining)
            physical_attempt += 1
            request_extra = dict(kwargs)
            request_extra["timeout"] = timeout_s
            request = CompletionRequest(
                model=active_model,
                messages=msgs,
                temperature=temperature,
                response_format=response_format,
                timeout_s=timeout_s,
                extra=request_extra,
            )
            tracer = _get_tracer()
            if tracer is not None:
                tracer.begin_attempt(
                    model=active_model, profile=profile, attempt=physical_attempt
                )
            attempt_started = clock.monotonic()
            try:
                response: CompletionResponse = _get_transport().send(
                    request, timeout_s=timeout_s
                )
                elapsed = clock.monotonic() - attempt_started
                if elapsed > timeout_s:
                    raise LLMTimeoutError(
                        f"LLM 调用超过单次时限 {timeout_s:.1f}s"
                    )
            except BaseException as raw_error:
                error = classify_exception(raw_error)
                elapsed_ms = int(
                    max(0.0, clock.monotonic() - attempt_started) * 1000
                )
                if tracer is not None:
                    tracer.end_attempt(
                        model=active_model,
                        status="timeout" if isinstance(error, LLMTimeoutError) else "failure",
                        latency_ms=elapsed_ms,
                        error_kind=type(error).__name__,
                    )
                if not isinstance(error, LLMError):
                    raise error
                # timeout 不在同一逻辑调用内重放，但要把主模型熔断状态保存到
                # run 目录；supervisor 从 checkpoint 启动新 worker 后会先试备用模型。
                if isinstance(error, LLMTimeoutError) or _can_fail_over(error):
                    _mark_model_unhealthy(active_model, now=clock.monotonic())
                failure_counts[active_model] = failure_counts.get(active_model, 0) + 1
                if (
                    _can_fail_over(error)
                    and candidate_index + 1 < len(ordered_candidates)
                ):
                    candidate_index += 1
                    continue
                allowed = _attempt_limit(error)
                if max_retries is not None:
                    allowed = min(allowed, max(1, int(max_retries) + 1))
                failure_count = failure_counts[active_model]
                if failure_count >= allowed:
                    raise error
                delay = _backoff_seconds(error, failure_count)
                if clock.monotonic() + delay >= deadline:
                    raise LLMBudgetExceededError(
                        f"LLM 重试等待将超过总预算 {total_timeout:.1f}s"
                    ) from error
                _sleep_with_clock(clock, delay)
                continue
            else:
                elapsed_ms = int(
                    max(0.0, clock.monotonic() - attempt_started) * 1000
                )
                if tracer is not None:
                    tracer.end_attempt(
                        model=active_model, status="success", latency_ms=elapsed_ms
                    )
                successful_model = active_model
                preferred_model = active_model
                _mark_model_healthy(active_model, now=clock.monotonic())
                break

        content = response.content or ""
        if schema is None:
            tracer = _get_tracer()
            if tracer is not None:
                tracer.log_llm(
                    model=successful_model,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    latency_ms=int(max(0.0, clock.monotonic() - started) * 1000),
                )
            return content
        # LLM 在 JSON 字符串里输出 LaTeX 命令 \beta/\big/\tau/\text/\top 时，
        # \b \t \f 都是合法 JSON 转义（backspace/tab/formfeed）。
        #
        # 两种损坏路径：
        #   Case A（正确转义）：LLM 写 \\text（JSON 标准），json.loads 正确得到 \text。
        #     旧修复的 str.replace 会把 \\text 里的 \t 误伤成 \\\text → 解析出 0x5C+0x09。
        #   Case B（欠转义）：LLM 写 \text（单反斜杠），json.loads 把 \t 当 TAB → 得到 0x09+ext。
        #
        # 修复策略：先直接解析（Case A 自然正确），解析后扫描控制字符并还原（Case B 修复）。
        # 只在直接解析失败时，才用反斜杠双写兜底重试。
        parsed, parse_err = _parse_json_with_latex_repair(content, schema)
        if parsed is not None:
            tracer = _get_tracer()
            if tracer is not None:
                tracer.log_llm(
                    model=successful_model,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    latency_ms=int(max(0.0, clock.monotonic() - started) * 1000),
                )
            return parsed
        # 直接解析失败——回退到反斜杠双写后重试
        # P1: 双写所有非法 JSON 转义（\h \s \m \S \l 等，排除合法的 \b \f \n \r \t \u \" \\ \/）
        # 用正则 (?<!\\) 确保不误伤已正确转义的 \\hat（第二个 \ 前有 \）
        patched = _double_illegal_json_backslashes(content)
        if patched != content:
            parsed, _ = _parse_json_with_latex_repair(patched, schema)
            if parsed is not None:
                tracer = _get_tracer()
                if tracer is not None:
                    tracer.log_llm(
                        model=successful_model,
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        latency_ms=int(max(0.0, clock.monotonic() - started) * 1000),
                    )
                return parsed
        last_validation_error = LLMValidationError(parse_err or "JSON 解析失败")
        parse_feedback = (content, parse_err or "JSON 解析失败")

    raise last_validation_error or LLMValidationError("JSON 解析失败")


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)



def _extract_json(content: str) -> str:
    """从 LLM 输出中提取 JSON 字符串。

    处理两类常见污染：
    1. <think></think> 思考标签（deepseek/kimi 系模型）：剥离。
    2. 前导/尾随非 JSON 文本（markdown 代码块、解释文字）：提取第一个 {...} 块。
    """
    cleaned = _THINK_TAG_RE.sub("", content).strip()
    start = cleaned.find("{")
    if start == -1:
        return cleaned

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start:i + 1]
    return cleaned[start:]


def _parse_json_with_latex_repair(
    content: str, schema: Type[T]
) -> tuple[T | None, str | None]:
    """解析 JSON 并修复 LaTeX 转义损坏的控制字符。

    返回 (schema 实例, None) 成功，或 (None, 错误消息) 失败。
    策略：先剥离 <think> 标签 + 提取 JSON 对象，再 model_validate_json，
    成功后递归扫描所有字符串字段，把 0x08/0x09/0x0C 控制字符还原为
    \\b/\\t/\\f（字面反斜杠+字母）。
    """
    json_str = _extract_json(content)
    try:
        obj = schema.model_validate_json(json_str)
    except (ValidationError, json.JSONDecodeError) as e:
        return None, str(e)
    _repair_control_chars_in_obj(obj)
    return obj, None


# 0x08/0x09/0x0C/0x0A/0x0D → 对应的 LaTeX 字面宏前缀（反斜杠+字母）
# json.loads 对 \b \t \f \n \r 静默接受（合法转义），会吃掉反斜杠产出裸控制字符
_CTRL_TO_LATEX = {0x08: "b", 0x09: "t", 0x0C: "f", 0x0A: "n", 0x0D: "r"}
_CTRL_CHARS = set(_CTRL_TO_LATEX)  # int codepoints

# JSON 合法转义目标字符：\b \f \n \r \t \uXXXX \" \\ \/
# 非法转义（\h \s \m \l \S 等）会触发 JSONDecodeError，需在 pre-parse 双写反斜杠
_ILLEGAL_JSON_ESCAPE_RE = re.compile(r'(?<!\\)\\([^bfnrtu"\\/])')


def _double_illegal_json_backslashes(content: str) -> str:
    r"""双写非法 JSON 转义的反斜杠，保留合法转义。

    LLM 欠转义 \hat 时，\h 不是合法 JSON 转义 → JSONDecodeError。
    此函数把 \h 双写成 \\h，让 json.loads 正确解析为字面 \h。
    合法转义 \b \f \n \r \t 等不动（LLM 可能故意用它们表示控制字符）。
    (?<!\\) 否定后顾：不匹配已正确转义的 \\hat 里的第二个 \。
    """
    return _ILLEGAL_JSON_ESCAPE_RE.sub(lambda m: "\\" + "\\" + m.group(1), content)

# 每个控制字符后可能跟的 LaTeX 宏字母前缀（损坏特征）。
# 只在这些前缀匹配时才还原，避免误伤正常换行/TAB。
# 例：0x0A + 'abla' → \nabla；0x09 + 'ext' → \text；0x08 + 'ar' → \bar
_CTRL_MACRO_PREFIXES = {
    0x08: ("ar", "oldsymbol", "egin", "eta", "igl", "igr", "ox", "old"),
    0x09: ("ext", "ilde", "heta", "imes", "op", "au", "abular", "au"),
    0x0C: ("rac", "orall", "lot", "igure"),
    0x0A: ("abla", "ewcommand", "ewenvironment", "ode", "onumber", "u",
           "ot", "eq", "ewline", "uance"),
    0x0D: ("brace", "floor", "ceil", "estriction", "angle", "ight"),
}


def _repair_string(s: str) -> str:
    r"""把字符串里被 json.loads 吃掉反斜杠的控制字符还原为 LaTeX 宏。

    策略：只有当控制字符后跟已知 LaTeX 宏字母前缀时才还原，
    避免误伤正常换行符(0x0A)、TAB 缩进(0x09)等合法控制字符。
    如果控制字符前已有反斜杠（Case D 三反斜杠），只补字母。
    """
    if not any(ord(c) in _CTRL_CHARS for c in s):
        return s
    result = []
    i = 0
    while i < len(s):
        ch = s[i]
        code = ord(ch)
        if code in _CTRL_TO_LATEX:
            letter = _CTRL_TO_LATEX[code]
            # 检查后跟是否匹配已知宏前缀
            rest = s[i + 1:i + 15]
            matched_prefix = None
            for prefix in _CTRL_MACRO_PREFIXES.get(code, ()):
                if rest.startswith(prefix):
                    matched_prefix = prefix
                    break
            if matched_prefix is not None:
                # 是损坏还原图：补反斜杠+字母（或只补字母，若前已有反斜杠）
                if result and result[-1] == "\\":
                    result.append(letter)
                else:
                    result.append("\\" + letter)
            else:
                # 不跟宏前缀 → 正常控制字符，保留不动
                result.append(ch)
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def _repair_control_chars_in_obj(obj: Any) -> None:
    """递归遍历 Pydantic 模型/dict/list，就地修复所有字符串字段的控制字符。"""
    if isinstance(obj, str):
        return  # str 本身不可变，上层处理
    if hasattr(obj, "__dict__"):
        for field_name in list(obj.__dict__):
            val = getattr(obj, field_name)
            if isinstance(val, str):
                setattr(obj, field_name, _repair_string(val))
            else:
                _repair_control_chars_in_obj(val)
    elif isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, str):
                obj[k] = _repair_string(v)
            else:
                _repair_control_chars_in_obj(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str):
                obj[i] = _repair_string(v)
            else:
                _repair_control_chars_in_obj(v)
