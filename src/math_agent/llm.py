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
import re
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
# 强制本地 LLM 调用不走系统代理（Windows clash/v2ray 常设系统代理，
# httpx 会读系统代理设置把 localhost:20128 的请求也走代理转发，导致
# 代理转发本地请求时 socket read 永久阻塞。NO_PROXY 让 localhost 直连。
# 用 setdefault 不覆盖用户显式设的值。）
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
            return parsed
        # 直接解析失败——回退到反斜杠双写后重试
        # P1: 双写所有非法 JSON 转义（\h \s \m \S \l 等，排除合法的 \b \f \n \r \t \u \" \\ \/）
        # 用正则 (?<!\\) 确保不误伤已正确转义的 \\hat（第二个 \ 前有 \）
        patched = _double_illegal_json_backslashes(content)
        if patched != content:
            parsed, _ = _parse_json_with_latex_repair(patched, schema)
            if parsed is not None:
                return parsed
        # 两种方式都失败，进入重试
        last_err = LLMValidationError(parse_err or "JSON 解析失败")
        parse_feedback = (content, parse_err or "JSON 解析失败")
        continue

    raise LLMError(f"LLM 调用失败：{last_err}")


def _parse_json_with_latex_repair(
    content: str, schema: Type[T]
) -> tuple[T | None, str | None]:
    """解析 JSON 并修复 LaTeX 转义损坏的控制字符。

    返回 (schema 实例, None) 成功，或 (None, 错误消息) 失败。
    策略：先 model_validate_json，成功后递归扫描所有字符串字段，
    把 0x08/0x09/0x0C 控制字符还原为 \\b/\\t/\\f（字面反斜杠+字母）。
    """
    try:
        obj = schema.model_validate_json(content)
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
