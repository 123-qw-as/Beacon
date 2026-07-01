"""集中错误类型。

层次：
  MathAgentError
    ├── LLMError
    │     ├── LLMRateLimitError       # 触发指数退避
    │     ├── LLMValidationError      # 结构化输出解析失败；可"喂回错误"重试
    │     └── LLMTransportError       # 网络/超时；短间隔重试
    ├── RunnerError                   # 对应 tools/runner.py 的 subprocess 执行错误
    │     ├── RunnerTimeoutError
    │     └── RunnerRuntimeError
    └── LatexError
          ├── LatexMissingBinaryError # 不应重试
          └── LatexCompileError       # 可解析 .log 给出建议

命名约定：与 `tools/runner.py`（Plan A 已重命名）保持一致——所有错误类、结果字段、
fixture 都用 `Runner*` 前缀，不再使用历史名 `Sandbox*`。

分类原则：依据**重试策略不同**才单独建类；同策略归一类。
"""
from __future__ import annotations


class MathAgentError(Exception):
    """所有自定义错误的根。"""


class LLMError(MathAgentError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMValidationError(LLMError):
    pass


class LLMTransportError(LLMError):
    pass


class RunnerError(MathAgentError):
    pass


class RunnerTimeoutError(RunnerError):
    pass


class RunnerRuntimeError(RunnerError):
    pass


class LatexError(MathAgentError):
    pass


class LatexMissingBinaryError(LatexError):
    pass


class LatexCompileError(LatexError):
    pass


_RATE_LIMIT_HINTS = ("RateLimitError", "rate limit", "429")
_TRANSPORT_HINTS = (
    "APIConnectionError", "Timeout", "ReadTimeout", "ConnectionError",
    "Connection error",         # litellm/openai passes this in the message body
    "InternalServerError",      # 5xx from upstream router / gateway
    "ServiceUnavailable", "502", "503", "504",
)


def classify_exception(e: BaseException) -> MathAgentError:
    """把任意异常归一化为 MathAgentError 子类。

    - 已是 MathAgentError → 原样返回
    - 类名/消息命中 rate limit / transport → 对应子类
    - 否则视为通用 LLMError（调用方决定是否重试）
    """
    if isinstance(e, MathAgentError):
        return e
    name = type(e).__name__
    msg = str(e)
    blob = f"{name} {msg}"
    if any(h in blob for h in _RATE_LIMIT_HINTS):
        return LLMRateLimitError(msg)
    if any(h in blob for h in _TRANSPORT_HINTS):
        return LLMTransportError(msg)
    return LLMError(msg)
