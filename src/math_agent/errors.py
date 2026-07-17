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


# ---- 细分错误类型（§6.1）----
# 保留 LLMTransportError 作为兼容父类，新增按重试策略区分的子类。
class LLMTransportError(LLMError):
    """兼容父类：网络/超时类错误。新代码应抛更具体的子类。"""


class LLMAuthenticationError(LLMError):
    """401/403 鉴权失败，不重试。"""


class LLMInvalidRequestError(LLMError):
    """400/404 请求错误，不重试。"""


class LLMProtocolError(LLMError):
    """协议损坏（非法响应体），不重试。"""


class LLMRateLimitError(LLMError):
    """429，在 deadline 内退避重试。"""


class LLMConnectionError(LLMTransportError):
    """连接建立失败，可重试。"""


class LLMServerError(LLMTransportError):
    """5xx 服务端错误，部分可重试。"""


class LLMTimeoutError(LLMTransportError):
    """读取/生成超时，默认不自动重试。"""


class LLMBudgetExceededError(LLMError):
    """逻辑调用总 deadline 耗尽，不重试。"""


class LLMValidationError(LLMError):
    """结构化输出解析失败，进入 JSON 修复流程。"""


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


class FinalizationError(MathAgentError):
    """终态不变量未满足；保留 finalizer checkpoint，等待修复后恢复。"""


_RATE_LIMIT_HINTS = ("RateLimitError", "rate limit", "429")
# 超时类（读取/生成超时）：默认不重试
_TIMEOUT_HINTS = (
    "Timeout", "ReadTimeout", "APITimeoutError", "httpx.TimeoutException",
    "PoolTimeout", "timed out", "timed-out",
)
# 连接建立失败：可重试
_CONNECTION_HINTS = (
    "APIConnectionError", "ConnectionError", "Connection error",
    "ConnectError", "RemoteProtocolError", "NetworkError",
)
# 5xx 服务端错误：部分可重试
_SERVER_HINTS = (
    "InternalServerError", "ServiceUnavailable", "502", "503", "504",
)
# 鉴权/请求错误：不重试
_AUTH_HINTS = ("AuthenticationError", "401", "403", "PermissionDenied")
_INVALID_REQUEST_HINTS = ("BadRequestError", "400", "404", "NotFoundError")


def classify_exception(e: BaseException) -> MathAgentError:
    """把任意异常归一化为 MathAgentError 子类（§6.1 错误分类）。

    已是 MathAgentError -> 原样返回。
    按 timeout > connection > server > rate_limit > auth > invalid_request 顺序匹配，
    timeout 优先级最高（服务端可能仍在生成，默认不重试）。
    """
    if isinstance(e, MathAgentError):
        return e
    name = type(e).__name__
    msg = str(e)
    blob = f"{name} {msg}".lower()
    # 9router 的合成 502：实际表示连接 deadline 已耗尽，应立即交给 checkpoint
    # 恢复机制，而不是再按普通 5xx 重试。
    if "fetch connect timeout" in blob or "connect timeout" in blob:
        return LLMTimeoutError(msg)
    # 502/5xx 优先于 timeout 关键词（BadGatewayError 消息常含 "connect timeout"）
    if any(h.lower() in blob for h in _SERVER_HINTS):
        return LLMServerError(msg)
    if any(h.lower() in blob for h in _TIMEOUT_HINTS):
        return LLMTimeoutError(msg)
    if any(h.lower() in blob for h in _CONNECTION_HINTS):
        return LLMConnectionError(msg)
    if any(h.lower() in blob for h in _RATE_LIMIT_HINTS):
        return LLMRateLimitError(msg)
    if any(h.lower() in blob for h in _AUTH_HINTS):
        return LLMAuthenticationError(msg)
    if any(h.lower() in blob for h in _INVALID_REQUEST_HINTS):
        return LLMInvalidRequestError(msg)
    # 协议损坏（响应无法解析）--留作兜底 LLMError，调用方按不可重试处理
    return LLMError(msg)
