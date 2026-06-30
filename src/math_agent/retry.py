"""统一重试装饰器，基于 tenacity。

设计原则：
- 只有 *可重试* 的错误才重试；其他错误透传，让调用方/上游处理。
- 不在装饰器内做 sleep；用 tenacity 的 wait_exponential。
- 装饰器是同步版本；如未来引入 async，再扩 async_llm_retry。
- max_attempts 默认从 config 读，避免与 Plan A 的 MAX_LLM_RETRIES 漂移；调用方可显式覆盖。

可重试集合：
- llm_retry: LLMRateLimitError, LLMTransportError
- runner_retry: RunnerError 的全部子类
"""
from __future__ import annotations

from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
)

from math_agent.errors import (
    LLMRateLimitError, LLMTransportError, RunnerError,
)


def _default_llm_attempts() -> int:
    # 与 Plan A 的 MAX_LLM_RETRIES 保持同源；MAX_LLM_RETRIES 是"次数"语义（首次 + N 次重试），
    # tenacity 的 stop_after_attempt 是"总尝试次数"语义，故 attempts = MAX_LLM_RETRIES + 1。
    from math_agent.config import MAX_LLM_RETRIES
    return MAX_LLM_RETRIES + 1


def llm_retry(*, max_attempts: int | None = None, base_delay: float = 1.0, max_delay: float = 30.0):
    attempts = max_attempts if max_attempts is not None else _default_llm_attempts()
    return retry(
        retry=retry_if_exception_type((LLMRateLimitError, LLMTransportError)),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=base_delay, min=base_delay, max=max_delay),
        reraise=True,
    )


def runner_retry(*, max_attempts: int = 2, base_delay: float = 0.5):
    return retry(
        retry=retry_if_exception_type(RunnerError),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=base_delay, min=base_delay, max=5.0),
        reraise=True,
    )
