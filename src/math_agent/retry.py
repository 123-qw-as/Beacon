"""统一重试装饰器，基于 tenacity。

设计原则：
- 只有 *可重试* 的错误才重试；其他错误透传，让调用方/上游处理。
- wait 策略：指数退避 + 随机 jitter，避免 thundering herd / 同步重试打爆同一抖动的 router。
- 装饰器是同步版本；如未来引入 async，再扩 async_llm_retry。
- max_attempts 默认从 config 读，避免与 Plan A 的 MAX_LLM_RETRIES 漂移；调用方可显式覆盖。

可重试集合：
- llm_retry: LLMRateLimitError, LLMTransportError
- runner_retry: RunnerError 的全部子类

可调旋钮（环境变量，_seconds 级）：
  MATH_AGENT_LLM_RETRY_ATTEMPTS  默认 5（首次 + 4 重试）
  MATH_AGENT_LLM_RETRY_BASE_DELAY 默认 2.0 s
  MATH_AGENT_LLM_RETRY_MAX_DELAY  默认 60 s
  base_delay=2, max=60, 5 attempts -> 2,4,8,16,32 - jitter-window -> 累计 ~62s, ~323s 个 phase
  足以熬过本地 router 短暂 502 / fetch-connect-timeout。
"""
from __future__ import annotations

import os
import random

from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
)

from math_agent.errors import (
    LLMRateLimitError, LLMTransportError, RunnerError,
)


def _default_llm_attempts() -> int:
    # 与 Plan A 的 MAX_LLM_RETRIES 保持同源；MAX_LLM_RETRIES 是"次数"语义（首次 + N 次重试），
    # tenacity 的 stop_after_attempt 是"总尝试次数"语义，故 attempts = MAX_LLM_RETRIES + 1。
    # Plan C 复盘：本地 router 偶有 502/timeout，原 3 次太紧；改成可被 env override，默认 5。
    from math_agent.config import MAX_LLM_RETRIES
    return int(os.getenv("MATH_AGENT_LLM_RETRY_ATTEMPTS", MAX_LLM_RETRIES + 3))


def _default_llm_base_delay() -> float:
    return float(os.getenv("MATH_AGENT_LLM_RETRY_BASE_DELAY", "2.0"))


def _default_llm_max_delay() -> float:
    return float(os.getenv("MATH_AGENT_LLM_RETRY_MAX_DELAY", "60.0"))


def llm_retry(*, max_attempts: int | None = None,
              base_delay: float | None = None,
              max_delay: float | None = None):
    attempts = max_attempts if max_attempts is not None else _default_llm_attempts()
    base_delay = base_delay if base_delay is not None else _default_llm_base_delay()
    max_delay = max_delay if max_delay is not None else _default_llm_max_delay()

    # wait_exponential + 随机抖动：把固定退避打散 ±25%，避免多并发场景同步重试打爆 router。
    def _wait(retry_state):
        # tenacity 提供 retry_state.outcome_exception 但我们用退避指数 + 随机 jitter
        n = retry_state.attempt_number
        exp = min(base_delay * (2 ** (n - 1)), max_delay)
        jitter = exp * 0.25 * (random.random() - 0.5) * 2  # -25%..+25%
        return max(0.0, exp + jitter)

    return retry(
        retry=retry_if_exception_type((LLMRateLimitError, LLMTransportError)),
        stop=stop_after_attempt(attempts),
        wait=_wait,
        reraise=True,
    )


def runner_retry(*, max_attempts: int = 2, base_delay: float = 0.5):
    return retry(
        retry=retry_if_exception_type(RunnerError),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=base_delay, min=base_delay, max=5.0),
        reraise=True,
    )