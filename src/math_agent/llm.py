"""LiteLLM 统一入口。

设计原则：
- 所有节点必须通过 complete() 调用 LLM，不直接 import litellm。
- 支持 Pydantic schema → 结构化输出 + JSON 解析重试。
- 错误统一抛 LLMError，调用方决定降级策略。
"""
from __future__ import annotations

import json
from typing import Any, Type, TypeVar

import litellm
from pydantic import BaseModel, ValidationError

from math_agent.config import DEFAULT_MODEL, MAX_LLM_RETRIES
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """LLM 调用或结构化解析失败。"""


def complete(
    prompt: str,
    *,
    schema: Type[T] | None = None,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.3,
    max_retries: int = MAX_LLM_RETRIES,
    **kwargs: Any,
) -> str | T:
    """统一 LLM 调用。

    - schema 为 None 时返回纯文本；
    - schema 为 Pydantic 模型时强制 JSON 输出，解析失败会自动重试，并把"上次输出 + 错误"喂回模型让它修正。

    错误状态分两类，互不污染：
    - 网络/服务异常 (litellm.completion 抛错)：换一次，不构造反馈消息。
    - JSON 解析失败：把上次内容 + 解析错误打包成 parse_feedback，下一轮注入。
    """
    model = model or DEFAULT_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
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
            raw = litellm.completion(
                model=model,
                messages=msgs,
                temperature=temperature,
                response_format=response_format,
                **kwargs,
            )
        except Exception as e:  # 网络 / 鉴权 / 限流：不动 parse_feedback
            last_err = e
            continue

        content = raw.choices[0].message.content or ""
        if schema is None:
            return content
        try:
            return schema.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError) as e:
            last_err = e
            parse_feedback = (content, e)
            continue

    raise LLMError(f"LLM 调用失败：{last_err}")
