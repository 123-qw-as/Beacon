"""litellm.embedding 包装：支持批处理 + 经 retry 装饰。"""
from __future__ import annotations

# 继承 llm.py 模块级一次性抑制（LiteLLM logger / debug info 已在 llm 模块导入时设好）
import litellm

from math_agent.config import EMBED_TIMEOUT
from math_agent.errors import classify_exception
from math_agent.retry import llm_retry


def _do_embed(model: str, input: list[str]) -> list[list[float]]:
    try:
        # 与 _do_completion 同理：litellm 默认无 httpx timeout，ollama 半挂
        # 会让 RAG retrieval 永挂。超时被 classify 为 LLMTransportError → 退避重试。
        resp = litellm.embedding(model=model, input=input, timeout=EMBED_TIMEOUT)
    except Exception as e:
        raise classify_exception(e)
    return [item["embedding"] for item in resp.data]


@llm_retry(max_attempts=4, base_delay=1.0)
def _embed_with_retry(model: str, input: list[str]) -> list[list[float]]:
    return _do_embed(model, input)


def embed_texts(texts: list[str], *, model: str, batch_size: int = 64) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        out.extend(_embed_with_retry(model, texts[i : i + batch_size]))
    return out
