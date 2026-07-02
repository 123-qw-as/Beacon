"""litellm.embedding 包装：支持批处理 + 经 retry 装饰。"""
from __future__ import annotations

import threading

# 继承 llm.py 模块级一次性抑制（LiteLLM logger / debug info 已在 llm 模块导入时设好）
import litellm

from math_agent.config import EMBED_TIMEOUT
from math_agent.errors import LLMTransportError, classify_exception
from math_agent.retry import llm_retry


def _do_embed(model: str, input: list[str]) -> list[list[float]]:
    """单次 litellm.embedding 调用，用 Thread.join 强制超时窗口兜底。

    与 llm._do_completion 同一种第一性原理：不依赖 litellm 内部 timeout 透传
    （实测不可靠），而是把阻塞调用隔离到 daemon 子线程，主线程 join 超时后
    必然继续。ollama 半挂时不会无限阻塞 retrieval 路径。
    """
    box: dict = {}

    def _run():
        try:
            box["resp"] = litellm.embedding(model=model, input=input, timeout=EMBED_TIMEOUT)
        except BaseException as e:
            box["err"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(EMBED_TIMEOUT)

    if t.is_alive():
        raise LLMTransportError(
            f"embedding 调用 {EMBED_TIMEOUT}s 未返回（ollama 半挂？），强制超时"
        )
    if "err" in box:
        raise classify_exception(box["err"])
    return [item["embedding"] for item in box["resp"].data]


@llm_retry(max_attempts=4, base_delay=1.0)
def _embed_with_retry(model: str, input: list[str]) -> list[list[float]]:
    return _do_embed(model, input)


def embed_texts(texts: list[str], *, model: str, batch_size: int = 64) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        out.extend(_embed_with_retry(model, texts[i : i + batch_size]))
    return out
