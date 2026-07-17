"""可取消的 embedding 包装：批处理、硬 deadline、timeout 不重放。"""
from __future__ import annotations

from math_agent.config import EMBED_TIMEOUT
from math_agent.transport import LiteLLMWorkerTransport


_embedding_transport = LiteLLMWorkerTransport()


def _do_embed(model: str, input: list[str]) -> list[list[float]]:
    """单次 embedding；timeout 会终止 worker，默认不自动重放请求。"""
    return _embedding_transport.embed(model, input, timeout_s=EMBED_TIMEOUT)


def embed_texts(texts: list[str], *, model: str, batch_size: int = 64) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        out.extend(_do_embed(model, texts[i : i + batch_size]))
    return out
