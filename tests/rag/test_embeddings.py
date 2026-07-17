import pytest

from math_agent.errors import LLMTimeoutError
from math_agent.rag.embeddings import embed_texts


class _FakeEmbeddingTransport:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def embed(self, model, input, *, timeout_s):
        self.calls.append((model, list(input), timeout_s))
        item = next(self.responses)
        if isinstance(item, BaseException):
            raise item
        return item


def test_embed_texts_calls_litellm_embedding(mocker):
    fake = _FakeEmbeddingTransport([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
    mocker.patch("math_agent.rag.embeddings._embedding_transport", fake)

    out = embed_texts(["a", "b"], model="text-embedding-3-small")
    assert out == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_embed_texts_batches_when_over_limit(mocker):
    """batch_size=2，3 段文本应切成 2 次调用。"""
    fake = _FakeEmbeddingTransport([
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        [[2.0, 2.0, 2.0]],
    ])
    mocker.patch("math_agent.rag.embeddings._embedding_transport", fake)
    out = embed_texts(["x", "y", "z"], model="m", batch_size=2)
    assert len(fake.calls) == 2
    assert len(out) == 3


def test_embedding_timeout_is_not_retried(mocker):
    fake = _FakeEmbeddingTransport([LLMTimeoutError("timeout")])
    mocker.patch("math_agent.rag.embeddings._embedding_transport", fake)

    with pytest.raises(LLMTimeoutError):
        embed_texts(["x"], model="m")

    assert len(fake.calls) == 1
