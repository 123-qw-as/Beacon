from math_agent.rag.embeddings import embed_texts


def test_embed_texts_calls_litellm_embedding(mocker):
    fake = mocker.MagicMock()
    fake.data = [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]
    mocker.patch("litellm.embedding", return_value=fake)

    out = embed_texts(["a", "b"], model="text-embedding-3-small")
    assert out == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_embed_texts_batches_when_over_limit(mocker):
    """batch_size=2，3 段文本应切成 2 次调用。"""
    call_count = {"n": 0}

    def _fake(model, input, **kw):
        call_count["n"] += 1
        return type("R", (), {
            "data": [{"embedding": [float(call_count["n"])] * 3} for _ in input]
        })()

    mocker.patch("litellm.embedding", side_effect=_fake)
    out = embed_texts(["x", "y", "z"], model="m", batch_size=2)
    assert call_count["n"] == 2
    assert len(out) == 3
