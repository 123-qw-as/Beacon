import pytest
from pydantic import BaseModel
import math_agent.llm as llm


class _Answer(BaseModel):
    summary: str
    score: int


def test_complete_returns_text_when_no_schema(mocker):
    mocker.patch(
        "litellm.completion",
        return_value=mocker.MagicMock(
            choices=[mocker.MagicMock(message=mocker.MagicMock(content="hello"))]
        ),
    )
    out = llm.complete("say hi", model="gpt-4o-mini")
    assert out == "hello"


def test_complete_returns_pydantic_when_schema(mocker):
    payload = '{"summary": "ok", "score": 9}'
    mocker.patch(
        "litellm.completion",
        return_value=mocker.MagicMock(
            choices=[mocker.MagicMock(message=mocker.MagicMock(content=payload))]
        ),
    )
    out = llm.complete("rate it", schema=_Answer, model="gpt-4o-mini")
    assert isinstance(out, _Answer)
    assert out.score == 9


def test_complete_retries_on_invalid_json(mocker):
    bad = mocker.MagicMock(choices=[mocker.MagicMock(message=mocker.MagicMock(content="not json"))])
    good = mocker.MagicMock(choices=[mocker.MagicMock(message=mocker.MagicMock(content='{"summary":"x","score":1}'))])
    mocker.patch("litellm.completion", side_effect=[bad, good])
    out = llm.complete("x", schema=_Answer, model="gpt-4o-mini", max_retries=2)
    assert out.score == 1


def test_complete_raises_after_all_retries_exhausted(mocker):
    bad = mocker.MagicMock(choices=[mocker.MagicMock(message=mocker.MagicMock(content="nope"))])
    mocker.patch("litellm.completion", return_value=bad)
    with pytest.raises(llm.LLMError):
        llm.complete("x", schema=_Answer, max_retries=1)
