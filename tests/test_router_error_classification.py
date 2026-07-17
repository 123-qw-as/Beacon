"""9router 合成错误的分类回归测试。"""
from math_agent.errors import LLMTimeoutError, classify_exception
from math_agent.transport import _classify_worker_error
from math_agent.transport import FakeCompletionTransport
import math_agent.llm as llm
from math_agent.errors import LLMServerError


def test_router_fetch_connect_timeout_is_timeout_even_when_status_is_502():
    message = "BadGatewayError: fetch connect timeout"
    assert isinstance(_classify_worker_error({
        "class": "InternalServerError", "msg": message, "status_code": 502,
    }), LLMTimeoutError)
    assert isinstance(classify_exception(RuntimeError(f"502 {message}")), LLMTimeoutError)


def test_max_retries_zero_disables_beacon_server_retry(monkeypatch):
    transport = FakeCompletionTransport()
    transport.enqueue_error(LLMServerError("502"))
    transport.enqueue_ok("must not be consumed")
    monkeypatch.setattr(llm, "_get_transport", lambda: transport)
    try:
        llm.complete("x", model="test", max_retries=0)
    except LLMServerError:
        pass
    else:
        raise AssertionError("应直接抛出 502")
    assert transport.send_count == 1
