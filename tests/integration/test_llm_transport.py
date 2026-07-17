"""§10.3 本地 HTTP 集成测试。

启动可控的本地 OpenAI-compatible fake server，覆盖：
- /hang: 接受请求后不返回（模拟 router 半挂）
- /slow: 延迟后返回合法响应
- /rate-limit: 先返回 429 再成功
- /invalid-json: 返回协议损坏内容
- /disconnect: 响应中途断开

重点断言 /hang 超时后 worker 进程退出、客户端连接被关闭，
随后正常请求由新 worker 完成，不受前一个请求影响。

用 stdlib http.server，无外部依赖。真实 spawn worker，不 mock litellm。
"""
from __future__ import annotations

import json
import socketserver
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import pytest

from math_agent.transport import LiteLLMWorkerTransport, CompletionRequest
from math_agent.errors import (
    LLMTimeoutError, LLMRateLimitError, LLMProtocolError, LLMError,
)


class _FakeOpenAIServer(BaseHTTPRequestHandler):
    """OpenAI-compatible fake server，按 path 返回不同故障模式。"""

    # /rate-limit 用实例属性决定返回 429 还是 200（通过 server 对象共享）
    def _rate_limit_state(self):
        server = self.server
        if not hasattr(server, "_rl_hits"):
            server._rl_hits = 0
        return server

    def do_POST(self):
        path = self.path.split("?")[0]
        # litellm 向 {api_base}/chat/completions 发请求
        # 测试用 api_base=http://host/v1/<mode> -> path=/v1/<mode>/chat/completions
        # 读取 body（litellm 会发 JSON）
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)

        if path.endswith("/embeddings"):
            if "/embed-hang/" in path:
                time.sleep(600)
                return
            body = json.dumps({
                "object": "list",
                "data": [{"object": "embedding", "index": 0,
                          "embedding": [0.1, 0.2, 0.3]}],
                "model": "text-embedding-test",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())
            return

        if "/hang/" in path or path.endswith("/hang"):
            time.sleep(600)  # 永不返回
            return

        if "/slow/" in path or path.endswith("/slow"):
            time.sleep(0.3)
            self._send_ok("slow response")
            return

        if "/rate-limit/" in path or path.endswith("/rate-limit"):
            state = self._rate_limit_state()
            state._rl_hits += 1
            if state._rl_hits <= 1:
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                body = json.dumps({"error": {"message": "Rate limit exceeded"}})
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body.encode())
                return
            self._send_ok("after rate limit")
            return

        if "/invalid-json/" in path or path.endswith("/invalid-json"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            body = "this is {not valid json at all"
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())
            return

        if "/disconnect/" in path or path.endswith("/disconnect"):
            # 发一半响应然后断开
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "99999")  # 声称很大但只发一点
            self.end_headers()
            self.wfile.write(b'{"choices":[{"message":{"content":"par')
            self.wfile.close()
            return

        # 默认：正常响应
        self._send_ok("default ok")

    def _send_ok(self, content: str):
        body = json.dumps({
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        })
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass  # 静默


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """多线程 HTTP server：每个请求独立线程，避免 /hang 阻塞 /slow。"""
    daemon_threads = True


@pytest.fixture
def fake_server():
    """启动 fake OpenAI server，返回 base_url。"""
    server = _ThreadedHTTPServer(("127.0.0.1", 0), _FakeOpenAIServer)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    yield base_url
    server.shutdown()


def _make_transport(base_url: str, path: str, **extra) -> tuple[LiteLLMWorkerTransport, CompletionRequest]:
    t = LiteLLMWorkerTransport(kill_grace=1)
    req = CompletionRequest(
        model="openai/gpt-test",
        messages=[{"role": "user", "content": "hi"}],
        timeout_s=3.0,
        extra={"api_base": f"{base_url}{path}", "api_key": "sk-fake", **extra},
    )
    return t, req


# ---- §10.3 /hang：超时后 worker 退出，下次请求由新 worker 完成 ----

def test_hang_timeout_terminates_worker_and_restarts(fake_server):
    """/hang 超时后 worker 被终止回收；下一次正常请求由新 worker 完成。"""
    t = LiteLLMWorkerTransport(kill_grace=1)
    req_hang = CompletionRequest(
        model="openai/gpt-test",
        messages=[{"role": "user", "content": "hi"}],
        timeout_s=2.0,
        extra={"api_base": f"{fake_server}/v1/hang", "api_key": "sk-fake"},
    )

    t0 = time.monotonic()
    with pytest.raises(LLMTimeoutError):
        t.send(req_hang, timeout_s=2.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 10.0, f"超时耗时 {elapsed:.1f}s，worker 未被及时终止"
    assert t._worker is None, "timeout 后 worker 未被回收"

    # 下一次正常请求由新 worker 完成（复用同一 transport，惰性重启）
    req_slow = CompletionRequest(
        model="openai/gpt-test",
        messages=[{"role": "user", "content": "hi"}],
        timeout_s=10.0,
        extra={"api_base": f"{fake_server}/v1/slow", "api_key": "sk-fake"},
    )
    resp = t.send(req_slow, timeout_s=10.0)
    assert resp.content == "slow response"
    t.close()


def test_embedding_hang_terminates_worker_and_next_embedding_succeeds(fake_server):
    t = LiteLLMWorkerTransport(kill_grace=1)
    with pytest.raises(LLMTimeoutError):
        t.embed(
            "openai/text-embedding-test", ["hello"], timeout_s=1.0,
            extra={"api_base": f"{fake_server}/v1/embed-hang", "api_key": "sk-fake"},
        )
    assert t._worker is None

    vectors = t.embed(
        "openai/text-embedding-test", ["hello"], timeout_s=10.0,
        extra={"api_base": f"{fake_server}/v1/embed-ok", "api_key": "sk-fake"},
    )
    assert vectors == [[0.1, 0.2, 0.3]]
    t.close()


# ---- §10.3 /slow：延迟后返回合法响应 ----

def test_slow_returns_after_delay(fake_server):
    """/slow 延迟 0.3s 后返回，在 timeout 内成功。"""
    t, req = _make_transport(fake_server, "/v1/slow")
    resp = t.send(req, timeout_s=5.0)
    assert resp.content == "slow response"
    t.close()


# ---- §10.3 /rate-limit：先 429 再成功 ----

def test_rate_limit_then_success(fake_server):
    """/rate-limit 第一次 429，第二次成功。验证 429 分类和重试。"""
    t, req = _make_transport(fake_server, "/v1/rate-limit")
    # transport.send 是单次尝试；429 分类由 _classify_worker_error 处理
    # 第一次 send 应抛 LLMRateLimitError
    with pytest.raises(LLMRateLimitError):
        t.send(req, timeout_s=5.0)
    # 第二次 send（新请求）应成功
    resp = t.send(req, timeout_s=5.0)
    assert resp.content == "after rate limit"
    t.close()


# ---- §10.3 /invalid-json：返回协议损坏内容 ----

def test_invalid_json_raises_error(fake_server):
    """/invalid-json 返回非 JSON -> litellm 解析失败 -> 抛 LLMError 子类。"""
    t, req = _make_transport(fake_server, "/v1/invalid-json")
    with pytest.raises(LLMError):
        t.send(req, timeout_s=5.0)
    t.close()


# ---- §10.3 /disconnect：响应中途断开 ----

def test_disconnect_raises_connection_error(fake_server):
    """/disconnect 发一半断开 -> 连接错误。"""
    t, req = _make_transport(fake_server, "/v1/disconnect")
    with pytest.raises(LLMError):
        t.send(req, timeout_s=5.0)
    # worker 可能因连接损坏需要回收（断开后的 worker 状态不确定）
    # 但下一次请求应能正常工作
    t.close()


# ---- 综合：worker 复用 ----

def test_worker_reused_across_successful_requests(fake_server):
    """连续成功请求复用同一 worker（不重新 spawn）。"""
    t, req = _make_transport(fake_server, "/v1/slow")
    resp1 = t.send(req, timeout_s=5.0)
    worker1 = t._worker
    assert worker1 is not None

    resp2 = t.send(req, timeout_s=5.0)
    worker2 = t._worker
    assert worker2 is not None
    # 同一 worker 实例（复用）
    assert worker1[0] is worker2[0], "worker 未复用，每次都重新 spawn"

    assert resp1.content == "slow response"
    assert resp2.content == "slow response"
    t.close()
