"""LLM 传输层 seam（§4.2）。

对外只暴露 CompletionTransport 协议和数据类；complete() 通过模块内部 fixture
注入 adapter，业务节点不感知传输细节。

生产 adapter（LiteLLMWorkerTransport）在阶段 B 实现；FakeCompletionTransport
供测试模拟成功/挂起/超时/429/5xx/非法 JSON。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class CompletionRequest:
    """一次物理请求的入参（普通 dict 友好，可跨 IPC 序列化）。"""
    model: str
    messages: list[dict]
    temperature: float = 0.3
    response_format: dict | None = None
    timeout_s: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_wire(self) -> dict:
        return {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "response_format": self.response_format,
            "timeout_s": self.timeout_s,
            "extra": self.extra,
        }


@dataclass
class CompletionResponse:
    """一次物理请求的出参（仅内容 + usage + 元数据，不含 prompt/key）。"""
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    @classmethod
    def from_wire(cls, d: dict) -> "CompletionResponse":
        return cls(
            content=d["content"],
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            model=d.get("model", ""),
        )


@dataclass
class EmbeddingRequest:
    model: str
    input: list[str]
    timeout_s: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_wire(self) -> dict:
        return {
            "model": self.model, "input": self.input,
            "timeout_s": self.timeout_s, "extra": self.extra,
        }


@runtime_checkable
class CompletionTransport(Protocol):
    """同步传输 port：发送一次请求，在 timeout_s 内返回或抛 LLMError 子类。"""

    def send(self, request: CompletionRequest, *, timeout_s: float) -> CompletionResponse: ...


@dataclass
class _QueuedResponse:
    """FakeTransport 的单条队列项。"""
    kind: str           # "ok" | "hang" | "error"
    content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Exception | None = None
    status_code: int | None = None
    retry_after: float | None = None
    delay_s: float = 0.0  # 返回前推进的模拟延迟（fake clock 下生效）


class FakeCompletionTransport:
    """测试 adapter：按预设队列返回响应或异常，记录所有调用。

    用法：
        t = FakeCompletionTransport()
        t.enqueue_ok("hi")
        t.enqueue_hang()
        t.enqueue_error(LLMRateLimitError("429"))
        resp = t.send(req, timeout_s=5)
    """

    def __init__(self, clock: "FakeClock | None" = None):
        self._queue: list[_QueuedResponse] = []
        self.calls: list[CompletionRequest] = []
        self.send_count = 0
        self._clock = clock

    def enqueue_ok(self, content: str, *, prompt_tokens: int = 10,
                   completion_tokens: int = 5, delay_s: float = 0.0) -> None:
        self._queue.append(_QueuedResponse("ok", content=content,
                                           prompt_tokens=prompt_tokens,
                                           completion_tokens=completion_tokens,
                                           delay_s=delay_s))

    def enqueue_hang(self) -> None:
        self._queue.append(_QueuedResponse("hang"))

    def enqueue_error(self, error: Exception, *, status_code: int | None = None,
                      retry_after: float | None = None) -> None:
        self._queue.append(_QueuedResponse("error", error=error,
                                           status_code=status_code,
                                           retry_after=retry_after))

    def send(self, request: CompletionRequest, *, timeout_s: float) -> CompletionResponse:
        self.calls.append(request)
        self.send_count += 1
        if not self._queue:
            raise RuntimeError("FakeCompletionTransport queue empty")
        item = self._queue.pop(0)
        if item.kind == "hang":
            # 模拟永不返回：真实 transport 会在这里被 worker deadline 杀掉；
            # FakeTransport 推进时钟直到 timeout_s，由调用方（budget/timeout 层）裁决。
            clk = self._clock
            if clk is not None:
                # fake clock：推进到 deadline 后抛 timeout（无需真 sleep）
                clk.advance(timeout_s)
                from math_agent.errors import LLMTimeoutError
                raise LLMTimeoutError(f"fake hang, {timeout_s}s 超时")
            # 真实时钟：忙等到 deadline（阶段 B 的 worker 会真终止，这里兜底）
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                time.sleep(0.01)
            from math_agent.errors import LLMTimeoutError
            raise LLMTimeoutError(f"fake hang, {timeout_s}s 超时")
        if item.kind == "error":
            assert item.error is not None
            if self._clock is not None and item.delay_s:
                self._clock.advance(item.delay_s)
            raise item.error
        # ok
        if self._clock is not None and item.delay_s:
            self._clock.advance(item.delay_s)
        return CompletionResponse(
            content=item.content,
            prompt_tokens=item.prompt_tokens,
            completion_tokens=item.completion_tokens,
            model=request.model,
        )


class FakeClock:
    """可控单调时钟：测试中前进时间而无需真 sleep。

    预算计算用 monotonic()，FakeClock 替换 time.monotonic 即可驱动 deadline 判断。
    """

    def __init__(self, start: float = 0.0):
        self._t = start

    def monotonic(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


# ---- 生产 adapter（§4.3）----

import os
import multiprocessing as _mp

from math_agent.errors import (
    LLMError, LLMTimeoutError, LLMRateLimitError, LLMConnectionError,
    LLMServerError, LLMAuthenticationError, LLMInvalidRequestError,
    LLMProtocolError,
)

_KILL_GRACE = float(os.getenv("MATH_AGENT_LLM_WORKER_KILL_GRACE", "2"))


def _classify_worker_error(err: dict) -> LLMError:
    """把 worker 序列化的 error dict 映射为 LLMError 子类（§6.1）。"""
    cls = err.get("class", "")
    msg = err.get("msg", "")
    status = err.get("status_code")
    blob = f"{cls} {msg}".lower()
    # 9router 会把自身的连接 deadline 包装成 HTTP 502；语义上仍是超时，
    # 不能按普通 5xx 再叠加多轮重试。
    if "fetch connect timeout" in blob or "connect timeout" in blob:
        return LLMTimeoutError(msg)
    # 502/5xx 服务端错误优先于 timeout 关键词（502 消息常含 "connect timeout" 但本质是 server error）
    if status in (502, 503, 504) or "badgateway" in blob or "serviceunavailable" in blob:
        return LLMServerError(msg)
    if status == 429 or "ratelimit" in blob or "rate limit" in blob:
        return LLMRateLimitError(msg)
    if status in (401, 403) or "auth" in blob:
        return LLMAuthenticationError(msg)
    if status in (400, 404) or "notfound" in blob or "badrequest" in blob:
        return LLMInvalidRequestError(msg)
    # 超时类（仅当不是 5xx 时）
    if "timeout" in blob or "timed out" in blob:
        return LLMTimeoutError(msg)
    if "connection" in blob or "connect" in blob:
        return LLMConnectionError(msg)
    if "protocol" in blob:
        return LLMProtocolError(msg)
    return LLMError(msg)


class LiteLLMWorkerTransport:
    """生产 adapter：常驻 LiteLLM worker 进程，poll 实现 deadline，terminate/kill 回收。

    - 惰性启动 worker，正常复用。
    - timeout 后 terminate -> 短宽限 -> kill -> join，清空引用，下次惰性重启。
    - A.1：worker 启动后发 ready 握手，import 成本不计入首次 poll 窗口。
    """

    def __init__(self, kill_grace: float = _KILL_GRACE):
        self._worker = None  # (Process, PipeConnection)
        self._kill_grace = kill_grace
        self._ctx = _mp.get_context("spawn")

    def _ensure_worker(self):
        """惰性启动 worker 并等待 ready 握手（A.1）。返回 (Process, conn)。"""
        if self._worker is not None:
            return self._worker
        parent_conn, child_conn = self._ctx.Pipe()
        p = self._ctx.Process(
            target=_worker_entrypoint, args=(child_conn,), daemon=True,
        )
        p.start()
        child_conn.close()  # 父进程关闭子端
        # A.1 等待 ready（import 完成）；给 30s 上限避免 spawn 卡死
        if not parent_conn.poll(30):
            self._terminate(p, parent_conn)
            raise LLMConnectionError("worker 启动握手超时（30s 未 ready）")
        try:
            msg = parent_conn.recv()
        except (EOFError, OSError) as e:
            self._terminate(p, parent_conn)
            raise LLMConnectionError(f"worker 握手失败：{e}")
        if not isinstance(msg, dict) or msg.get("type") != "ready":
            self._terminate(p, parent_conn)
            raise LLMConnectionError(f"worker 握手消息异常：{msg}")
        self._worker = (p, parent_conn)
        return self._worker

    def send(self, request: CompletionRequest, *, timeout_s: float) -> CompletionResponse:
        p, conn = self._ensure_worker()
        # 排空残留（上一次超时后可能有未读消息）
        while conn.poll(0):
            try:
                conn.recv()
            except (EOFError, OSError):
                break
        try:
            conn.send({"type": "req", "payload": request.to_wire()})
        except (BrokenPipeError, OSError) as e:
            self._discard_worker()
            raise LLMConnectionError(f"发送请求失败（worker 可能已死）：{e}")

        if not conn.poll(timeout_s):
            # deadline 到达：终止并回收 worker
            self._discard_worker()
            raise LLMTimeoutError(
                f"LLM 调用 {timeout_s}s 未返回，worker 已终止"
            )
        try:
            msg = conn.recv()
        except (EOFError, OSError) as e:
            self._discard_worker()
            raise LLMConnectionError(f"读取响应失败（worker 崩溃）：{e}")

        if not isinstance(msg, dict):
            self._discard_worker()
            raise LLMProtocolError(f"worker 响应非 dict：{type(msg)}")
        if msg.get("type") == "ok":
            return CompletionResponse.from_wire(msg.get("payload", {}))
        if msg.get("type") == "err":
            raise _classify_worker_error(msg.get("error", {}))
        self._discard_worker()
        raise LLMProtocolError(f"worker 响应类型未知：{msg.get('type')}")

    def embed(
        self, model: str, input: list[str], *, timeout_s: float,
        extra: dict | None = None,
    ) -> list[list[float]]:
        """在同类可强杀 worker 中执行 embedding；timeout 后不残留后台线程。"""
        p, conn = self._ensure_worker()
        while conn.poll(0):
            try:
                conn.recv()
            except (EOFError, OSError):
                break
        request = EmbeddingRequest(
            model=model, input=input, timeout_s=timeout_s, extra=extra or {},
        )
        try:
            conn.send({"type": "embed", "payload": request.to_wire()})
        except (BrokenPipeError, OSError) as exc:
            self._discard_worker()
            raise LLMConnectionError(f"发送 embedding 请求失败：{exc}")
        if not conn.poll(timeout_s):
            self._discard_worker()
            raise LLMTimeoutError(f"embedding 调用 {timeout_s}s 未返回，worker 已终止")
        try:
            msg = conn.recv()
        except (EOFError, OSError) as exc:
            self._discard_worker()
            raise LLMConnectionError(f"读取 embedding 响应失败：{exc}")
        if not isinstance(msg, dict):
            self._discard_worker()
            raise LLMProtocolError(f"embedding worker 响应非 dict：{type(msg)}")
        if msg.get("type") == "embed_ok":
            embeddings = (msg.get("payload") or {}).get("embeddings")
            if not isinstance(embeddings, list):
                raise LLMProtocolError("embedding worker 缺少 embeddings 数组")
            return embeddings
        if msg.get("type") == "err":
            raise _classify_worker_error(msg.get("error", {}))
        self._discard_worker()
        raise LLMProtocolError(f"embedding worker 响应类型未知：{msg.get('type')}")

    def _discard_worker(self):
        """终止并回收当前 worker，清空引用。"""
        if self._worker is None:
            return
        p, conn = self._worker
        self._worker = None
        self._terminate(p, conn)

    def _terminate(self, p, conn):
        """关闭连接 -> terminate -> 宽限 -> kill -> join（§4.3）。"""
        try:
            conn.close()
        except Exception:
            pass
        if p.is_alive():
            p.terminate()
            p.join(self._kill_grace)
            if p.is_alive():
                p.kill()
                p.join(self._kill_grace)

    def close(self):
        """优雅关闭：发 stop 再 join。"""
        if self._worker is None:
            return
        p, conn = self._worker
        self._worker = None
        try:
            conn.send({"type": "stop"})
            p.join(self._kill_grace)
        except Exception:
            pass
        finally:
            if p.is_alive():
                p.terminate()
                p.join(self._kill_grace)
                if p.is_alive():
                    p.kill()
                    p.join(self._kill_grace)
            try:
                conn.close()
            except Exception:
                pass


def _worker_entrypoint(child_conn):
    """spawn 入口：import worker 模块并运行主循环。"""
    from math_agent.llm_worker import _worker_main
    _worker_main(child_conn)
