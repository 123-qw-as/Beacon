"""轻量 tracing：把 LLM 调用与节点执行写入一份 trace.json。

设计：
- 同步、单线程；不引入 OTel/LangSmith（它们留作可选 callback）。
- Tracer 暴露 log_llm() 与 node() 上下文管理器；调用方负责打点。
- 在 llm.complete 内统一调 Tracer（通过 contextvars，避免显式传参）。
"""
from __future__ import annotations

import contextvars
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


_current: "contextvars.ContextVar[Tracer | None]" = contextvars.ContextVar(
    "math_agent_tracer", default=None,
)


@dataclass
class _NodeRecord:
    name: str
    start_ms: int
    duration_ms: int = 0


@dataclass
class Tracer:
    thread_id: str
    out_dir: Path

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    per_model: dict = field(default_factory=dict)
    nodes: list = field(default_factory=list)

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 公共 API ----

    def log_llm(self, *, model: str, prompt_tokens: int, completion_tokens: int,
                latency_ms: int) -> None:
        self.llm_calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        m = self.per_model.setdefault(model, {
            "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0,
        })
        m["calls"] += 1
        m["prompt_tokens"] += prompt_tokens
        m["completion_tokens"] += completion_tokens
        m["latency_ms"] += latency_ms

    @contextmanager
    def node(self, name: str):
        # 用 monotonic_ns 测 duration，避免 NTP 校时回拨导致负值。
        start_ns = time.monotonic_ns()
        rec = _NodeRecord(name=name, start_ms=start_ns // 1_000_000)
        try:
            yield
        finally:
            rec.duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000
            self.nodes.append(rec)

    def flush(self) -> Path:
        out = self.out_dir / "trace.json"
        out.write_text(json.dumps({
            "thread_id": self.thread_id,
            "llm_calls": self.llm_calls,
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
            },
            "per_model": self.per_model,
            "nodes": [{"name": r.name, "duration_ms": r.duration_ms}
                      for r in self.nodes],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return out


# ---- 全局当前 Tracer 句柄 ----

def set_current(tracer: "Tracer | None") -> contextvars.Token:
    return _current.set(tracer)


def get_current() -> "Tracer | None":
    return _current.get()


def reset_current(token: contextvars.Token) -> None:
    _current.reset(token)


# ---- 全局最后节点名（供 CLI 报错时显示） ----

_last_node: "contextvars.ContextVar[str]" = contextvars.ContextVar(
    "math_agent_last_node", default="(unknown)",
)


def set_last_node(name: str) -> contextvars.Token:
    return _last_node.set(name)


def get_last_node() -> str:
    return _last_node.get()


def reset_last_node(token: contextvars.Token) -> None:
    _last_node.reset(token)
