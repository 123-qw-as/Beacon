"""常驻 LiteLLM 工作进程入口与 IPC 协议（§4.3）。

worker 通过 multiprocessing spawn 启动，用 Pipe 传普通 dict。
父进程 connection.poll(timeout_s) 实现 deadline；超时 terminate/kill 回收。

IPC 协议（父 -> 子 -> 父）：
  父->子: {"type":"req", "payload": <request dict>}
  子->父: {"type":"ready"}                       # import 完成握手（A.1）
  子->父: {"type":"ok", "payload": <response dict>}
  子->父: {"type":"err", "error": <error dict>}
  父->子: {"type":"stop"}                        # 优雅退出

错误 dict: {"class": str, "msg": str, "status_code": int|None, "retry_after": float|None}
"""
from __future__ import annotations

import logging
import os

# spawn 上下文：子进程重新 import 本模块，这些 setdefault 在子进程也生效。
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("LITELLM_LOG", "CRITICAL")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import litellm  # noqa: E402  顶层 import：子进程启动即摊销，A.1

litellm.suppress_debug_info = True
litellm.set_verbose = False

_pl = logging.getLogger("LiteLLM")
_pl.propagate = False
_pl.handlers.clear()
_pl.addHandler(logging.NullHandler())


def _worker_main(conn) -> None:
    """worker 主循环。conn 为 Pipe 的一端。"""
    # A.1 握手：import 已在模块顶层完成，发 ready 通知父进程可以开始计时
    if not _safe_send(conn, {"type": "ready"}):
        return

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return  # 父进程关闭连接
        if not isinstance(msg, dict):
            continue
        if msg.get("type") == "stop":
            return
        if msg.get("type") not in {"req", "embed"}:
            continue

        req = msg.get("payload", {})
        try:
            if msg.get("type") == "embed":
                embed_kwargs = dict(req.get("extra") or {})
                embed_kwargs.update(
                    model=req["model"], input=req["input"],
                    timeout=req.get("timeout_s"), num_retries=0,
                )
                resp = litellm.embedding(**embed_kwargs)
                embeddings = [
                    item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding")
                    for item in resp.data
                ]
                if not _safe_send(
                    conn, {"type": "embed_ok", "payload": {"embeddings": embeddings}}
                ):
                    return
            else:
                resp = litellm.completion(**_to_litellm_kwargs(req))
                content = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                payload = {
                    "content": content,
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "model": req.get("model", ""),
                }
                if not _safe_send(conn, {"type": "ok", "payload": payload}):
                    return
        except BaseException as e:  # 连 KeyboardInterrupt 也序列化
            err = _serialize_error(e)
            # 父进程在 deadline/人工终止时会先关闭 Pipe；此时 worker 应安静退出，
            # 不能再因 BrokenPipeError 打出一整段误导性的二次 traceback。
            if not _safe_send(conn, {"type": "err", "error": err}):
                return


def _safe_send(conn, payload: dict) -> bool:
    try:
        conn.send(payload)
        return True
    except (BrokenPipeError, EOFError, OSError):
        return False


def _to_litellm_kwargs(req: dict) -> dict:
    """把 IPC request dict 转为 litellm.completion 入参。"""
    kw = {
        "model": req["model"],
        "messages": req["messages"],
        "temperature": req.get("temperature", 0.3),
        # §6.3：Beacon 是唯一重试编排者，关闭 LiteLLM client 内部重试
        "num_retries": 0,
    }
    if req.get("response_format"):
        kw["response_format"] = req["response_format"]
    # extra 透传（timeout, max_tokens 等）
    for k, v in (req.get("extra") or {}).items():
        kw[k] = v
    return kw


def _serialize_error(e: BaseException) -> dict:
    """把异常序列化为安全 dict（不含 API key、prompt 等）。"""
    status_code = getattr(e, "status_code", None)
    retry_after = None
    # litellm/openai 的 RateLimitError 有 retry_after 属性
    ra = getattr(e, "retry_after", None)
    if ra is not None:
        try:
            retry_after = float(ra)
        except (TypeError, ValueError):
            retry_after = None
    return {
        "class": type(e).__name__,
        "msg": str(e)[:500],
        "status_code": int(status_code) if status_code is not None else None,
        "retry_after": retry_after,
    }


if __name__ == "__main__":
    # spawn 入口：conn 通过 stdio handle 传递
    import multiprocessing as mp
    from multiprocessing.spawn import spawn_main
    spawn_main()
