"""输出目录级单 worker 锁；进程退出时由操作系统自动释放。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import BinaryIO


class RunLockedError(RuntimeError):
    pass


class RunLock:
    def __init__(self, out: str | Path, *, filename: str = ".beacon-worker.lock"):
        self.out = Path(out)
        self.path = self.out / filename
        self._handle: BinaryIO | None = None

    def acquire(self) -> "RunLock":
        if self._handle is not None:
            return self
        self.out.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError) as exc:
            handle.close()
            raise RunLockedError(f"输出目录已有运行中的 worker: {self.out}") from exc

        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid()}, ensure_ascii=False).encode("utf-8"))
        handle.flush()
        self._handle = handle
        return self

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
