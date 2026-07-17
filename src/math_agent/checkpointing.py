"""Beacon checkpoint saver 工厂：显式登记可反序列化的状态类型。"""
from __future__ import annotations

import inspect
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

import math_agent.state as state_module


def _allowed_state_types() -> tuple[type[BaseModel], ...]:
    """只允许 math_agent.state 中定义的 Pydantic checkpoint 类型。"""
    allowed = []
    for value in vars(state_module).values():
        if (inspect.isclass(value) and issubclass(value, BaseModel)
                and value.__module__ == state_module.__name__):
            allowed.append(value)
    return tuple(sorted(allowed, key=lambda item: item.__name__))


def checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_allowed_state_types())


@contextmanager
def sqlite_saver(path: str | Path):
    """创建启用严格类型 allowlist 的 SqliteSaver。"""
    with closing(sqlite3.connect(str(path), check_same_thread=False)) as conn:
        yield SqliteSaver(conn, serde=checkpoint_serializer())

