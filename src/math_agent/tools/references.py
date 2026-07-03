"""参考文献选择：优先 Semantic Scholar API，网络失败降级到静态库。"""
from __future__ import annotations

import json
from pathlib import Path

from math_agent.state import Reference
from math_agent.tools.scholar import search_references

_REF_DIR = Path(__file__).resolve().parent.parent / "references"
_LIB_PATH = _REF_DIR / "builtin_library.json"


def select_references(problem: str, problem_domains: list[str], *, limit: int = 10) -> list[Reference]:
    """优先 Semantic Scholar API；网络失败降级到静态库。"""
    refs = search_references(problem[:200], limit=limit)
    if refs:
        return refs
    # fallback：从静态库按 domain 交集筛选
    return _from_static_library(problem_domains, limit=limit)


def _from_static_library(problem_domains: list[str], *, limit: int = 10) -> list[Reference]:
    if not _LIB_PATH.exists():
        return []
    lib = json.loads(_LIB_PATH.read_text(encoding="utf-8"))
    candidates = [Reference(**r) for r in lib
                  if set(r.get("domains", [])) & set(problem_domains)]
    if not candidates:
        candidates = [Reference(**r) for r in lib if "generic" in r.get("domains", [])]
    return sorted(candidates, key=lambda r: -r.year)[:limit]
