"""Semantic Scholar API 封装：按关键词检索真实文献。

- 免费 API，rate limit 100 req/s（无 key）/ 1 req/s（有 key 但非会员）
- 返回 Reference 列表，DOI/卷期号由 API 验证
- 网络错误 / rate limit 优雅降级返回空列表，writer 回退到静态库或写"未找到"
"""
from __future__ import annotations

import requests

from math_agent.state import Reference

_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "paperId,title,authors,year,venue,externalIds"


def search_references(query: str, *, limit: int = 10, timeout: int = 10) -> list[Reference]:
    """按关键词检索 Semantic Scholar，返回 Reference 列表。

    网络错误 / 非 200 响应（含 429 rate limit / 5xx）一律降级返回空列表，
    让上层 (references.select_references) 回退到静态库。
    """
    try:
        resp = requests.get(
            _API_URL,
            params={"query": query, "limit": limit, "fields": _FIELDS},
            timeout=timeout,
        )
    except (requests.RequestException, OSError):
        # RequestException 覆盖 requests 抛出的各类连接/超时错误；
        # OSError 兜住内置 ConnectionError（测试与部分环境会直接抛内置异常）。
        return []

    if resp.status_code != 200:
        return []  # 429 rate limit / 5xx 都降级

    try:
        payload = resp.json()
    except (ValueError, TypeError):
        return []
    if not isinstance(payload, dict):
        return []
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    refs: list[Reference] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        authors = item.get("authors") or []
        if not isinstance(authors, list):
            authors = []
        external_ids = item.get("externalIds") or {}
        if not isinstance(external_ids, dict):
            external_ids = {}
        refs.append(Reference(
            id=item.get("paperId") or "",
            title=item.get("title") or "",
            authors=[a.get("name") or "" for a in authors if isinstance(a, dict)],
            venue=item.get("venue", "") or "",
            year=item.get("year", 0) or 0,
            doi=external_ids.get("DOI") or "",
        ))
    return refs
