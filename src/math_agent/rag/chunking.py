"""文本切块：固定窗口 + 重叠，优先在段落 / 句子边界切。

接口刻意保持简单：不依赖 langchain 或 llama-index。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    index: int  # 在该 source 内的顺序


_PREFERRED_BREAKS = ("\n\n", "。", "\n", "，", " ")


def _best_break(s: str, near: int) -> int:
    """在 [near*0.7, near] 范围里找首选边界，找不到就返回 near。"""
    lo = max(0, int(near * 0.7))
    for sep in _PREFERRED_BREAKS:
        idx = s.rfind(sep, lo, near)
        if idx != -1:
            return idx + len(sep)
    return near


def chunk_text(text: str, *, max_chars: int, overlap: int, source: str) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")

    chunks: list[Chunk] = []
    n = len(text)
    if n == 0:
        return chunks

    i = 0
    idx = 0
    while i < n:
        end = min(i + max_chars, n)
        if end < n:
            end = i + _best_break(text[i:end], max_chars)
        chunks.append(Chunk(text=text[i:end], source=source, index=idx))
        idx += 1
        if end >= n:
            break
        i = max(end - overlap, i + 1)  # 防止 overlap 导致原地踏步
    return chunks
