"""文本切块：先按 markdown 标题分节，再每节内走固定窗口 + 重叠。

接口刻意保持简单：不依赖 langchain 或 llama-index。
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    index: int  # 在该 source 内的顺序
    source_type: str = ""   # "paper" | "model_lib" | ""
    section: str = ""       # 最近的 ##/### 标题文本，无标题则 ""


_PREFERRED_BREAKS = ("\n\n", "。", "\n", "，", " ")
# ponytail: PDF 标题识别是启发式——靠"第X章/§N/N.N + 后接中文/字母"模式匹配，
# 不保证覆盖所有 PDF 格式，但对 CUMCM 教材足够；漏识别的标题退化为无 section，
# 不影响检索。N.N 分支要求后接中文/字母，滤掉 "9.4    1" 这类数据行。
# 分隔符用 [ \t]+ 而非 \s+，防止跨行把 "0.5 \nand..." 误并成标题。
_HEADING = re.compile(
    r"^(?:"
    r"(?:#{1,6})[ \t]+(.+)"                                   # markdown ## / ###
    r"|第[一二三四五六七八九十百零]+章[ \t]+(.+)"             # 第六章 / 第二十五章
    r"|§\s*\d+[ \t]+(.+)"                                     # §1 / § 3 / §10
    r"|\d+(?:\.\d+){1,2}[ \t]+([\u4e00-\u9fff].*)"          # 1.1 标题（后接中文，滤数据行与英文正文）
    r")\s*$",
    re.MULTILINE,
)


def _heading_text(m: re.Match) -> str:
    """从多分支匹配里取第一个非 None 捕获组（即标题文本）。"""
    for g in m.groups():
        if g is not None:
            return g.strip()
    return ""


def _best_break(s: str, near: int) -> int:
    """在 [near*0.7, near] 范围里找首选边界，找不到就返回 near。"""
    lo = max(0, int(near * 0.7))
    for sep in _PREFERRED_BREAKS:
        idx = s.rfind(sep, lo, near)
        if idx != -1:
            return idx + len(sep)
    return near


def _split_by_headings(text: str):
    """按 markdown/PDF 标题切成 (section, body) 序列；标题前的前言 section=''。"""
    positions = list(_HEADING.finditer(text))
    if not positions:
        yield ("", text)
        return
    if positions[0].start() > 0:
        pre = text[: positions[0].start()].strip()
        if pre:
            yield ("", pre)
    for i, m in enumerate(positions):
        start = m.end()
        end = positions[i + 1].start() if i + 1 < len(positions) else len(text)
        body = text[start:end].strip()
        if body:
            yield (_heading_text(m), body)


def _window(text: str, *, max_chars: int, overlap: int) -> list[str]:
    """单段文本内的固定窗口 + 重叠切（原 chunk_text 的循环逻辑）。"""
    out: list[str] = []
    n = len(text)
    if n == 0:
        return out
    i = 0
    while i < n:
        end = min(i + max_chars, n)
        if end < n:
            end = i + _best_break(text[i:end], max_chars)
        out.append(text[i:end])
        if end >= n:
            break
        i = max(end - overlap, i + 1)  # 防止 overlap 导致原地踏步
    return out


def chunk_text(
    text: str, *, max_chars: int, overlap: int, source: str,
    source_type: str = "",
) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")

    chunks: list[Chunk] = []
    n = len(text)
    if n == 0:
        return chunks

    idx = 0
    for section, body in _split_by_headings(text):
        for piece in _window(body, max_chars=max_chars, overlap=overlap):
            chunks.append(Chunk(
                text=piece, source=source, index=idx,
                source_type=source_type, section=section,
            ))
            idx += 1
    return chunks
