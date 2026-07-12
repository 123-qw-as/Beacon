"""统一检索入口；节点只 import 这一个函数。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from math_agent.rag.embeddings import embed_texts
from math_agent.rag.store import VectorStore


@dataclass
class Snippet:
    text: str
    source: str
    score: float
    source_type: str = ""
    section: str = ""


def search(
    query: str,
    *,
    db_path: str | Path,
    k: int = 5,
    embedding_model: str = "text-embedding-3-small",
    dim: int = 1536,
    source_type: str | None = None,
) -> list[Snippet]:
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    store = VectorStore.open(db_path, dim=dim)
    try:
        # 先打开并校验库维度，再付费调用 embedding；配置错误不应消耗 API。
        vec = embed_texts([query], model=embedding_model)[0]
        rows = store.search(vec, k=k, source_type=source_type)
        # 过滤无结果 → 退回全库 top-k，避免 writer 要论文却只有 model_lib 时返空。
        if not rows and source_type is not None:
            rows = store.search(vec, k=k)
    finally:
        store.close()
    return [Snippet(text=r.text, source=r.source, score=r.score,
                    source_type=r.source_type, section=r.section) for r in rows]


def format_snippets(snippets: list[Snippet], *, max_chars: int | None = None) -> str:
    """供 prompt 拼接的统一格式。

    max_chars: 若给出则按 snippet 块边界截断（不切中段），整段输出（含 header）
    控制在 max_chars 附近。单块本身超长才对该块逐字符截并标注。
    """
    if not snippets:
        return ""
    header = "# 检索到的参考资料（仅供启发，不可照抄）"

    if max_chars is None:
        parts = [header]
        for i, s in enumerate(snippets, 1):
            parts.append(f"## [{i}] 来源：{s.source}\n{s.text}")
        return "\n\n".join(parts)

    # 按块累积：每加一块前预估总长，超 max_chars 就停，尾部加截断标记。
    # ponytail: 单条超长才逐字符截，正常块永不切中段（保护来源引用可读性）。
    suffix = "\n...（已截断，共 {total} 条，显示 {shown} 条）"
    blocks: list[str] = []
    shown = 0
    total = len(snippets)
    # header 始终保留；预算从 header 之后算
    for i, s in enumerate(snippets, 1):
        block = f"## [{i}] 来源：{s.source}\n{s.text}"
        sep = "\n\n" if blocks else ""
        tentative = sep + block
        # 预估：当前累积 + 本块 + 最坏截断标记长度
        worst_suffix = suffix.format(total=total, shown=i)
        if len(header) + len("".join(blocks)) + len(tentative) + len(worst_suffix) <= max_chars:
            blocks.append(tentative)
            shown = i
        else:
            break

    if shown == 0 and snippets:
        # 第一块就超预算：对该单块逐字符截，保留 header + 来源 + 截断标记
        s = snippets[0]
        single_suffix = "\n...（单条过长已截断）"
        budget = max_chars - len(header) - len(single_suffix) - len("\n\n## [1] 来源：") - len(s.source)
        if budget < 10:
            budget = 10   # 至少留一点正文
        body = s.text[:budget]
        return f"{header}\n\n## [1] 来源：{s.source}\n{body}{single_suffix}"

    out = header + "".join(blocks)
    if shown < total:
        out += suffix.format(total=total, shown=shown)
    return out
