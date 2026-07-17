"""共享渲染工具函数。

latex_node 和 writer.render_markdown 共用：代码/stdout 截取、路径转义、
图注截断、纯文本转义。独立于 markdown→LaTeX 转换链（latex_transform.py），
打破 latex↔writer 的双向导入。
"""
from __future__ import annotations

import re


# ---- 纯文本转义 ----

_PLAIN_TEXT_ESCAPES = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
    "%": r"\%", "#": r"\#", "&": r"\&", "$": r"\$",
    "_": r"\_", "^": r"\^{}", "~": r"\~{}",
}

_ESCAPED_LAYOUT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\\r\\n"), "\n"),
    (re.compile(r"\\n\\n"), "\n\n"),
    (re.compile(r"\\n(?=(?:\\|[ \t]*[-*#]|\d+\.)|$)"), "\n"),
    (re.compile(r"\\t(?=(?:\\|[ \t]*[-*#]|\d+\.)|$)"), "\t"),
]


def _latex_plain_text(s: str | None) -> str | None:
    """完整转义封面元数据等不允许包含 LaTeX 数学的纯文本。"""
    if s is None:
        return None
    return "".join(_PLAIN_TEXT_ESCAPES.get(ch, ch) for ch in s)


def _normalize_escaped_layout_text(s: str | None) -> str | None:
    """把误写成字面量的 \\n / \\t 恢复为排版空白，避免污染 markdown/latex 渲染。"""
    if s is None or "\\" not in s:
        return s
    out = s
    prev = None
    while out != prev:
        prev = out
        for pattern, repl in _ESCAPED_LAYOUT_PATTERNS:
            out = pattern.sub(repl, out)
    return out


# ---- 路径转义 ----

def _latex_path(p: str) -> str:
    """把 Windows 路径包成 LaTeX 可读形式：正斜杠 + \\detokenize 阻止解释 _ 等。"""
    # 右花括号会结束 \detokenize 参数，需关闭参数、输出字符 125 后再开启。
    safe = p.replace("\\", "/").replace("}", r"}\char125\detokenize{")
    return r"\detokenize{" + safe + "}"


# ---- 图注截断 ----

def _truncate_caption(s: str, *, max_chars: int = 55) -> str:
    """把长图注截到 max_chars 以内，但优先切在完整句/短语边界。

    LLM 写的图 caption 常常两三个句子；直接 `s[:55]` 会切在逗号/单字上。
    策略：先看 max_chars 处是否已是终结符；否则在整个可保留前缀中找最靠后的
    句末字符；没有才退到后 40% 区间内最靠后的逗号；再退不到就硬截。

    句末搜索不能只看后 40%：真实图注的第一句常很短，第二句包含小数，旧逻辑会
    忽略前一句句号并把 ``44.0%`` 或 ``0.97`` 截成 ``44.`` / ``0.``。
    """
    if not s or len(s) <= max_chars:
        return s
    hard_end = s[max_chars - 1]
    hard_dot_is_decimal = (
        hard_end == "."
        and max_chars >= 2
        and max_chars < len(s)
        and s[max_chars - 2].isdigit()
        and s[max_chars].isdigit()
    )
    if hard_end in "。！？；!?" or (hard_end == "." and not hard_dot_is_decimal):
        return s[:max_chars]
    prefix = s[:max_chars]
    sentence_ends = [
        match.end() - 1
        for match in re.finditer(r"[。！？；!?]|(?<!\d)\.|\.(?!\d)", prefix)
    ]
    sentence_idx = max(sentence_ends, default=-1)
    if sentence_idx >= 4:
        return s[: sentence_idx + 1]
    lo = max(1, int(max_chars * 0.6))
    phrase = s[lo:max_chars]
    phrase_idx = max((phrase.rfind(c) for c in "，、,"), default=-1)
    if phrase_idx != -1:
        candidate = s[: lo + phrase_idx].rstrip()
        return candidate + ("。" if any("\u4e00" <= ch <= "\u9fff" for ch in candidate) else ".")
    return s[:max_chars]


# ---- 代码/stdout 截取 ----

def _curate_code(code: str, max_lines: int = 80) -> str:
    """截取代码前 max_lines 行。"""
    lines = code.split("\n")
    if len(lines) <= max_lines:
        return code
    return "\n".join(lines[:max_lines]) + f"\n# ... (共 {len(lines)} 行，截取前 {max_lines} 行)"


def _curate_stdout(stdout: str) -> str:
    """提取 stdout 关键行：RESULT: 行 + 末尾 5 行。"""
    if not stdout:
        return ""
    lines = stdout.splitlines()
    result_lines = [l for l in lines if l.strip().startswith("RESULT:")]
    tail = lines[-5:]
    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for l in result_lines + tail:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return "\n".join(out)
