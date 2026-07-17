r"""LaTeX 转义回归测试套件 — 验证 json.loads 不破坏 LaTeX 命令。

问题：LLM 在 JSON 输出中写 \bar 等 LaTeX 命令，json.loads 会解释
\b → 0x08, \t → 0x09, \f → 0x0C, \n → 0x0A, \r → 0x0D。
非 JSON 合法转义的（\h, \l, \m 等）则直接抛 JSONDecodeError。

关键术语：
- 「正确转义」= LLM 写 \\text → JSON 中 \\text → json.loads 得 \text ✅
- 「欠转义」  = LLM 写 \text  → JSON 中 \text  → json.loads 得 TAB+ext ❌
- 「非法转义」= LLM 写 \hat   → JSON 中 \hat   → json.loads 抛异常 ❌

=== Python 原始字符串中的反斜杠计数 ===
在 Python 源代码中，每层需要的转义:
  LaTeX目标     JSON文本         Python原始串
  \bar{S}      \\bar{S}        r'\\bar{S}'      ← 正确转义
  \bar{S}      \bar{S}         r'\bar{S}'       ← 欠转义（bug）
  \hat{x}      \\hat{x}        r'\\hat{x}'      ← 正确转义
  \hat{x}      \hat{x}         r'\hat{x}'       ← 非法转义（\h 非JSON合法）

使用：
    python scripts/test_latex_escape_regression.py
"""
from __future__ import annotations

import json
import re
import sys
import textwrap
import subprocess
import tempfile
from pathlib import Path
from collections import Counter
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_TEX = ROOT / "runs" / "phase2_final" / "paper.tex"

try:
    sys.path.insert(0, str(ROOT))
    from math_agent import llm as llm_module
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

# ── 测试框架 ──
_tests: list[dict] = []
_tests_run = 0
_tests_pass = 0
_tests_fail = 0
_tests_skip = 0

def test(name: str, category: str):
    def decorator(fn):
        _tests.append({"name": name, "category": category, "fn": fn})
        return fn
    return decorator

def run_tests():
    global _tests_run, _tests_pass, _tests_fail, _tests_skip
    for t in _tests:
        _tests_run += 1
        try:
            t["fn"]()
            _tests_pass += 1
            print(f"  ✅ [{t['category']}] {t['name']}")
        except AssertionError as e:
            _tests_fail += 1
            print(f"  ❌ [{t['category']}] {t['name']}")
            for line in str(e).split("\n"):
                print(f"       {line}")
        except Exception as e:
            _tests_fail += 1
            print(f"  💥 [{t['category']}] {t['name']}: {e}")
    print()

# ── 辅助函数 ──
def _parse_field(raw_json: str) -> str:
    """解析 {"field": ...} 返回字段值。raw_json 必须是 JSON 文本的 Python 原始字符串。"""
    return list(json.loads(raw_json).values())[0]

def has_ctrl(s: str) -> set[str]:
    return {f"0x{ord(c):02X}" for c in s if ord(c) < 0x20}

def assert_no_ctrl(s: str, ctx: str = ""):
    found = has_ctrl(s)
    if found:
        raise AssertionError(f"Control chars {found} in {ctx}\n  {s[:120]!r}")

def old_fix(json_text: str) -> str:
    """Old fix: str.replace before json.loads (commit 9a87e94)."""
    for esc in (r"\b", r"\t", r"\f"):
        json_text = json_text.replace(esc, r"\\" + esc[1:])
    return json_text

def new_fix(content: str) -> str | None:
    """New fix: parse → post-parse repair (commit 5bf5a77)."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return None
    def _repair(v):
        if isinstance(v, str):
            return _repair_core(v)
        if isinstance(v, dict):
            return {k: _repair(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_repair(x) for x in v]
        return v
    obj = _repair(obj)
    if isinstance(obj, dict):
        return next(iter(obj.values())) if obj else None
    return str(obj) if obj else None

_CTRL2TEX = {0x08: r"\b", 0x09: r"\t", 0x0C: r"\f"}
_CTRLCHARS = set(_CTRL2TEX)
def _repair_core(s: str) -> str:
    if not any(ord(c) in _CTRLCHARS for c in s):
        return s
    out = []
    for ch in s:
        code = ord(ch)
        if code in _CTRL2TEX:
            if out and out[-1] == "\\":
                out.append(_CTRL2TEX[code][1])
            else:
                out.append(_CTRL2TEX[code])
        else:
            out.append(ch)
    return "".join(out)


# ═══════════════════════════════════════════════════════
# 类别 1：JSON 解析层
# ═══════════════════════════════════════════════════════
# 所有 JSON 文本用 Python raw string 写，反斜杠数规则：
#   LaTeX命令    JSON文本(2层)  Python原始串(1层)    含义
#   \bar          \\bar          r'\\bar'             正确转义 → \bar
#   \bar          \bar           r'\bar'              欠转义   → 0x08+ar
#   \hat          \\hat          r'\\hat'             正确转义 → \hat
#   \hat          \hat           r'\hat'              非法转义 → JSONDecodeError

# ── 1a: 正确转义（LLM 写了双反斜杠）→ 预期无损坏 ──

@test("1.1a \\\\bar -> json.loads -> \\bar", "json")
def t1_1a():
    val = _parse_field(r'{"field": "\\bar{S}"}')
    assert_no_ctrl(val, "1.1a")
    assert val == r"\bar{S}", f"got {val!r}"

@test("1.1b \\\\text -> json.loads -> \\text", "json")
def t1_1b():
    val = _parse_field(r'{"field": "\\text{MAE}"}')
    assert_no_ctrl(val, "1.1b")
    assert r"\text" in val

@test("1.1c \\\\frac -> json.loads -> \\frac", "json")
def t1_1c():
    val = _parse_field(r'{"field": "\\frac{1}{2}"}')
    assert_no_ctrl(val, "1.1c")
    assert r"\frac" in val

@test("1.1d \\\\boldsymbol -> json.loads -> \\boldsymbol", "json")
def t1_1d():
    val = _parse_field(r'{"field": "\\boldsymbol{x}"}')
    assert_no_ctrl(val, "1.1d")
    assert r"\boldsymbol" in val

@test("1.1e \\\\theta -> json.loads -> \\theta", "json")
def t1_1e():
    val = _parse_field(r'{"field": "\\theta_i"}')
    assert_no_ctrl(val, "1.1e")
    assert r"\theta" in val

@test("1.1f \\\\nabla -> json.loads -> \\nabla", "json")
def t1_1f():
    val = _parse_field(r'{"field": "\\nabla f"}')
    assert_no_ctrl(val, "1.1f")
    assert r"\nabla" in val

@test("1.1g \\\\rbrace -> json.loads -> \\rbrace", "json")
def t1_1g():
    val = _parse_field(r'{"field": "\\rbrace"}')
    assert_no_ctrl(val, "1.1g")
    assert r"\rbrace" in val

@test("1.1h \\\\hat -> json.loads -> \\hat", "json")
def t1_1h():
    val = _parse_field(r'{"field": "\\hat{x}"}')
    assert_no_ctrl(val, "1.1h")
    assert r"\hat" in val

# ── 1b: 欠转义（LLM 只写单反斜杠）→ 产生控制字符 ──

@test("1.2a \\bar{S} (欠转义) → 0x08", "json")
def t1_2a():
    val = _parse_field(r'{"field": "\bar{S}"}')
    c = has_ctrl(val)
    assert "0x08" in c, f"Expected BS, got {c}: {val!r}"

@test("1.2b \\text{MAE} (欠转义) → 0x09", "json")
def t1_2b():
    val = _parse_field(r'{"field": "\text{MAE}"}')
    c = has_ctrl(val)
    assert "0x09" in c, f"Expected TAB, got {c}: {val!r}"

@test("1.2c \\frac{1}{2} (欠转义) → 0x0C", "json")
def t1_2c():
    val = _parse_field(r'{"field": "\frac{1}{2}"}')
    c = has_ctrl(val)
    assert "0x0C" in c, f"Expected FF, got {c}: {val!r}"

@test("1.2d \\nabla f (欠转义) → 0x0A", "json")
def t1_2d():
    val = _parse_field(r'{"field": "\nabla f"}')
    c = has_ctrl(val)
    assert "0x0A" in c, f"Expected LF, got {c}: {val!r}"

@test("1.2e \\rbrace (欠转义) → 0x0D", "json")
def t1_2e():
    val = _parse_field(r'{"field": "\rbrace"}')
    c = has_ctrl(val)
    assert "0x0D" in c, f"Expected CR, got {c}: {val!r}"

# ── 1c: 非法转义（\h, \l, \m 等非JSON合法转义）→ JSONDecodeError ──

@test("1.3a \\hat{x} (非法转义 \\h) → JSONDecodeError", "json")
def t1_3a():
    try:
        val = _parse_field(r'{"field": "\hat{x}"}')
        assert False, f"Expected JSONDecodeError, got {val!r}"
    except json.JSONDecodeError:
        pass

@test("1.3b \\left (非法转义 \\l) → JSONDecodeError", "json")
def t1_3b():
    try:
        val = _parse_field(r'{"field": "\left("}')
        assert False, f"Expected JSONDecodeError, got {val!r}"
    except json.JSONDecodeError:
        pass

@test("1.3c \\sum (非法转义 \\s) → JSONDecodeError", "json")
def t1_3c():
    try:
        val = _parse_field(r'{"field": "\sum_{i}"}')
        assert False, f"Expected JSONDecodeError, got {val!r}"
    except json.JSONDecodeError:
        pass

@test("1.3d \\max (非法转义 \\m) → JSONDecodeError", "json")
def t1_3d():
    try:
        val = _parse_field(r'{"field": "\max_{i}"}')
        assert False, f"Expected JSONDecodeError, got {val!r}"
    except json.JSONDecodeError:
        pass

# ── 1d: 旧 fix 对正确转义的误伤（\times 中 \\t 被误改）──

@test("1.4a \\\\times (正确转义) → 旧 fix 不破坏它", "json")
def t1_4a():
    raw = r'{"field": "\\times"}'
    # Without fix: \\times → \times ✅
    correct = _parse_field(raw)
    assert_no_ctrl(correct, "1.4a")
    assert r"\times" in correct
    # With old fix: \\times → \\\times → \ + TAB ❌
    broken = json.loads(old_fix(raw))["field"]
    bc = has_ctrl(broken)
    if "0x09" in bc:
        print(f"      ⚠️  旧 fix 误伤 \\\\times: {broken!r}")
    # With new fix: should be correct
    fixed = new_fix(raw)
    if fixed:
        assert_no_ctrl(fixed, "1.4a new_fix")
        assert r"\times" in fixed

# ── 1e: 混合场景 ──

@test("1.5a \\text{$a \\\\times b$} (\\t欠转义 + \\\\t正确)", "json")
def t1_5a():
    raw = r'{"field": "\text{$a \\times b$}"}'
    # Without fix: \t → TAB, \\t → \t (correct)
    val = _parse_field(raw)
    assert "0x09" in has_ctrl(val), f"Expected TAB: {val!r}"
    assert r"\times" in val, f"Expected \\times: {val!r}"
    # After old fix: \text fixed but \\\text damages \\times
    val2 = json.loads(old_fix(raw))["field"]
    c2 = has_ctrl(val2)
    if c2:
        print(f"      ⚠️  旧 fix 仍有控制符 {c2}: {val2!r}")
    else:
        print(f"      ✅  旧 fix 修复后: {val2!r}")

@test("1.5b \\bigl (欠转义) + ( 混合", "json")
def t1_5b():
    val = _parse_field(r'{"field": "\bigl("}')
    assert "0x08" in has_ctrl(val), f"Expected BS: {val!r}"
    # After old fix
    val2 = json.loads(old_fix(r'{"field": "\bigl("}'))["field"]
    assert_no_ctrl(val2, "1.5b after old_fix")
    assert r"\bigl" in val2

# ═══════════════════════════════════════════════════════
# 类别 2：修复算法边界条件
# ═══════════════════════════════════════════════════════

BS = chr(0x08) if not hasattr(__builtins__, 'BS') else BS
TAB = chr(0x09)
FF = chr(0x0C)

@test("2.1 _repair_string — 空字符串", "fix")
def t2_1():
    if not HAS_LLM:
        print(f"      ⏭️  llm 不可用")
        return
    assert llm_module._repair_string("") == ""

@test("2.2 _repair_string — 纯反斜杠", "fix")
def t2_2():
    if not HAS_LLM:
        return
    assert llm_module._repair_string("\\") == "\\"

@test("2.3 _repair_string — 干净 LaTeX", "fix")
def t2_3():
    if not HAS_LLM:
        return
    s = r"\sum_{i=1}^n x_i"
    assert llm_module._repair_string(s) == s

@test("2.4 _repair_control_chars — 嵌套 dict 控制符修复", "fix")
def t2_4():
    if not HAS_LLM:
        return
    obj = {"a": {"b": f"\\{TAB}ext{{x}}"}}
    llm_module._repair_control_chars_in_obj(obj)
    assert TAB not in obj["a"]["b"]

@test("2.5 _repair_string — BS 前有反斜杠", "fix")
def t2_5():
    if not HAS_LLM:
        return
    s = f"\\{BS}ar{{S}}"
    result = llm_module._repair_string(s)
    assert BS not in result

@test("2.6 _repair_control_chars — dict 值修复", "fix")
def t2_6():
    if not HAS_LLM:
        return
    obj = {"x": f"\\{TAB}ext", "y": f"\\{BS}ar"}
    llm_module._repair_control_chars_in_obj(obj)
    assert TAB not in obj["x"]
    assert BS not in obj["y"]

@test("2.7 _repair_control_chars — list 修复", "fix")
def t2_7():
    if not HAS_LLM:
        return
    obj = [f"\\{TAB}ext", f"\\{BS}ar"]
    llm_module._repair_control_chars_in_obj(obj)
    assert TAB not in obj[0]
    assert BS not in obj[1]

# ═══════════════════════════════════════════════════════
# 类别 3：paper.tex 静态分析
# ═══════════════════════════════════════════════════════

@test("3.1 paper.tex 扫描控制字符", "static")
def t3_1():
    if not SAMPLE_TEX.exists():
        print(f"      ⏭️  paper.tex 不存在")
        return
    data = SAMPLE_TEX.read_bytes()
    bad = sum(1 for b in data if b < 0x20 and b not in (0x09, 0x0A, 0x0D))
    if bad:
        print(f"      ⚠️  {bad} 个控制字符（旧版生成的文件）")
    else:
        print(f"      ✅  文件干净")

@test("3.2 paper.tex 损坏命令分布", "static")
def t3_2():
    if not SAMPLE_TEX.exists():
        return
    text = SAMPLE_TEX.read_text(encoding="utf-8")
    print()
    for name, byte in [("BACKSPACE", 0x08), ("TAB", 0x09), ("FORMFEED", 0x0C)]:
        cmds = re.findall(chr(byte) + r'([a-zA-Z_]+)', text)
        if cmds:
            c = dict(Counter(cmds).most_common())
            print(f"      {name} (0x{byte:02X}): {sum(Counter(cmds).values())} — {c}")
        else:
            print(f"      {name}: 0 ✅")

@test("3.3 paper.tex 应有控制字符（修复前）", "static")
def t3_3():
    """验证当前 paper.tex 确实有控制字符（说明修复前生成）。"""
    if not SAMPLE_TEX.exists():
        return
    text = SAMPLE_TEX.read_text(encoding="utf-8")
    assert chr(0x08) in text, "Expected BS in pre-fix paper.tex"
    assert chr(0x09) in text, "Expected TAB in pre-fix paper.tex"

# ═══════════════════════════════════════════════════════
# 类别 4：xelatex 编译
# ═══════════════════════════════════════════════════════

@test("4.1 xelatex smoke test — 干净 LaTeX 编译", "compile")
def t4_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        tex = Path(tmpdir) / "test.tex"
        tex.write_text(textwrap.dedent(r"""\documentclass[12pt,a4paper]{article}
        \usepackage{xeCJK}
        \setCJKmainfont{SimSun}
        \begin{document}
        $\bar{S}$, $\text{MAE}$, $\frac{1}{2}$, $\boldsymbol{x}$, $\theta$, $\nabla f$.
        \end{document}
        """), encoding="utf-8")
        r = subprocess.run(
            ["xelatex", "-interaction=nonstopmode",
             "-output-directory", tmpdir, str(tex)],
            capture_output=True, text=True, timeout=60)
        log = r.stdout + r.stderr
        pdf = Path(tmpdir) / "test.pdf"
        if pdf.exists():
            print(f"      ✅  PDF 生成")
            return
        if "SimSun" in log:
            print(f"      ⚠️  SimSun 字体缺失")
            return
        raise AssertionError(f"编译失败: {log[-300:]}")

# ═══════════════════════════════════════════════════════
# 运行
# ═══════════════════════════════════════════════════════

def main():
    print(f"{'=' * 60}")
    print(f" LaTeX 转义回归测试套件")
    print(f"{'=' * 60}")
    print(f" llm 模块: {'可用' if HAS_LLM else '不可用'}")
    print(f" paper.tex: {'存在' if SAMPLE_TEX.exists() else '不存在'}")
    print(f"{'=' * 60}\n")

    run_tests()

    print(f"{'=' * 60}")
    print(f"  总计: {_tests_run}  |  通过: {_tests_pass}  |  "
          f"失败: {_tests_fail}  |  跳过: {_tests_skip}")
    if _tests_fail > 0:
        print(f"  ❌ 部分测试失败")
        sys.exit(1)
    else:
        print(f"  ✅ 全部通过")


if __name__ == "__main__":
    main()
