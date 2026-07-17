# Fix `run_python` Subprocess Hang

> **历史方案记录（非现行事实源）**：路径、状态、命令和待办保留当时上下文；当前实现与限制见 [`../../README.md`](../../README.md)。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate `run_python` subprocess hangs caused by LLM-generated `plt.show()` reaching the subprocess without the `Agg` backend, by fixing the injection logic bug, adding code sanitization, and hardening prompts.

**Architecture:** Three-layer defense: (1) Prompts tell LLM not to emit `plt.show()`/`plt.ion()`/`fig.show()` and to use Agg. (2) `_auto_fix_imports` reliably injects `matplotlib.use('Agg')` regardless of whether the LLM already imported matplotlib, while still injecting pyplot import if missing. (3) A sanitization pass strips any `plt.show()`/`plt.ion()`/`fig.show()` calls that slip through. Per-attempt artifact persistence and progress logging are added to prevent data loss and improve observability.

**Tech Stack:** Python 3.11+, `subprocess`, `re`, existing `pytest` suite in `tests/test_runner.py` and `tests/nodes/test_coder.py`.

---

### Task 1: Fix `_auto_fix_imports` — change `already_marker` from "import matplotlib.pyplot" to "matplotlib.use('Agg')"

**Files:**
- Modify: `src/math_agent/tools/runner.py:49-58`

- [ ] **Step 1: Rewrite `_IMPORT_FIXES` table**

The root cause: when LLM code has `import matplotlib.pyplot as plt` but no `matplotlib.use('Agg')`, the `already_marker = "import matplotlib.pyplot"` matches the substring in `"import matplotlib.pyplot as plt"` → the entire fix_line (including `matplotlib.use('Agg')`) is skipped.

Fix: change `already_marker` for the matplotlib entries from import detection to Agg detection. The fix_line should inject `import matplotlib; matplotlib.use('Agg')` — the `import matplotlib` is idempotent, and `matplotlib.use('Agg')` runs before any downstream `import matplotlib.pyplot`.

Replace lines 51-57:

```python
_IMPORT_FIXES: list[tuple[str, str, str]] = [
    # (检测标志, 缺时补的行, already_marker — 若已在代码中存在则跳过)
    #
    # matplotlib Agg 后端：用 "matplotlib.use('Agg')" 做 already_marker，
    # 而不是 "import matplotlib.pyplot"——旧逻辑会因为 LLM 已 import pyplot
    # 就跳过 Agg 注入，导致 plt.show() 在子进程中尝试弹 GUI 窗口阻塞。
    # 注意：条目 2、3 的 fix_line 仍保留 import matplotlib.pyplot，
    # 防止 LLM 只写了 plt.plot() 却忘了 import pyplot 的遗漏场景。
    # import 语句是幂等的，重复注入无害。
    ("matplotlib.rcParams", "import matplotlib; matplotlib.use('Agg')", "matplotlib.use('Agg')"),
    ("matplotlib\\.pyplot", "import matplotlib; matplotlib.use('Agg')\nimport matplotlib.pyplot", "matplotlib.use('Agg')"),
    ("plt\\.", "import matplotlib; matplotlib.use('Agg')\nimport matplotlib.pyplot as plt", "matplotlib.use('Agg')"),
    # 常规 import（保持原逻辑不变）
    ("np\\.", "import numpy as np", "import numpy"),
    ("pd\\.", "import pandas as pd", "import pandas"),
]
```

- [ ] **Step 2: Run existing tests to confirm no regression**

```
pytest tests/test_runner.py -v
```
Expected: all 15 existing tests PASS. The `test_auto_fix_imports_handles_qualified_pyplot_after_matplotlib_import` test injects `import matplotlib.pyplot` into code that has `import matplotlib` — this test should still pass because the code has no `matplotlib.use('Agg')` so the new logic still triggers injection.

- [ ] **Step 3: Commit**

```
git add src/math_agent/tools/runner.py
git commit -m "fix(runner): change already_marker to check for Agg, not pyplot import

Old behavior: if LLM code had 'import matplotlib.pyplot', _auto_fix_imports
skipped injecting matplotlib.use('Agg') because already_marker='import
matplotlib.pyplot' matched. This left plt.show() active, causing subprocess
hangs on Windows GUI backends.

New behavior: already_marker='matplotlib.use(\"Agg\")' — Agg is injected
whenever matplotlib is used but Agg isn't already set, regardless of whether
pyplot was imported. Pyplot import injection is retained for entries 2 and 3
to handle LLM omissions of 'import matplotlib.pyplot'."
```

---

### Task 2: Add `_strip_dangerous_calls` sanitization in `run_python`

**Files:**
- Modify: `src/math_agent/tools/runner.py:61-74` (inside `_auto_fix_imports` or as a new function, and call it from `run_python`)

- [ ] **Step 1: Add sanitization regex and function**

Defense-in-depth: even with fixed Agg injection, strip `plt.show()`/`plt.ion()` calls so no blocking GUI call ever reaches the subprocess.

Add after `_IMPORT_FIXES` (after line 58), before `_auto_fix_imports`:

```python
# 危险调用：在非交互子进程中会弹 GUI 窗口阻塞。Agg 后端让它们变 no-op，
# 但显式移除更安全——万一 matplotlib 内部在 Agg 下仍走 event loop 呢。
# 覆盖 plt.show() / plt.ion() / fig.show()（fig 是 Figure 对象）。
_DANGEROUS_CALL_RE = _re.compile(
    r'^\s*(?:plt|fig)\.(?:show|ion)\s*\(.*\)\s*(?:#.*)?$',
    _re.MULTILINE,
)


def _strip_dangerous_calls(code: str) -> str:
    """移除 plt.show() / plt.ion() / fig.show() 等会阻塞子进程的 GUI 调用。"""
    return _DANGEROUS_CALL_RE.sub(
        '# [auto-removed: plt.show()/ion()/fig.show() blocks subprocess]',
        code,
    )
```

- [ ] **Step 2: Call `_strip_dangerous_calls` in `run_python`**

In `run_python`, line 82, add the call right after `_auto_fix_imports`:

Replace line 82:
```python
    code = _auto_fix_imports(code)   # 补缺的 import，避免 NameError 白跑
```

With:
```python
    code = _auto_fix_imports(code)   # 补缺的 import + matplotlib.use('Agg')
    code = _strip_dangerous_calls(code)  # 移除 plt.show()/ion()，防止 GUI 阻塞
```

- [ ] **Step 3: Run tests**

```
pytest tests/test_runner.py -v
```
Expected: all existing tests PASS.

- [ ] **Step 4: Commit**

```
git add src/math_agent/tools/runner.py
git commit -m "feat(runner): add _strip_dangerous_calls to remove plt.show()/ion()

Strips plt.show(), plt.ion(), and fig.show() calls from LLM-generated code before
subprocess execution. Even with Agg backend (Task 1), explicit stripping
provides defense-in-depth against GUI event loop hangs."
```

---

### Task 3: Add unit tests for the fix

**Files:**
- Modify: `tests/test_runner.py` (append new tests after line 129)

- [ ] **Step 1: Write tests for Agg injection logic**

Append to `tests/test_runner.py`:

```python
# ── Agg 注入 + plt.show 剥离 测试 ──


def test_auto_fix_injects_agg_when_pyplot_already_imported():
    """LLM 已 import pyplot 但未设置 Agg → 仍应注入 matplotlib.use('Agg')。"""
    from math_agent.tools.runner import _auto_fix_imports
    code = (
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3])\n"
        "plt.show()\n"
    )
    fixed = _auto_fix_imports(code)
    assert "matplotlib.use('Agg')" in fixed


def test_auto_fix_skips_agg_when_already_present():
    """LLM 已显式设置 Agg → 不重复注入。"""
    from math_agent.tools.runner import _auto_fix_imports
    code = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3])\n"
    )
    fixed = _auto_fix_imports(code)
    # 统计 Agg 出现次数：应该恰好 1 次（原代码的）
    assert fixed.count("matplotlib.use('Agg')") == 1


def test_auto_fix_handles_rcParams_without_pyplot():
    """仅使用 matplotlib.rcParams 也应注入 Agg。"""
    from math_agent.tools.runner import _auto_fix_imports
    code = (
        "import matplotlib\n"
        "matplotlib.rcParams['font.size'] = 10\n"
    )
    fixed = _auto_fix_imports(code)
    assert "matplotlib.use('Agg')" in fixed


def test_strip_dangerous_calls_removes_plt_show():
    """plt.show() 被移除。"""
    from math_agent.tools.runner import _strip_dangerous_calls
    code = (
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3])\n"
        "plt.show()\n"
    )
    stripped = _strip_dangerous_calls(code)
    assert "plt.show()" not in stripped
    assert "auto-removed" in stripped


def test_strip_dangerous_calls_removes_plt_ion():
    """plt.ion() 被移除。"""
    from math_agent.tools.runner import _strip_dangerous_calls
    code = (
        "import matplotlib.pyplot as plt\n"
        "plt.ion()\n"
        "plt.plot([1, 2, 3])\n"
    )
    stripped = _strip_dangerous_calls(code)
    assert "plt.ion()" not in stripped
    assert "auto-removed" in stripped


def test_strip_dangerous_calls_removes_fig_show():
    """fig.show() 被移除（Figure 对象调用 show 同理会阻塞）。"""
    from math_agent.tools.runner import _strip_dangerous_calls
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig = plt.figure()\n"
        "fig.show()\n"
        "plt.savefig('out.png')\n"
    )
    stripped = _strip_dangerous_calls(code)
    assert "fig.show()" not in stripped
    assert "auto-removed" in stripped


def test_strip_dangerous_calls_preserves_savefig():
    """savefig 不受影响。"""
    from math_agent.tools.runner import _strip_dangerous_calls
    code = (
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3])\n"
        "plt.savefig('out.png')\n"
    )
    stripped = _strip_dangerous_calls(code)
    assert "savefig" in stripped
    assert "auto-removed" not in stripped


def test_runner_end_to_end_no_hang_from_plt_show(workdir):
    """端到端：含 plt.show() 的代码在 run_python 中不阻塞，正常完成。"""
    code = (
        "import matplotlib\n"
        "matplotlib.use('TkAgg')  # LLM 可能显式设 GUI backend\n"
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3])\n"
        "plt.show()\n"
        "print('DONE')\n"
    )
    res = run_python(code, workdir=workdir, timeout=10)
    # 即使 LLM 设了 TkAgg，_auto_fix_imports 会前置 Agg（因为
    # already_marker 检查 Agg 而非 pyplot import），随后 _strip_dangerous_calls
    # 移除 plt.show() → 必须不阻塞，不超时
    assert res.success, f"timed out or crashed: {res.stderr[:200]}"
    assert "DONE" in res.stdout
```

- [ ] **Step 2: Run the new tests to verify they fail before implementation**

Wait — we should run these after Tasks 1 and 2 are both done, since they test the combined behavior. Run now, expect most to fail (except on the already-applied fixes):

```
pytest tests/test_runner.py -v -k "test_auto_fix_injects\|test_auto_fix_skips\|test_auto_fix_handles\|test_strip\|test_runner_end_to_end_no_hang"
```

Expected before Task 1+2: `test_auto_fix_injects_agg_when_pyplot_already_imported` FAILS (the core bug). After Tasks 1+2: all PASS.

- [ ] **Step 3: Commit**

```
git add tests/test_runner.py
git commit -m "test(runner): add tests for Agg injection fix and plt.show stripping"
```

---

### Task 4: Update coder prompt to instruct LLM to use Agg and avoid plt.show()

**Files:**
- Modify: `src/math_agent/prompts/coder.py:3-15`

- [ ] **Step 1: Add Agg backend + no plt.show() to coder SYSTEM prompt**

The coder SYSTEM prompt currently tells the LLM to "use matplotlib" and "save to *.png" but never mentions the Agg backend or prohibits `plt.show()`. The sensitivity prompt already has an "IRON RULE" about Agg — bring the coder prompt to parity.

Replace `src/math_agent/prompts/coder.py` lines 3-15:

```python
SYSTEM = (
    "你是建模队的工程师。把给定的最终模型实现为一段**独立可运行**的 Python 脚本。"
    "约束：只用 numpy / scipy / matplotlib；不联网；不读取本地未声明的文件；"
    "中文字体：开头加 `matplotlib.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']; matplotlib.rcParams['axes.unicode_minus']=False`；"
    "**IRON RULE：代码开头必须显式设置后端**——`import matplotlib; matplotlib.use('Agg')`，"
    "**禁止调用 plt.show() / plt.ion()**（非交互环境会阻塞）。"
    "图只通过 savefig 保存到当前目录 *.png；"
    "需 print 关键结果（含具体数字），并把图保存到当前目录的 *.png。\n"
    # 图表质量（参考 nature-figure 准则，只抓核心几条）：
    "绘图质量要求（每张图都必须满足）："
    "(1) 先定核心结论：每张图只论证一句话，多余面板不画；"
    "(2) 必备元素：title、坐标轴标签 + 单位、legend（除非只有一条曲线）、网格线适度（alpha≤0.3）；"
    "(3) 发表级 rcParams：`figure.dpi`=150（保存时 savefig dpi≥300）、`font.size`≥10、`axes.linewidth`=0.8、`axes.spines.right/top`=False、`legend.frameon`=False；"
    "(4) 配色克制：一张图最多 1 个 neutral + 1 个 signal + 1 个 accent 系列，避免彩虹色；"
    "(5) 多面板：优先 hero panel + 从属面板的非对称布局，少用等大小 2×2 网格。"
)
```

- [ ] **Step 2: Verify sensitivity prompt already covers this (no change needed)**

`sensitivity.py:30-32` already has: `"**IRON RULE：代码开头必须显式 import 所有用到的库**——import numpy as np; import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt"` — confirmed sufficient. No change needed.

- [ ] **Step 3: Run coder-related tests to check for prompt string changes**

```
pytest tests/nodes/test_coder.py -v
```
Expected: tests that check prompt contents may need updating if they assert on specific prompt strings. Review failures and update assertions as needed.

- [ ] **Step 4: Commit**

```
git add src/math_agent/prompts/coder.py
git commit -m "feat(prompt): add IRON RULE for Agg backend + no plt.show() in coder SYSTEM

Brings coder prompt to parity with sensitivity prompt (which already has this
rule). Defense-in-depth: even though runner.py now strips plt.show() and
injects Agg, telling the LLM upfront reduces the probability of generating
blocking calls in the first place."
```

---

### Task 5: Add per-attempt artifact persistence in `coder_node`

**Files:**
- Modify: `src/math_agent/nodes/coder.py:20-60`

- [ ] **Step 1: Save intermediate artifacts to disk after each attempt**

Currently, `coder_node` accumulates artifacts in `artifacts: list[CodeArtifact]` but only writes them to state at the very end (line 106: `delta: dict = {"code_artifacts": artifacts}`). If any attempt hangs or crashes mid-loop, all prior successful artifacts are lost.

Add a checkpoint mechanism that writes a JSON snapshot after each attempt:

Replace `coder_node` lines 32-60 with (only the changed parts — keep the retry loop, add persistence):

At the top of `coder_node`, after line 23, add:

```python
    checkpoint_path = workdir / "_checkpoint.json"
```

Inside the loop, after line 56 (after appending to artifacts), add:

```python
            # 每次 attempt 立即写 checkpoint，防止中间崩溃丢失已成功的图。
            import json as _json
            checkpoint_path.write_text(
                _json.dumps(
                    [{"purpose": a.purpose, "success": a.success,
                      "stdout": a.stdout[:500], "stderr": a.stderr[:500],
                      "artifact_paths": a.artifact_paths, "batch": a.batch}
                     for a in artifacts],
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
```

And add one structured log line for observability（`tracing.py` 没有独立的 `log` 函数，直接用 `print` 即可）：

```python
            print(f"[coder] purpose={i}/{len(purposes)} attempt={attempt} "
                  f"success={result.success} error_kind={result.error_kind}", flush=True)
```

- [ ] **Step 2: Run coder tests**

```
pytest tests/nodes/test_coder.py -v
```
Expected: existing tests PASS. Tests that mock `run_python` and inspect return values should be unaffected — checkpoints are a side effect.

- [ ] **Step 3: Commit**

```
git add src/math_agent/nodes/coder.py
git commit -m "feat(coder): add per-attempt checkpoint and progress logging

Writes _checkpoint.json after each code execution attempt so intermediate
results survive crashes. Adds structured progress log lines for observability
during long-running multi-figure batches."
```

---

### Task 6: Add same checkpoint to `sensitivity_node`

**Files:**
- Modify: `src/math_agent/nodes/sensitivity.py:88-105`

- [ ] **Step 1: Add checkpoint in sensitivity retry loop**

Same pattern as Task 5. Inside the sensitivity retry loop (after line 100), add:

```python
            # Checkpoint: save attempt result immediately.
            import json as _json
            cp_path = workdir / "_sensitivity_checkpoint.json"
            cp_path.write_text(
                _json.dumps({
                    "attempt": attempt,
                    "success": sandbox_result.success,
                    "stdout": sandbox_result.stdout[:500],
                    "stderr": sandbox_result.stderr[:500],
                    "error_kind": sandbox_result.error_kind,
                    "artifact_paths": sandbox_result.artifact_paths,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
```
And add a progress log line after line 100:

```python
            print(f"[sensitivity] attempt={attempt} success={sandbox_result.success} "
                  f"error_kind={sandbox_result.error_kind}", flush=True)
```

- [ ] **Step 2: Run sensitivity tests**

```
pytest tests/nodes/test_sensitivity.py -v
```
Expected: all PASS.

- [ ] **Step 3: Commit**

```
git add src/math_agent/nodes/sensitivity.py
git commit -m "feat(sensitivity): add per-attempt checkpoint and progress logging"
```

---

### Task 7: Integration verification

**Files:**
- None (verification only)

- [ ] **Step 1: Run full test suite**

```
pytest tests/ -v --timeout=60
```
Expected: all tests PASS. Pay special attention to `test_runner.py`, `test_coder.py`, `test_sensitivity.py`.

- [ ] **Step 2: Manual smoke test with a known plt.show()-prone code**

Create a temporary test file and run it through `run_python`:

```python
# save as /tmp/smoke_test.py and run with pytest
def test_smoke_plt_show_does_not_hang(workdir):
    from math_agent.tools.runner import run_python
    code = """
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.title("Test")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.savefig("test.png")
plt.show()
print("RESULT: baseline=test success=1")
"""
    res = run_python(code, workdir=workdir, timeout=10)
    assert res.success
    assert "success=1" in res.stdout
    assert "plt.show()" not in res.stdout  # stripped from original code
```

- [ ] **Step 3: Final commit (if smoke test is kept as a test)**

```
git add -A
git commit -m "test: add smoke test for plt.show() not hanging run_python"
```

---

## Summary of Changes

| File | Change | Purpose |
|------|--------|---------|
| `src/math_agent/tools/runner.py:51-57` | Change `already_marker` from import check to Agg check | Fix root cause: Agg injected even when pyplot already imported |
| `src/math_agent/tools/runner.py:58-70` | Add `_strip_dangerous_calls()` function | Defense-in-depth: strip `plt.show()`/`plt.ion()`/`fig.show()` |
| `src/math_agent/tools/runner.py:82` | Call `_strip_dangerous_calls` in `run_python` | Wire sanitization into execution path |
| `src/math_agent/prompts/coder.py:3-15` | Add IRON RULE for Agg + no plt.show() | Prevention: tell LLM upfront |
| `src/math_agent/nodes/coder.py:33,60` | Add checkpoint JSON + progress log | Robustness: survive mid-batch crashes |
| `src/math_agent/nodes/sensitivity.py:100` | Add checkpoint JSON + progress log | Same robustness for sensitivity |
| `tests/test_runner.py` | Add 8 new tests | Verify Agg injection, stripping (incl. fig.show), end-to-end |
