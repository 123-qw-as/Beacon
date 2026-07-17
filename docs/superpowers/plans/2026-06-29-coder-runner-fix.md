# Coder/Runner 修复与可观察性增强计划

> **历史方案记录（非现行事实源）**：路径、状态、命令和待办保留当时上下文；当前实现与限制见 [`../../README.md`](../../README.md)。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `tools/runner.py` 的相对路径 bug，让 LLM 生成的代码真正能在沙箱里跑通；同时把 coder 全失败的情况显式回写到 `state.errors`，让下游 writer 不再"误把 success=False 当数字来源"。

**Architecture:** 单点根因（`subprocess.run(cwd=relative, [..., relative_script])` 触发 cwd+script 双重前缀）只需一行 `.resolve()` 修复 + 1 条回归测试；同时利用同一次改动顺手收口 coder 全失败不写 errors 的问题。两处改动各自独立、各自可测。

**Tech Stack:** 不引入新依赖。Python 3.11+ / pytest / subprocess（stdlib）。

---

## 根因诊断（已验证）

实测复现：
```python
# tools/runner.py:38-54 当前行为
wd = Path("runs/eval_v2/attempt_0")       # 相对路径
script = wd / "_run.py"                    # 仍是相对
subprocess.run([sys.executable, str(script)], cwd=str(wd), ...)
# subprocess 启动子进程时 cwd 会被切换；python 解释器把 argv 里的
# 相对 script 路径解释为相对于新 cwd——与平台无关。
# 结果：尝试打开 runs/eval_v2/attempt_0/runs/eval_v2/attempt_0/_run.py
```

**stderr 实测**：
```
can't open file 'C:\...\runs\eval_v2\attempt_0\runs\eval_v2\attempt_0\_run.py'
```

**为什么测试没抓住**：`tests/conftest.py:13-15` 的 `workdir` fixture 用 `monkeypatch.chdir(tmp_path)` 把 cwd 切到了 `tmp_path`，传给 runner 的 `tmp_path` 是绝对路径 → 单测一直走绝对路径分支，永远绕过 bug。

**修复**：在 `run_python` 开头加 `workdir = Path(workdir).resolve()`，保证 cwd 和 script 都是绝对路径。

**用户暴露面**：`cli.py:33` 的 `--out` 默认值是 `Path("runs/latest")`（相对路径）。任何不显式传绝对路径的用户都会踩坑——`coder` 永远 `success=False`，`writer` 拿到空 stdout 仍编造数字。这就是上一轮 `runs/eval_v2/paper.md` 附录全 False 的实际成因。

**未受影响的分支**：`coder.py:23` 中 `state.output_dir is None` 时回退用 `tempfile.mkdtemp(...)`，该函数返回绝对路径，本 bug 不触发。CLI 总是传 `output_dir`，所以生产路径必然踩坑；只有不走 CLI 的单元测试因 `tmp_path` 是绝对路径而逃过一劫。

---

## 文件结构

```
src/math_agent/
├── tools/
│   └── runner.py            # MODIFY: 1 行 .resolve()
└── nodes/
    └── coder.py             # MODIFY: 重试用尽后写 state.errors
tests/
├── test_runner.py           # MODIFY: 新增 1 条相对路径回归测试
├── conftest.py              # （不动；新测试用 tmp_path 直接构造相对路径，避免 chdir 副作用）
└── nodes/
    └── test_coder.py        # MODIFY: 新增 1 条 "全失败写 errors" 测试
```

职责说明：
- `tools/runner.py` 的边界是「给一段代码与一个工作目录就能跑」。调用方传相对路径是合法的；runner 自己 normalize 是它的责任。
- `nodes/coder.py` 在 retry 用尽后必须显式向下游报告失败，避免 writer 看到 `success=False` 仍当作真数据。
- 不动 `conftest.py` 的 `workdir` fixture，因为它满足其它 4 个测试的语义需求；新测试自己用 `tmp_path` 构造相对路径场景。

---

## Phase 1：runner 相对路径修复（核心）

### Task 1.1：写失败的回归测试

**Files:**
- Test: `tests/test_runner.py`

- [ ] **Step 1: 在 `tests/test_runner.py` 末尾新增测试**

```python
def test_runner_accepts_relative_workdir(tmp_path, monkeypatch):
    """回归：workdir 传相对路径时不应触发 cwd+script 双重前缀。

    历史 bug：subprocess 切换 cwd 后，python 解释器把 argv 里的相对 script
    解释为相对于新 cwd，导致 "attempt_0/runs/.../attempt_0/_run.py"
    （所有平台都有此行为，不是 Windows 特有）。
    """
    monkeypatch.chdir(tmp_path)
    rel = Path("runs/x/attempt_0")  # 故意传相对路径
    res = run_python("print('relpath_ok')", workdir=rel)
    assert res.success, f"stderr={res.stderr!r}"
    assert "relpath_ok" in res.stdout
```

- [ ] **Step 2: 加 `from pathlib import Path` 到测试顶部（如尚未引入）**

`tests/test_runner.py` 当前没有 import Path，需要补：

```python
from pathlib import Path

from math_agent.tools.runner import run_python, RunResult
```

- [ ] **Step 3: 跑测试确认失败**

Run: `pytest tests/test_runner.py::test_runner_accepts_relative_workdir -v`

Expected: FAIL，stderr 包含双重路径前缀，例如：
```
can't open file '...\runs\x\attempt_0\runs\x\attempt_0\_run.py': [Errno 2]
```

### Task 1.2：修复 runner

**Files:**
- Modify: `src/math_agent/tools/runner.py:38-42`

- [ ] **Step 1: 把 `run_python` 入口的 workdir normalize 改为绝对路径**

把当前的：

```python
def run_python(code: str, *, workdir: Path, timeout: int = 60) -> RunResult:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / "_run.py"
    script.write_text(code, encoding="utf-8")
```

改为：

```python
def run_python(code: str, *, workdir: Path, timeout: int = 60) -> RunResult:
    # 必须 resolve()：subprocess 切换 cwd 后，python 解释器把 argv 里的
    # 相对 script 解释为相对于新 cwd，导致路径双重前缀。
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / "_run.py"
    script.write_text(code, encoding="utf-8")
```

只改 1 行（加 `.resolve()`）+ 1 行注释。下方 `subprocess.run([sys.executable, str(script)], cwd=workdir, ...)` 不必改：cwd 和 script 现在都是绝对路径，不会再有歧义。

- [ ] **Step 2: 跑新测试确认通过**

Run: `pytest tests/test_runner.py::test_runner_accepts_relative_workdir -v`
Expected: PASS

- [ ] **Step 3: 跑全部 runner 测试确认无回归**

Run: `pytest tests/test_runner.py -v`
Expected: 5 passed（原 4 条 + 新增 1 条）

- [ ] **Step 4: 提交**

```bash
git add src/math_agent/tools/runner.py tests/test_runner.py
git commit -m "fix(runner): resolve workdir to absolute path to avoid cwd+script double-prefix"
```

---

## Phase 2：coder 全失败显式写 errors

### Task 2.1：写失败的测试

**Files:**
- Test: `tests/nodes/test_coder.py`

当前 `coder_node`（`src/math_agent/nodes/coder.py:21-50`）在所有重试都失败时悄无声息地返回失败 artifacts，writer 仍可能拿到 `success=False` 的 stdout/stderr 误以为有数据。要让 coder 在「最后一次仍失败」时显式写一条人类可读消息进 `state.errors`。

- [ ] **Step 1: 在 `tests/nodes/test_coder.py` 末尾新增测试**

```python
def test_coder_records_error_when_all_retries_fail(mocker, workdir):
    """所有尝试都失败时，应在 state.errors 中显式记录，便于下游识别。

    用 itertools.cycle 而非固定列表：若未来 MAX_CODE_RETRIES 上调，
    测试不会因 mock 耗尽而 StopIteration 掩盖真正问题。
    """
    from itertools import cycle
    mocker.patch(
        "math_agent.nodes.coder.complete",
        side_effect=cycle([CoderDraft(purpose="solve", code="raise RuntimeError('boom')")]),
    )
    s = MathModelingState(problem="p", output_dir=str(workdir))
    s.model_versions.append(ModelVersion(stage="final", description="d"))

    delta = coder_node(s)

    # 既有行为：保留全部失败 artifact
    assert len(delta["code_artifacts"]) >= 1
    assert all(a.success is False for a in delta["code_artifacts"])

    # 新增：必须显式写入一条 error，且前缀与 sensitivity_node 约定一致
    assert "errors" in delta and delta["errors"], "coder 全失败时应写 state.errors"
    assert delta["errors"][0].startswith("coder:")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/nodes/test_coder.py::test_coder_records_error_when_all_retries_fail -v`
Expected: FAIL with `KeyError: 'errors'` 或 `AssertionError`（delta 里没 errors 键）。

### Task 2.2：在 coder_node 写入 errors

**Files:**
- Modify: `src/math_agent/nodes/coder.py:21-50`

- [ ] **Step 1: 修改 `coder_node` 的返回**

把当前结尾：

```python
        if result.success:
            break
        prev_err = result.stderr

    return {"code_artifacts": artifacts}
```

改为：

```python
        if result.success:
            break
        prev_err = result.stderr

    delta: dict = {"code_artifacts": artifacts}
    if not any(a.success for a in artifacts):
        # 全部 attempts 都失败：显式写 error 让 writer / paper_critic 看到，
        # 避免 IRON RULE 1（"数字必须可追溯"）被失败 stdout 偷偷绕过。
        # 前缀与 sensitivity_node 已有的 "sensitivity: ..." 约定对齐。
        last_stderr = artifacts[-1].stderr if artifacts else ""
        delta["errors"] = [
            f"coder: 所有 {len(artifacts)} 次尝试均失败；最后一次 stderr 节选：{last_stderr[:300]}"
        ]
    return delta
```

- [ ] **Step 2: 跑新测试确认通过**

Run: `pytest tests/nodes/test_coder.py::test_coder_records_error_when_all_retries_fail -v`
Expected: PASS

- [ ] **Step 3: 跑全部 coder 测试确认无回归**

Run: `pytest tests/nodes/test_coder.py -v`
Expected: 3 passed（含既有的 2 条 + 新增 1 条）

- [ ] **Step 4: 提交**

```bash
git add src/math_agent/nodes/coder.py tests/nodes/test_coder.py
git commit -m "feat(coder): record explicit error when all retries fail"
```

---

## Phase 3：端到端验收

### Task 3.1：跑全量测试

- [ ] **Step 1: 全量 pytest**

Run: `pytest -q`
Expected: **65 passed, 1 skipped（共 66 collected）**。原先 64 collected / 63 passed + 1 skipped；本计划 +2 测试 +2 pass。无新增 warning。

### Task 3.2：（手动验收，非门禁）跑端到端 demo

⚠️ **不是 pytest 测试，是人工验收步骤**。
- 跑通需 10-15 分钟，远超 Bash tool 10 分钟超时上限，无法在 agent harness 内 inline 执行
- 模型不确定性 + 本地路由 → 不适合作为 CI/回归门禁
- 真正的回归门禁在 Task 1.1：Phase 1 失败 = bug 复发

用途：跑一次确认 paper 附录的 `success=True` 真的能出现，并肉眼看一下数字来源是否真的能追溯。

- [ ] **Step 1: 清理旧的 eval_v2 输出**

Run（Git Bash）:
```bash
cd C:/Users/lwh86/Desktop/progame/build-agent && rm -rf runs/eval_v3
```

- [ ] **Step 2: 跑端到端**

Run（推荐直接在用户终端跑，不在 agent harness 内）:
```bash
cd C:/Users/lwh86/Desktop/progame/build-agent && python -m math_agent.cli run --problem tests/fixtures/sample_problem.json --out runs/eval_v3 --no-interrupt
```

Expected: 大约 10-15 分钟跑完，最后一行 `done. paper at runs\eval_v3\paper.md`。

- [ ] **Step 3: 验证 paper.md 附录里至少有一个 `success=True`**

```bash
grep -E "success=(True|False)" runs/eval_v3/paper.md
```

Expected: 至少出现一次 `（success=True）`。如果仍全部 False，stderr 应当不再是路径双重前缀，而是别的问题（如 matplotlib 中文字体、scipy 缺包等）——那不在本计划范围。

- [ ] **Step 4: 顺带确认 sensitivity 也受益**

```bash
ls runs/eval_v3/sensitivity/
```

Expected: 看到 `_run.py` + 至少一个 `*.png` 或数据文件。若 sensitivity 仍空，stderr 也不应再含路径双重前缀。

- [ ] **Step 5: 肉眼检查求解段数字是否能追溯**

```bash
grep -A2 "Claim" runs/eval_v3/paper.md
```

观察：求解章节引的关键数字是否在附录 stdout 段里实际出现过。不强求 100% 匹配（LLM 不会精确复刻 print 输出），但应当出现"附录有数字"而非"附录全空"的状态。

---

## 已知非本计划范围（明确不做）

- **真隔离沙箱**（docker/firejail）→ Plan C
- **coder 多轮回灌错误自修复**（当前只重试 1 次，不传递结构化错误类型）→ 留给将来
- **sensitivity_node 顺带受益**：本计划修了 `tools/runner.py:39` 的 normalize，`sensitivity.py:87` 调用 `run_python(...)` 同样不再踩坑。但 sensitivity 没有 retry，且本计划**不为它写 e2e 回归测试**——受益情况靠 Task 3.2 手动验收时观察 `runs/eval_v3/sensitivity/` 是否产出图来确认
- **不重写 writer prompt**：上一轮已完成 Jinja2 化 + IRON RULES，本计划不动
- **不改 `state.errors` 的 reducer**：state.py 已用 `Annotated[list[str], add]`，append 语义直接生效

---

## 自我审查

**1. Spec 覆盖：**
- runner 相对路径 bug → Task 1.1 + 1.2 ✓
- coder 全失败不写 errors → Task 2.1 + 2.2 ✓
- 端到端是否真的修复了 → Task 3.2（手动验收）✓

**2. Placeholder 扫描：** 全文无 TBD/TODO/「类似 Task N」/「适当的错误处理」。每个 step 都给出可粘贴的代码或精确的 grep/pytest 命令。

**3. 类型一致性：**
- `RunResult` 字段名 `success / stdout / stderr / artifact_paths` 与 `tools/runner.py:21-25` 一致
- `CoderDraft` 与 `coder.py:13-15` 一致，测试 import 路径 `from math_agent.nodes.coder import coder_node, CoderDraft` 与现有 `test_coder.py:2` 一致
- `MathModelingState.errors` 是 `Annotated[list[str], add]`（见 `state.py:124`）→ 节点返回 `{"errors": [...]}` 会被 reducer 追加 ✓
- `delta` 字典 key `code_artifacts` / `errors` 与现有节点惯例一致
- `errors` 元素前缀 `"coder: ..."` 与 `sensitivity_node` 的 `"sensitivity: ..."` 约定对齐

**4. 改动半径**：4 个文件，~8 行代码（runner +2 行含注释，coder +6 行），**2 条新 pytest 测试**（Phase 3 是手动验收，不计入 pytest 测试）。2 个 commit。Phase 1+2 编辑+跑测试预计 15 分钟；Phase 3 手动验收另需 10-15 分钟。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-29-coder-runner-fix.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
