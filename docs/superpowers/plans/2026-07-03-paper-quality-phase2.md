# Paper Quality Improvement — Phase 2 Implementation Plan

> **历史方案记录（非现行事实源）**：路径、状态、命令和待办保留当时上下文；当前实现与限制见 [`../../README.md`](../../README.md)。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add baseline comparison runs (3 control experiments) + numeric result extraction + comparison table generation — so papers contain real quantitative comparisons instead of LLM-improvised prose.

**Architecture:** After coder's main figure runs, coder generates + runs 3 baseline scripts (no-schedule / simple-prediction / greedy). `extract_numeric_results()` in runner.py parses `RESULT:` lines from all artifacts' stdout. `table_assembler` generates a comparison table from these results. No new Pydantic models — reuse `CodeArtifact` with a `category` field.

**Tech Stack:** Python 3.11, LangGraph, Pydantic, pytest.

**Ponytail cuts from original Phase 2 design:**
- ~~BaselineResult model~~ → reuse CodeArtifact + `category` field
- ~~NumericResult model~~ → plain `dict[str, dict[str, float]]`
- ~~Table 5 (优缺点表)~~ → cut, conclusion section already has prose
- ~~Writer prompt numeric traceability injection~~ → cut, IRON RULE 1 + stdout injection already covers this
- ~~Structured numeric check in table_assembler~~ → cut, high false-positive, PaperCritic already covers

**Spec:** `docs/superpowers/specs/2026-07-03-paper-quality-improvement-design.md` (Section E, F)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/math_agent/state.py` | Modify | Add `category: str` to `CodeArtifact` |
| `src/math_agent/tools/runner.py` | Modify | Add `extract_numeric_results()` function |
| `src/math_agent/prompts/coder_baseline.py` | Create | Baseline prompt builder (1 parameterized prompt × 3 calls) |
| `src/math_agent/nodes/coder.py` | Modify | After main runs, generate + run 3 baselines |
| `src/math_agent/nodes/table_assembler.py` | Modify | Add comparison table from baseline artifacts |
| `tests/test_runner.py` | Modify | Tests for `extract_numeric_results` |
| `tests/test_table_assembler.py` | Modify | Tests for comparison table generation |

---

## Task 1: Add `category` field to CodeArtifact

**Files:**
- Modify: `src/math_agent/state.py:56-62`
- Test: `tests/test_state.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_state.py`:

```python
def test_code_artifact_has_category_field():
    """category 区分主方案(figure) vs 对照方案(baseline)。默认 'figure' 保持向后兼容。"""
    from math_agent.state import CodeArtifact
    a = CodeArtifact(purpose="test", code="print(1)")
    assert a.category == "figure"  # 默认值
    b = CodeArtifact(purpose="baseline", code="print(2)", category="baseline:no_schedule")
    assert b.category == "baseline:no_schedule"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_state.py::test_code_artifact_has_category_field -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Add the field**

In `src/math_agent/state.py`, find `CodeArtifact` (L56-62):

```python
class CodeArtifact(BaseModel):
    purpose: str
    code: str
    stdout: str = ""
    stderr: str = ""
    success: bool = False
    artifact_paths: list[str] = Field(default_factory=list)
```

Add `category` field:

```python
class CodeArtifact(BaseModel):
    purpose: str
    code: str
    stdout: str = ""
    stderr: str = ""
    success: bool = False
    artifact_paths: list[str] = Field(default_factory=list)
    # ponytail: 不新建 BaselineResult 模型，复用 CodeArtifact + category 区分
    # "figure" = 主方案绘图, "baseline:no_schedule" / "baseline:simple_pred" / "baseline:greedy" = 对照方案
    category: str = "figure"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_state.py::test_code_artifact_has_category_field -v`
Expected: PASS

- [ ] **Step 5: Run full suite for regression**

Run: `python -m pytest tests/ -q`
Expected: all PASS (existing code creates CodeArtifact without category → defaults to "figure")

- [ ] **Step 6: Commit**

```bash
git add src/math_agent/state.py tests/test_state.py
git commit -m "feat(state): add category field to CodeArtifact for baseline tagging"
```

---

## Task 2: Create `extract_numeric_results` in runner.py

**Files:**
- Modify: `src/math_agent/tools/runner.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_runner.py`:

```python
def test_extract_numeric_results_baseline_format():
    """RESULT: baseline=X metric1=Y metric2=Z 格式。"""
    from math_agent.tools.runner import extract_numeric_results
    stdout = (
        "一些输出...\n"
        "RESULT: baseline=no_schedule total_cost=1245.3 service_rate=0.82\n"
        "更多输出\n"
        "RESULT: baseline=greedy total_cost=980.0 service_rate=0.91 solve_time=12.5\n"
    )
    results = extract_numeric_results(stdout)
    assert "no_schedule" in results
    assert results["no_schedule"]["total_cost"] == 1245.3
    assert results["no_schedule"]["service_rate"] == 0.82
    assert results["greedy"]["solve_time"] == 12.5


def test_extract_numeric_results_scenario_format():
    """RESULT: scenario=X metric=Y 格式也支持。"""
    from math_agent.tools.runner import extract_numeric_results
    stdout = "RESULT: scenario=high_demand objective=9876.5\n"
    results = extract_numeric_results(stdout)
    assert "high_demand" in results
    assert results["high_demand"]["objective"] == 9876.5


def test_extract_numeric_results_no_result_lines():
    """没有 RESULT 行返回空 dict。"""
    from math_agent.tools.runner import extract_numeric_results
    assert extract_numeric_results("普通输出\n没有结果行") == {}
    assert extract_numeric_results("") == {}


def test_extract_numeric_results_ignores_malformed():
    """RESULT 行缺 key=value 对时跳过。"""
    from math_agent.tools.runner import extract_numeric_results
    stdout = "RESULT: baseline=test\n"  # 没有 metric=value
    results = extract_numeric_results(stdout)
    assert results == {} or results.get("test") == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runner.py::test_extract_numeric_results_baseline_format -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

Append to `src/math_agent/tools/runner.py` (after `run_python`):

```python
import re as _re

_RESULT_LINE_RE = _re.compile(
    r"^RESULT:\s*(?:baseline|scenario|method|config)=(\S+)\s+(.+)$",
    _re.MULTILINE,
)
_RESULT_PAIR_RE = _re.compile(r"(\w+)=(-?\d+\.?\d*(?:[eE][+-]?\d+)?)")


def extract_numeric_results(stdout: str) -> dict[str, dict[str, float]]:
    """从 stdout 提取所有 RESULT: 行，返回 {identifier: {metric: value}} 映射。

    支持格式（与 sensitivity.py 的 RESULT: parameter=... 互补，不冲突）：
      RESULT: baseline=no_schedule total_cost=1245.3 service_rate=0.82
      RESULT: scenario=high_demand objective=9876 solve_time=12.5
    """
    results: dict[str, dict[str, float]] = {}
    for m in _RESULT_LINE_RE.finditer(stdout):
        identifier = m.group(1)
        pairs_str = m.group(2)
        metrics: dict[str, float] = {}
        for pm in _RESULT_PAIR_RE.finditer(pairs_str):
            metrics[pm.group(1)] = float(pm.group(2))
        if metrics:
            results[identifier] = metrics
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_runner.py -v`
Expected: all PASS (existing + 4 new)

- [ ] **Step 5: Commit**

```bash
git add src/math_agent/tools/runner.py tests/test_runner.py
git commit -m "feat(runner): add extract_numeric_results for RESULT: line parsing"
```

---

## Task 3: Create baseline prompt builder

**Files:**
- Create: `src/math_agent/prompts/coder_baseline.py`
- Test: `tests/test_coder_baseline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_coder_baseline.py`:

```python
"""对照方案 prompt 构建器测试。"""
from math_agent.prompts.coder_baseline import BASELINE_SPECS, build_baseline_prompt


def test_baseline_specs_has_three():
    assert len(BASELINE_SPECS) == 3
    names = [s[0] for s in BASELINE_SPECS]
    assert "无调度" in names
    assert "简单平均预测" in names
    assert "贪婪启发式" in names


def test_baseline_prompt_contains_main_code():
    prompt = build_baseline_prompt(
        problem="共享单车调度",
        main_code="import numpy as np\nprint('main')",
        name="无调度",
        category="no_schedule",
        instruction="把优化步骤删除",
    )
    assert "import numpy as np" in prompt
    assert "无调度" in prompt
    assert "RESULT: baseline=no_schedule" in prompt


def test_baseline_prompt_contains_output_contract():
    prompt = build_baseline_prompt(
        problem="test",
        main_code="print(1)",
        name="贪婪",
        category="greedy",
        instruction="用贪心替换",
    )
    assert "RESULT: baseline=greedy" in prompt
    assert "JSON" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_coder_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/math_agent/prompts/coder_baseline.py`:

```python
"""对照方案 prompt 构建器。

3 个预设对照方案，用一个参数化 prompt 模板 × 3 次调用。
每个方案基于主方案代码做最小修改（删优化 / 换预测 / 换求解器），
保留数据加载和评估逻辑不变，输出 RESULT: 行供 extract_numeric_results 解析。
"""
from __future__ import annotations

# (name, category, instruction) — 3 个预设对照方案
BASELINE_SPECS: list[tuple[str, str, str]] = [
    (
        "无调度",
        "no_schedule",
        "把主方案代码中的优化/调度求解步骤全部删除，改为'不调整/保持现状'（所有决策变量取默认值 0 或保持初始值）。"
        "保留数据加载、需求计算和评估逻辑不变。最终用相同的指标函数计算成本和服务率。",
    ),
    (
        "简单平均预测",
        "simple_pred",
        "把主方案代码中的预测模型（XGBoost/STGNN/回归等）替换为简单历史均值预测："
        "prediction = np.mean(historical_data, axis=0)。保留调度/优化代码和评估逻辑不变。",
    ),
    (
        "贪婪启发式",
        "greedy",
        "把主方案代码中的优化求解器（MILP/随机规划/滚动优化等）替换为贪心策略："
        "while 循环 + 每次取当前需求最大（或缺口最大）的站点优先分配，直到资源耗尽。"
        "保留数据加载、预测和评估逻辑不变。",
    ),
]


def build_baseline_prompt(
    problem: str,
    main_code: str,
    name: str,
    category: str,
    instruction: str,
) -> str:
    """构造对照方案代码生成 prompt。

    一个模板，3 次调用不同 (name, category, instruction)。
    输出 RESULT: baseline={category} metric1=v1 metric2=v2 格式。
    """
    return (
        f"# 题目\n{problem[:500]}\n\n"
        f"# 对照方案：{name}\n"
        f"## 修改指令\n{instruction}\n\n"
        f"# 主方案代码（参考基础）\n```python\n{main_code[:3000]}\n```\n\n"
        f"# 输出要求\n"
        f"基于主方案代码做上述修改，生成一段**独立可运行**的 Python 脚本。\n"
        f"脚本末尾必须用 print 输出至少 2 个指标，格式严格如下：\n"
        f"print(f'RESULT: baseline={category} total_cost={{total_cost}} service_rate={{service_rate}}')\n"
        f"（指标名可按题目调整，但必须以 RESULT: baseline={category} 开头）\n\n"
        f"请输出 JSON：{{\"purpose\": \"{name}对照方案\", \"code\": str}}，code 字段是完整的 Python 源码。"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_coder_baseline.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/math_agent/prompts/coder_baseline.py tests/test_coder_baseline.py
git commit -m "feat(coder): add baseline prompt builder — 1 template × 3 control experiments"
```

---

## Task 4: Add baseline runs to coder_node

**Files:**
- Modify: `src/math_agent/nodes/coder.py:22-67`
- Test: `tests/test_coder_baseline.py` (integration)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_coder_baseline.py`:

```python
def test_coder_node_produces_baseline_artifacts(monkeypatch):
    """coder_node 应在主方案后追加 3 个 category='baseline:...' 的 artifacts。"""
    from math_agent.nodes.coder import coder_node, CoderDraft
    from math_agent.state import MathModelingState, ModelVersion
    from math_agent.tools.runner import RunResult

    s = MathModelingState(problem="test", output_dir="/tmp/test_coder_baseline")
    s.model_versions.append(ModelVersion(
        stage="final", description="test model",
        variables={"x": "v"}, figure_purposes=["plot1"],
    ))

    call_count = {"n": 0}
    def mock_complete(prompt, *, schema=None, **kw):
        call_count["n"] += 1
        if call_count["n"] <= 1:
            # 主方案
            return CoderDraft(purpose="main plot", code="print('main')")
        # 对照方案（第 2-4 次调用）
        specs = ["no_schedule", "simple_pred", "greedy"]
        idx = call_count["n"] - 2
        return CoderDraft(
            purpose=f"baseline {specs[idx]}",
            code=f"print('RESULT: baseline={specs[idx]} total_cost=100.0 service_rate=0.9')",
        )

    def mock_run(code, *, workdir, timeout=60, **kw):
        return RunResult(success=True, stdout=code.replace("print(", "").replace("')", "").replace("'", ""),
                         artifact_paths=[])

    monkeypatch.setattr("math_agent.nodes.coder.complete", mock_complete)
    monkeypatch.setattr("math_agent.nodes.coder.run_python", mock_run)

    result = coder_node(s)
    artifacts = result["code_artifacts"]
    # 1 主方案 + 3 对照方案 = 4
    assert len(artifacts) == 4
    baseline_arts = [a for a in artifacts if a.category.startswith("baseline:")]
    assert len(baseline_arts) == 3
    categories = [a.category for a in baseline_arts]
    assert "baseline:no_schedule" in categories
    assert "baseline:simple_pred" in categories
    assert "baseline:greedy" in categories
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_coder_baseline.py::test_coder_node_produces_baseline_artifacts -v`
Expected: FAIL (only 1 artifact, no baselines)

- [ ] **Step 3: Modify coder_node**

In `src/math_agent/nodes/coder.py`, add imports after L9:

```python
from math_agent.prompts.coder_baseline import BASELINE_SPECS, build_baseline_prompt
```

After the main loop (after L57, before `delta: dict = ...`), add baseline runs:

```python
    # ---- Phase 2: 对照方案运行 ----
    # 取最后一个成功的主方案代码作为基础
    main_code = next((a.code for a in reversed(artifacts) if a.success), "")
    if main_code:
        for name, category, instruction in BASELINE_SPECS:
            try:
                baseline_draft: CoderDraft = complete(
                    build_baseline_prompt(state.problem, main_code, name, category, instruction),
                    schema=CoderDraft,
                    system=SYSTEM,
                    model=MODEL_ROUTING["coder"],
                )
                baseline_result = run_python(
                    baseline_draft.code,
                    workdir=workdir / f"baseline_{category}",
                    timeout=300,
                )
                artifacts.append(
                    CodeArtifact(
                        purpose=baseline_draft.purpose,
                        code=baseline_draft.code,
                        stdout=baseline_result.stdout,
                        stderr=baseline_result.stderr,
                        success=baseline_result.success,
                        artifact_paths=baseline_result.artifact_paths,
                        category=f"baseline:{category}",
                    )
                )
            except Exception as e:
                # 对照方案失败不阻断主流程
                artifacts.append(
                    CodeArtifact(
                        purpose=f"{name}对照方案（失败）",
                        code="",
                        stdout="",
                        stderr=str(e)[:500],
                        success=False,
                        category=f"baseline:{category}",
                    )
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_coder_baseline.py -v`
Expected: 4 PASS

- [ ] **Step 5: Run full suite for regression**

Run: `python -m pytest tests/ -q`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/math_agent/nodes/coder.py tests/test_coder_baseline.py
git commit -m "feat(coder): run 3 baseline experiments after main figure runs"
```

---

## Task 5: Add comparison table to table_assembler

**Files:**
- Modify: `src/math_agent/nodes/table_assembler.py`
- Test: `tests/test_table_assembler.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_table_assembler.py`:

```python
from math_agent.nodes.table_assembler import _generate_comparison_table
from math_agent.state import CodeArtifact


def test_comparison_table_from_baselines():
    """从 baseline artifacts 的 stdout 提取结果生成对比表。"""
    artifacts = [
        CodeArtifact(
            purpose="无调度对照", code="", success=True,
            stdout="RESULT: baseline=no_schedule total_cost=1245.3 service_rate=0.82",
            category="baseline:no_schedule",
        ),
        CodeArtifact(
            purpose="贪婪对照", code="", success=True,
            stdout="RESULT: baseline=greedy total_cost=980.0 service_rate=0.91",
            category="baseline:greedy",
        ),
        CodeArtifact(
            purpose="主方案", code="", success=True,
            stdout="RESULT: baseline=ours total_cost=750.5 service_rate=0.95",
            category="figure",  # 主方案
        ),
    ]
    table = _generate_comparison_table(artifacts)
    assert "| 方案 |" in table
    assert "无调度" in table or "no_schedule" in table
    assert "1245.3" in table
    assert "980.0" in table
    assert "750.5" in table


def test_comparison_table_empty_when_no_baselines():
    """没有 baseline artifacts 时返回空字符串。"""
    artifacts = [
        CodeArtifact(purpose="main", code="", success=True, stdout="", category="figure"),
    ]
    table = _generate_comparison_table(artifacts)
    assert table == ""


def test_comparison_table_handles_failed_baselines():
    """失败的对照方案在表中标记'运行失败'。"""
    artifacts = [
        CodeArtifact(
            purpose="无调度对照", code="", success=False,
            stdout="", stderr="error", category="baseline:no_schedule",
        ),
        CodeArtifact(
            purpose="贪婪对照", code="", success=True,
            stdout="RESULT: baseline=greedy total_cost=980.0",
            category="baseline:greedy",
        ),
    ]
    table = _generate_comparison_table(artifacts)
    assert "运行失败" in table
    assert "980.0" in table
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_table_assembler.py::test_comparison_table_from_baselines -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

In `src/math_agent/nodes/table_assembler.py`, add import at top (after existing imports):

```python
from math_agent.tools.runner import extract_numeric_results
```

Add the function (after `_generate_sensitivity_table`):

```python
# baseline category → 中文显示名
_BASELINE_NAMES = {
    "no_schedule": "无调度",
    "simple_pred": "简单平均预测",
    "greedy": "贪婪启发式",
    "ours": "本文方案",
}


def _generate_comparison_table(artifacts: list) -> str:
    """从 code_artifacts 中提取 baseline 对照结果生成对比表。

    主方案（category='figure'）的 stdout 如果也含 RESULT: baseline=ours 也纳入。
    无 baseline artifacts 或无 RESULT 行时返回空字符串。
    """
    # 收集所有 artifact 的提取结果
    rows: list[dict[str, str]] = []
    for a in artifacts:
        if not a.stdout:
            continue
        results = extract_numeric_results(a.stdout)
        if not results:
            # 有 baseline category 但没提取到结果 → 标记失败
            if a.category.startswith("baseline:"):
                cat_key = a.category.split(":", 1)[1]
                name = _BASELINE_NAMES.get(cat_key, cat_key)
                rows.append({"方案": name, "状态": "运行失败"})
            continue
        for identifier, metrics in results.items():
            name = _BASELINE_NAMES.get(identifier, identifier)
            row = {"方案": name}
            row.update({k: f"{v:.4g}" for k, v in metrics.items()})
            rows.append(row)

    if not rows:
        return ""

    # 收集所有指标列名（除"方案"外）
    all_metrics: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen and k != "方案":
                seen.add(k)
                all_metrics.append(k)

    # 如果没有指标列（全是失败的），加一个状态列
    if not all_metrics:
        all_metrics = ["状态"]

    header = "| 方案 | " + " | ".join(all_metrics) + " |"
    sep = "|---|" + "|".join(["---" for _ in all_metrics]) + "|"
    lines = [header, sep]
    for r in rows:
        cells = [r.get("方案", "—")]
        for m in all_metrics:
            cells.append(r.get(m, "—"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_table_assembler.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/math_agent/nodes/table_assembler.py tests/test_table_assembler.py
git commit -m "feat(table_assembler): generate comparison table from baseline artifacts"
```

---

## Task 6: Inject comparison table into paper.solution

**Files:**
- Modify: `src/math_agent/nodes/table_assembler.py` (in `table_assembler_node`)
- Test: `tests/test_table_assembler.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_table_assembler.py`:

```python
def test_table_assembler_node_injects_comparison_table():
    """table_assembler 应把对比表注入 paper.solution。"""
    from math_agent.state import MathModelingState, ModelVersion, PaperSections, CodeArtifact
    s = MathModelingState(problem="test")
    s.model_versions.append(ModelVersion(
        stage="final", description="m", variables={"x": "v"},
    ))
    s.code_artifacts = [
        CodeArtifact(
            purpose="无调度", code="", success=True,
            stdout="RESULT: baseline=no_schedule total_cost=1245.3 service_rate=0.82",
            category="baseline:no_schedule",
        ),
        CodeArtifact(
            purpose="贪婪", code="", success=True,
            stdout="RESULT: baseline=greedy total_cost=980.0 service_rate=0.91",
            category="baseline:greedy",
        ),
    ]
    s.paper = PaperSections(solution="## 求解算法与流程\n求解过程。")
    result = table_assembler_node(s)
    assert "| 方案 |" in result["paper"].solution
    assert "无调度" in result["paper"].solution
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_table_assembler.py::test_table_assembler_node_injects_comparison_table -v`
Expected: FAIL (no comparison table in solution)

- [ ] **Step 3: Modify table_assembler_node**

In `src/math_agent/nodes/table_assembler.py`, find `table_assembler_node` function. After the sensitivity table injection (after `paper.sensitivity = _inject_table(...)`), add:

```python
    # 对比表（从 baseline artifacts 提取）
    comp_table = _generate_comparison_table(state.code_artifacts)
    paper.solution = _inject_table(paper.solution, "各方案结果对比表", comp_table)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_table_assembler.py -v`
Expected: all PASS

- [ ] **Step 5: Run full suite for regression**

Run: `python -m pytest tests/ -q`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/math_agent/nodes/table_assembler.py tests/test_table_assembler.py
git commit -m "feat(table_assembler): inject comparison table into paper.solution"
```

---

## Task 7: Update e2e test script for Phase 2

**Files:**
- Modify: `scripts/e2e_table_assembler.py`

- [ ] **Step 1: Add Phase 2 checks to the e2e script**

In `scripts/e2e_table_assembler.py`, in the `main()` function, after the existing checks (before `# 5) 空 state`), add:

```python
    # Phase 2: 对比表
    from math_agent.state import CodeArtifact
    s.code_artifacts = [
        CodeArtifact(
            purpose="无调度", code="", success=True,
            stdout="RESULT: baseline=no_schedule total_cost=1500.0 service_rate=0.70",
            category="baseline:no_schedule",
        ),
        CodeArtifact(
            purpose="贪婪", code="", success=True,
            stdout="RESULT: baseline=greedy total_cost=1100.0 service_rate=0.85",
            category="baseline:greedy",
        ),
    ]
    result_p2 = table_assembler_node(s)
    assert "| 方案 |" in result_p2["paper"].solution, "对比表未注入 solution"
    assert "无调度" in result_p2["paper"].solution
    print("✓ 对比表注入 solution（Phase 2）")
```

- [ ] **Step 2: Run the e2e script**

Run: `python scripts/e2e_table_assembler.py`
Expected: all ✓ + "全部通过。"

- [ ] **Step 3: Commit**

```bash
git add scripts/e2e_table_assembler.py
git commit -m "test: add Phase 2 comparison table checks to e2e script"
```

---

## Task 8: Run full regression + Plan D e2e

**Files:** none (verification only)

- [ ] **Step 1: Run all unit tests**

Run: `python -m pytest tests/ -v --tb=short -q`
Expected: all PASS

- [ ] **Step 2: Run Plan D e2e (mock LLM, real graph)**

Run: `python scripts/e2e_plan_d.py`
Expected: PASS — verifies coder's new baseline runs don't break the graph.
Note: the mock LLM in e2e_plan_d.py may need to handle the extra 3 coder calls for baselines. If it fails, update the mock to return baseline drafts.

- [ ] **Step 3: Run table_assembler e2e**

Run: `python scripts/e2e_table_assembler.py`
Expected: all ✓

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "test: fix e2e regression after Phase 2 baseline runs"
```
(only if fixes were needed)

---

## Self-Review

**Spec coverage:**
- Section E (baseline comparison): Tasks 3, 4 ✓
- Section F (numeric extraction): Task 2 ✓
- Comparison table: Tasks 5, 6 ✓

**Ponytail cuts applied:**
- No BaselineResult model → CodeArtifact.category (Task 1)
- No NumericResult model → plain dict (Task 2)
- No table 5 (优缺点) → cut
- No writer prompt numeric injection → cut
- 1 parameterized prompt × 3 calls (Task 3)

**Risk:** LLM-generated baseline code may not run successfully. Mitigation: failures don't block the graph (Task 4 try/except), table marks "运行失败" (Task 5).
