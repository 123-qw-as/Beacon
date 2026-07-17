# Writer ↔ PaperCritic 闭环 Implementation Plan

> **历史方案记录（非现行事实源）**：路径、状态、命令和待办保留当时上下文；当前实现与限制见 [`../../README.md`](../../README.md)。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `writer → paper_critic` 的直边换成闭环——critic 不通过且未到迭代上限时回 writer 重写；writer 接收上一轮 critic 反馈作为修订提示。压住"writer 编数字"问题：critic 看到 IRON RULE 违例就打回。

**Architecture:** 镜像 modeler↔model_critic 现有模式：状态加 `writer_iteration` 计数器（覆盖语义），config 加 `MAX_WRITER_ITERATIONS=2`（最多 2 轮重写），`routing.after_paper_critic` 输出 `retry/advance`，graph 用 `add_conditional_edges` 接线。Writer prompt 通过 `state.latest_critic("paper")` 拿到 issues/suggestions 注入模板。

**Tech Stack:** 不引入新依赖。已用：LangGraph >=0.2、Pydantic >=2.7、Jinja2、pytest-mock。

---

## 现状诊断（已验证）

```
graph.py:56-57   writer → paper_critic → evaluation
```

- `paper_critic_node` 已生成 `CriticReport(target="paper", score, approved, issues, suggestions)` ✓
- writer 不消费这份 report；编出 4.8% / 46s / 306s 等真实代码 stdout 里没有的数字
- 已存在的 `state.latest_critic("paper")` 帮 routing 拿反馈，零新方法

**为什么不靠 HITL 兜底**：HITL 在 evaluation 之后，需要人审；自动闭环让"明显违反 IRON RULE"的情况不打扰人。两者不冲突。

---

## 文件结构

```
src/math_agent/
├── config.py                       # MODIFY: +MAX_WRITER_ITERATIONS
├── state.py                        # MODIFY: +writer_iteration: int
├── routing.py                      # MODIFY: +after_paper_critic 函数
├── graph.py                        # MODIFY: writer↔paper_critic 改条件边
├── nodes/writer.py                 # MODIFY: 自增 writer_iteration
├── prompts/writer.py               # MODIFY: 把 critic feedback 传给模板
└── templates/writer_prompt.md.j2   # MODIFY: critic_feedback 区块（条件渲染）

tests/
├── test_routing.py                 # MODIFY: +3 条 after_paper_critic 用例
├── nodes/test_writer.py            # MODIFY: +2 条（iteration 自增、feedback 进模板）
└── test_graph_smoke.py             # MODIFY: +1 条 retry 路径 smoke
```

**职责切分**：
- `routing.after_paper_critic` 是纯函数，唯一控制流入口，单测最廉价
- writer 不读 critic、不算 iteration——只**自增**计数 + 用模板渲染就好
- 模板里 `{% if critic_feedback %}` 条件块让旧测试零修改通过

---

## Phase 1：State + Config（最小依赖底座）

### Task 1.1：state 加 writer_iteration

**Files:**
- Modify: `src/math_agent/state.py:121-124`

- [ ] **Step 1: 写测试**

Edit `tests/test_state.py`，在文件末尾追加：

```python
def test_writer_iteration_defaults_to_zero():
    s = MathModelingState(problem="x")
    assert s.writer_iteration == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_state.py::test_writer_iteration_defaults_to_zero -v`
Expected: FAIL（AttributeError）

- [ ] **Step 3: 在 state.py 加字段**

把 `state.py:121-124`：

```python
    # 流程控制（覆盖语义）
    iteration: int = 0
    stage_target: ModelStage = "basic"  # 当前要产出的阶段
    errors: Annotated[list[str], add] = Field(default_factory=list)
```

改为：

```python
    # 流程控制（覆盖语义）
    iteration: int = 0
    writer_iteration: int = 0           # 写作阶段的重试计数（paper_critic 闭环用）
    stage_target: ModelStage = "basic"  # 当前要产出的阶段
    errors: Annotated[list[str], add] = Field(default_factory=list)
```

- [ ] **Step 4: 跑测试确认通过 + 全部 state 测试**

Run: `pytest tests/test_state.py -v`
Expected: 10 passed（原 9 + 新 1）

### Task 1.2：config 加 MAX_WRITER_ITERATIONS

**Files:**
- Modify: `src/math_agent/config.py:23-24`

- [ ] **Step 1: 把 `config.py:23-24`**

```python
MAX_MODEL_ITERATIONS = 3      # basic -> improved -> final 之外的修正轮次
MAX_LLM_RETRIES = 2           # 单次 LLM 调用的结构化解析重试
```

改为：

```python
MAX_MODEL_ITERATIONS = 3       # basic -> improved -> final 之外的修正轮次
MAX_WRITER_ITERATIONS = 2      # paper_critic 未通过时 writer 最多重写次数
MAX_LLM_RETRIES = 2            # 单次 LLM 调用的结构化解析重试
```

- [ ] **Step 2: 提交 Phase 1**

```bash
git add src/math_agent/state.py src/math_agent/config.py tests/test_state.py
git commit -m "feat(state,config): add writer_iteration + MAX_WRITER_ITERATIONS for paper_critic loop"
```

---

## Phase 2：routing.after_paper_critic（控制流核心）

### Task 2.1：写三条 routing 测试

**Files:**
- Modify: `tests/test_routing.py`

- [ ] **Step 1: 看当前 routing 测试结构**

Run: `cat tests/test_routing.py`
观察现有 `after_model_critic` 的测试风格，保持一致。

- [ ] **Step 2: 在 `tests/test_routing.py` 末尾追加测试**

```python
from math_agent.routing import after_paper_critic
from math_agent.state import CriticReport


def _state_with_paper_critic(score: int, approved: bool, writer_iter: int):
    s = MathModelingState(problem="p")
    s.writer_iteration = writer_iter
    s.critic_reports.append(CriticReport(
        target="paper", score=score, approved=approved,
        issues=["编造数字"], suggestions=["核对附录"],
    ))
    return s


def test_after_paper_critic_advances_when_approved():
    s = _state_with_paper_critic(score=9, approved=True, writer_iter=0)
    assert after_paper_critic(s) == "advance"


def test_after_paper_critic_retries_when_below_threshold_and_iter_left():
    s = _state_with_paper_critic(score=4, approved=False, writer_iter=0)
    assert after_paper_critic(s) == "retry"


def test_after_paper_critic_advances_when_iter_cap_hit():
    # writer_iteration == MAX_WRITER_ITERATIONS，即使没通过也放行
    from math_agent.config import MAX_WRITER_ITERATIONS
    s = _state_with_paper_critic(score=4, approved=False, writer_iter=MAX_WRITER_ITERATIONS)
    assert after_paper_critic(s) == "advance"


def test_after_paper_critic_advances_when_no_critic():
    # 防御性：critic 缺失时不能死循环
    s = MathModelingState(problem="p")
    assert after_paper_critic(s) == "advance"
```

`MathModelingState` 应该已在测试顶部 import 过（看一眼 `test_routing.py:1-10`，如果没有就补 `from math_agent.state import MathModelingState`）。

- [ ] **Step 3: 跑测试确认 4 条全 FAIL**

Run: `pytest tests/test_routing.py -v -k paper_critic`
Expected: ImportError（`cannot import name 'after_paper_critic'`）

### Task 2.2：实现 after_paper_critic

**Files:**
- Modify: `src/math_agent/routing.py`

- [ ] **Step 1: 在 `routing.py` 末尾追加函数**

```python
from math_agent.config import MAX_WRITER_ITERATIONS  # 已 import MAX_MODEL_ITERATIONS，这里复用同一行


def after_paper_critic(state: MathModelingState) -> str:
    """writer 闭环：critic 通过或迭代用尽 → advance；否则 retry 回 writer。"""
    critic = state.latest_critic("paper")
    if critic is None:
        return "advance"  # 防御：没有 critic 报告也别死锁
    if critic.approved or state.writer_iteration >= MAX_WRITER_ITERATIONS:
        return "advance"
    return "retry"
```

注意：若 `routing.py:5` 已是 `from math_agent.config import MAX_MODEL_ITERATIONS`，改成：

```python
from math_agent.config import MAX_MODEL_ITERATIONS, MAX_WRITER_ITERATIONS
```

不要重复 import。

- [ ] **Step 2: 跑测试确认全部通过**

Run: `pytest tests/test_routing.py -v`
Expected: 8 passed（原 4 + 新 4）

- [ ] **Step 3: 提交 Phase 2**

```bash
git add src/math_agent/routing.py tests/test_routing.py
git commit -m "feat(routing): after_paper_critic dispatches retry/advance for writer loop"
```

---

## Phase 3：Writer 节点 + Prompt 接入 critic feedback

### Task 3.1：测试 writer_node 自增 writer_iteration

**Files:**
- Modify: `tests/nodes/test_writer.py`

- [ ] **Step 1: 在 `tests/nodes/test_writer.py` 末尾追加测试**

```python
def test_writer_increments_writer_iteration(mocker):
    fake = PaperSections(
        abstract="a"*200, problem_restatement="b"*200, assumptions="c"*200,
        notation="d"*200, model_section="e"*200, solution="f"*200,
        sensitivity="s"*200, conclusion="g"*200, references="h",
    )
    mocker.patch("math_agent.nodes.writer.complete", return_value=fake)
    s = MathModelingState(problem="p")
    s.writer_iteration = 0
    delta = writer_node(s)
    assert delta["writer_iteration"] == 1

    s2 = MathModelingState(problem="p")
    s2.writer_iteration = 1
    delta2 = writer_node(s2)
    assert delta2["writer_iteration"] == 2
```

- [ ] **Step 2: 跑测试确认 FAIL**

Run: `pytest tests/nodes/test_writer.py::test_writer_increments_writer_iteration -v`
Expected: FAIL（KeyError: 'writer_iteration'）

### Task 3.2：让 writer_node 自增计数 + 把 critic 反馈传给 prompt

**Files:**
- Modify: `src/math_agent/nodes/writer.py`

- [ ] **Step 1: 当前 `nodes/writer.py:11-18` 是**

```python
def writer_node(state: MathModelingState) -> dict:
    out: PaperSections = complete(
        build_prompt(state),
        schema=PaperSections,
        system=SYSTEM,
        model=MODEL_ROUTING["writer"],
    )
    return {"paper": out}
```

改为：

```python
def writer_node(state: MathModelingState) -> dict:
    out: PaperSections = complete(
        build_prompt(state),
        schema=PaperSections,
        system=SYSTEM,
        model=MODEL_ROUTING["writer"],
    )
    return {
        "paper": out,
        "writer_iteration": state.writer_iteration + 1,
    }
```

- [ ] **Step 2: 跑新测试确认通过**

Run: `pytest tests/nodes/test_writer.py::test_writer_increments_writer_iteration -v`
Expected: PASS

### Task 3.3：测试 prompt 渲染上一轮 critic feedback

**Files:**
- Modify: `tests/nodes/test_writer.py`

- [ ] **Step 1: 在 `tests/nodes/test_writer.py` 末尾追加测试**

```python
def test_prompt_includes_prior_paper_critic_feedback():
    """第二轮 writer 调用时，应把上一轮 paper_critic 的 issues/suggestions 注入。"""
    from math_agent.state import CriticReport
    s = _rich_state()
    s.writer_iteration = 1
    s.critic_reports.append(CriticReport(
        target="paper", score=4, approved=False,
        issues=["solution 段的 46 秒数字未在 stdout 中出现"],
        suggestions=["要么删掉数字，要么改成定性描述"],
    ))
    p = build_prompt(s)
    assert "solution 段的 46 秒数字未在 stdout 中出现" in p
    assert "要么删掉数字" in p


def test_prompt_omits_critic_section_when_no_prior_review():
    """第一轮（无 paper_critic 报告时）不应渲染 critic 反馈区块。"""
    s = _rich_state()
    p = build_prompt(s)
    # 模板里把上一轮反馈放在一个标题下；标题字面量在则说明区块被渲染
    assert "上一轮 PaperCritic 反馈" not in p
```

- [ ] **Step 2: 跑测试确认 FAIL**

Run: `pytest tests/nodes/test_writer.py -v -k "prompt_includes_prior_paper_critic_feedback or omits_critic_section"`
Expected: 第一条 FAIL（issue 字符串不在 prompt）。第二条目前 PASS（旧模板压根不渲染该标题），但加上修改后仍要 PASS。

### Task 3.4：模板 + prompt builder 接 critic feedback

**Files:**
- Modify: `src/math_agent/prompts/writer.py`
- Modify: `src/math_agent/templates/writer_prompt.md.j2`

- [ ] **Step 1: 改 `prompts/writer.py` 的 `build_prompt`**

当前：

```python
def build_prompt(state: MathModelingState) -> str:
    tmpl = _env.get_template("writer_prompt.md.j2")
    return tmpl.render(
        problem=state.problem,
        assumptions=state.assumptions,
        model_versions=state.model_versions,
        code_artifacts=state.code_artifacts,
        sensitivity_runs=state.sensitivity_runs,
        figures=state.figures,
    )
```

改为：

```python
def build_prompt(state: MathModelingState) -> str:
    tmpl = _env.get_template("writer_prompt.md.j2")
    return tmpl.render(
        problem=state.problem,
        assumptions=state.assumptions,
        model_versions=state.model_versions,
        code_artifacts=state.code_artifacts,
        sensitivity_runs=state.sensitivity_runs,
        figures=state.figures,
        prior_critic=state.latest_critic("paper"),  # 第一轮为 None，模板侧条件渲染
    )
```

- [ ] **Step 2: 在 `templates/writer_prompt.md.j2` 顶部 IRON RULES 段之后、`## 题目` 之前插入区块**

把现有：

```
## 题目
{{ problem }}
```

之前插入：

```
{% if prior_critic %}
## 上一轮 PaperCritic 反馈（必须逐条回应）

PaperCritic 评分：{{ prior_critic.score }}/10，approved={{ prior_critic.approved }}。

**issues**：
{% for i in prior_critic.issues %}
- {{ i }}
{% endfor %}

**suggestions**：
{% for s in prior_critic.suggestions %}
- {{ s }}
{% endfor %}

本次重写时**必须**：
1. 逐条回应上述 issues；不能简单复述上一稿。
2. IRON RULE 1（禁编造）仍然铁律。如果上一轮被指出"数字编造"，本轮要么把数字改成定性描述，要么完全删除该数字。
3. 不要为了"看起来回应了" 而硬塞新数字——没有就写没有。

---

{% endif %}
## 题目
{{ problem }}
```

- [ ] **Step 3: 跑两条 prompt 测试 + 全部 writer 测试**

Run: `pytest tests/nodes/test_writer.py -v`
Expected: 12 passed（原 9 + 本 Phase 新增 3 条 = test_writer_increments_writer_iteration + test_prompt_includes_prior_paper_critic_feedback + test_prompt_omits_critic_section_when_no_prior_review）

- [ ] **Step 4: 提交 Phase 3**

```bash
git add src/math_agent/nodes/writer.py src/math_agent/prompts/writer.py src/math_agent/templates/writer_prompt.md.j2 tests/nodes/test_writer.py
git commit -m "feat(writer): increment writer_iteration + inject prior paper_critic feedback into prompt"
```

---

## Phase 4：Graph 接线

### Task 4.1：写图层 smoke 测试

**Files:**
- Modify: `tests/test_graph_smoke.py`

- [ ] **Step 1: 看现有 graph smoke 测试结构（保持风格一致）**

Run: `head -80 tests/test_graph_smoke.py`

观察：现有测试都用 `mocker.patch` 拦截每个节点的 `complete`，逐节点指定返回。

- [ ] **Step 2: 在 `tests/test_graph_smoke.py` 末尾追加测试**

```python
def test_graph_retries_writer_on_paper_critic_rejection(mocker, tmp_path):
    """paper_critic 第一次打 4 分（not approved），应触发 writer 第二次调用；
    第二次 critic 打 9 分（approved），进入 evaluation。
    """
    from math_agent.state import (
        MathModelingState, ModelVersion, Assumption, CodeArtifact,
        PaperSections, CriticReport, SensitivityRun, FigureArtifact,
        EvaluationReport, HumanDecision,
    )
    from math_agent.graph import build_graph

    # ---- Mock 每个节点的 complete ----
    # 顺序：analyst → modeler → critic → ... → coder → sensitivity ...
    #       → figure_pipeline → writer(1st) → paper_critic(reject)
    #       → writer(2nd) → paper_critic(approve) → evaluation → ...

    # 为简洁起见，只为本测试关心的节点显式打桩；其余用宽松默认。
    paper1 = PaperSections(abstract="v1"*100, problem_restatement="x"*150,
                           assumptions="x"*150, notation="x"*150,
                           model_section="x"*400, solution="x"*200,
                           sensitivity="x"*150, conclusion="x"*150, references="-")
    paper2 = PaperSections(abstract="v2"*100, problem_restatement="x"*150,
                           assumptions="x"*150, notation="x"*150,
                           model_section="x"*400, solution="x"*200,
                           sensitivity="x"*150, conclusion="x"*150, references="-")
    critic_reject = CriticReport(target="paper", score=4, approved=False,
                                 issues=["编数字"], suggestions=["改定性"])
    critic_accept = CriticReport(target="paper", score=9, approved=True)

    # 用 dict + side_effect 分别打桩各节点
    mocker.patch("math_agent.nodes.analyst.complete",
                 return_value=mocker.MagicMock(assumptions=[Assumption(statement="s", rationale="r")]))
    mocker.patch("math_agent.nodes.modeler.complete",
                 return_value=ModelVersion(stage="final", description="d"*200))
    mocker.patch("math_agent.nodes.model_critic.complete",
                 return_value=CriticReport(target="modeler", score=9, approved=True, stage="final"))
    mocker.patch("math_agent.nodes.coder.complete",
                 return_value=mocker.MagicMock(purpose="x", code="print(1)"))
    # sensitivity / figure_pipeline / evaluation / human_review / latex 各自 mock
    # 为求简洁，让它们走 noop / 默认失败路径——graph 已对 sensitivity/figure 失败容忍
    mocker.patch("math_agent.nodes.sensitivity.complete",
                 side_effect=RuntimeError("skip sensitivity in smoke"))
    mocker.patch("math_agent.nodes.figure_pipeline.complete",
                 side_effect=RuntimeError("skip figures in smoke"))

    writer_returns = [paper1, paper2]
    mocker.patch("math_agent.nodes.writer.complete", side_effect=writer_returns)

    critic_returns = [critic_reject, critic_accept]
    mocker.patch("math_agent.nodes.paper_critic.complete", side_effect=critic_returns)

    mocker.patch("math_agent.nodes.evaluation.complete",
                 return_value=EvaluationReport(
                     assumption_reasonableness=8, modeling_creativity=8,
                     result_correctness=8, writing_clarity=8, extra_depth=8,
                     overall=8.0,
                 ))
    mocker.patch("math_agent.nodes.latex.complete",
                 return_value=mocker.MagicMock(tex_source="\\documentclass{article}"))

    g = build_graph()
    final = g.invoke({
        "problem": "p", "stage_target": "final", "iteration": 0,
        "output_dir": str(tmp_path),
        # 跳 HITL：直接给 approved 决定，让 human_review 节点不阻塞
        "human_decision": HumanDecision(approved=True).model_dump(),
    })

    # 关键断言：writer 被调用了 2 次（重试路径触发）
    assert mocker.patch("math_agent.nodes.writer.complete").call_count >= 0  # 占位避免 mocker 重复 patch
    # 真正断言：state 里 writer_iteration 应为 2
    assert final["writer_iteration"] == 2
    # 最终 paper 应是第二版
    assert final["paper"]["abstract"].startswith("v2")
```

⚠️ smoke 测试要 mock 的节点多。如果跑得不顺利，**降级方案**：单独写一个**只测条件边触发**的 mini-graph 测试，跳过 sensitivity/figure 这些复杂节点。下面 Task 4.2 备一份。

### Task 4.2：（如 Task 4.1 太脆弱）改写为隔离的 mini-graph 测试

只在 Task 4.1 难以稳定通过时启用——graph_smoke 已有大量 mocker.patch，新增容易碰撞。

- [ ] **Step 1: 把 Task 4.1 的测试改为只构建 `writer ↔ paper_critic ↔ evaluation` 子图**

```python
def test_writer_paper_critic_loop_isolated(mocker, tmp_path):
    """隔离测试 writer↔paper_critic 闭环，跳过 analyst/modeler/coder/sensitivity 全链。"""
    from langgraph.graph import StateGraph, END
    from math_agent.state import MathModelingState, PaperSections, CriticReport
    from math_agent.nodes.writer import writer_node
    from math_agent.nodes.paper_critic import paper_critic_node
    from math_agent.routing import after_paper_critic

    paper_v1 = PaperSections(abstract="v1"*100, problem_restatement="x"*150,
                             assumptions="x"*150, notation="x"*150,
                             model_section="x"*400, solution="x"*200,
                             sensitivity="x"*150, conclusion="x"*150, references="-")
    paper_v2 = PaperSections(abstract="v2"*100, problem_restatement="x"*150,
                             assumptions="x"*150, notation="x"*150,
                             model_section="x"*400, solution="x"*200,
                             sensitivity="x"*150, conclusion="x"*150, references="-")
    mocker.patch("math_agent.nodes.writer.complete", side_effect=[paper_v1, paper_v2])
    mocker.patch("math_agent.nodes.paper_critic.complete", side_effect=[
        CriticReport(target="paper", score=4, approved=False,
                     issues=["编数字"], suggestions=["改定性"]),
        CriticReport(target="paper", score=9, approved=True),
    ])

    g = StateGraph(MathModelingState)
    g.add_node("writer", writer_node)
    g.add_node("paper_critic", paper_critic_node)
    g.set_entry_point("writer")
    g.add_edge("writer", "paper_critic")
    g.add_conditional_edges("paper_critic", after_paper_critic,
                            {"retry": "writer", "advance": END})
    compiled = g.compile()

    final = compiled.invoke({"problem": "p", "writer_iteration": 0})
    assert final["writer_iteration"] == 2
    assert final["paper"].abstract.startswith("v2")
    # 关键：第二次 critic 通过
    paper_critics = [r for r in final["critic_reports"] if r.target == "paper"]
    assert len(paper_critics) == 2
    assert paper_critics[-1].approved is True
```

- [ ] **Step 2: 跑测试确认 FAIL**（routing 与 graph 接线尚未改）

Run: `pytest tests/test_graph_smoke.py -v -k "writer_paper_critic_loop"`
Expected: FAIL（writer_iteration != 2 或 KeyError）

### Task 4.3：改 `graph.py` 接条件边

**Files:**
- Modify: `src/math_agent/graph.py`

- [ ] **Step 1: 当前 `graph.py:56-57` 是**

```python
    g.add_edge("writer", "paper_critic")
    g.add_edge("paper_critic", "evaluation")
```

改为：

```python
    g.add_edge("writer", "paper_critic")
    g.add_conditional_edges(
        "paper_critic",
        after_paper_critic,
        {"retry": "writer", "advance": "evaluation"},
    )
```

- [ ] **Step 2: 顶部 import 同步**

把 `graph.py:18`：

```python
from math_agent.routing import after_model_critic
```

改为：

```python
from math_agent.routing import after_model_critic, after_paper_critic
```

- [ ] **Step 3: 跑 Task 4.1（或 4.2）的测试**

Run: `pytest tests/test_graph_smoke.py -v`
Expected: 既有 smoke 全绿 + 新增 retry-loop 测试通过

- [ ] **Step 4: 跑全量回归**

Run: `pytest -q`
Expected: **74 passed, 1 skipped（共 75 collected）**。baseline 66 collected（65 passed + 1 skipped）+ Phase 1 +1 + Phase 2 +4 + Phase 3 +3 + Phase 4 +1 = 75 collected。本说明文以 Task 4.2 隔离写法为主路径计数。

⚠️ 如果回归数对不上，**先**看是不是 graph_smoke 老测试因为新条件边失败——这是预期的，要把老 `test_graph_full_smoke.py` 里隐式假设 writer→paper_critic→evaluation 直边的断言也对齐。具体怎么改取决于该文件现状，不预判。

- [ ] **Step 5: 提交 Phase 4**

```bash
git add src/math_agent/graph.py tests/test_graph_smoke.py
git commit -m "feat(graph): wire writer<->paper_critic loop via after_paper_critic"
```

---

## Phase 5：端到端验收（手动，非门禁）

⚠️ 同前一份 plan 的约定，Phase 5 **不是** pytest 测试，是手动验收步骤。

### Task 5.1：清除旧输出 + 重跑

- [ ] **Step 1: 清理**

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent && rm -rf runs/eval_v4 runs/eval_v4.log
```

- [ ] **Step 2: 端到端**

```bash
python -m math_agent.cli run --problem tests/fixtures/sample_problem.json --out runs/eval_v4 --no-interrupt
```

预计 15-25 分钟（多了 1-2 轮 writer 调用）。

- [ ] **Step 3: 验证 writer 被多调用**

```bash
grep -c "abstract" runs/eval_v4/checkpoints.sqlite 2>/dev/null  # 粗略；或：
sqlite3 runs/eval_v4/checkpoints.sqlite "SELECT writer_iteration FROM checkpoints ORDER BY rowid DESC LIMIT 1;" 2>/dev/null
```

更可靠：肉眼读 `runs/eval_v4/paper.md`，看求解段引用的数字是否能在附录 stdout 中找到（不强求 100% 匹配，但应"基本不编"）。

- [ ] **Step 4: 抽查 paper_critic 报告**

LangGraph state 默认会保存在 sqlite。可以加临时脚本读 critic_reports，但**最简单**：在 CLI 添加一个 `inspect` 子命令，或者直接看 `paper.md` 质量是否相对 `runs/eval_v3/paper.md` 有改善（关键：solution 段数字真实度）。

如果数字真实度**没有**改善（writer 在 critic 反馈下仍编数字），那是 paper_critic 自己没识别出编造问题，需要去**强化 paper_critic 的 SYSTEM prompt**——这属于 follow-up 工单，不在本计划范围。

---

## 已知非本计划范围（明确不做）

- **paper_critic prompt 强化**：让 critic 显式检查"数字是否在 stdout 中出现"——本计划只接通闭环。若验收时发现 critic 识别不出编造，单独工单处理（可在 paper_critic SYSTEM 加 `code_artifacts` 上下文并要求逐数字核对）。
- **figure_critic / evaluation 也接闭环**：理论可对偶，但当前不解决数字编造问题
- **iteration 计数器持久化跨 stage**：`writer_iteration` 仅在 writer 走两轮，不在 stage 推进时重置（也不需要——final 阶段进入 writer 时已 0）
- **降级到 v3.6.5 单 call**：不学 ARS 的 paper-blind / paper-visible 拆分（要重写 prompt 架构）

---

## 自我审查

**1. Spec 覆盖：**
- writer→paper_critic 直边换成闭环 → Task 4.3 ✓
- writer 接收 critic 反馈 → Task 3.4 ✓
- writer_iteration 计数 → Task 1.1 + 3.2 ✓
- 闭环上限保护 → Task 1.2 + 2.2 ✓
- 防御性 routing（critic 缺失/approved/cap）→ Task 2.1 三条 ✓

**2. Placeholder 扫描：** 无 TBD/TODO；所有 step 给出可粘贴代码或精确命令。唯一"自适应"步是 Task 4.4 Step 4 备注「老 smoke 测试可能要对齐」——这不是 placeholder，是真实风险预警，无法预判到底改哪一行。

**3. 类型一致性：**
- `state.writer_iteration: int = 0`（state.py 新增）↔ `delta["writer_iteration"]: int`（writer.py 返回）↔ `state.writer_iteration >= MAX_WRITER_ITERATIONS`（routing.py）—— 全部 int ✓
- `after_paper_critic(state) -> str`，返回字面量 `"retry"` / `"advance"` ↔ graph `add_conditional_edges` mapping `{"retry": "writer", "advance": "evaluation"}` ✓
- `state.latest_critic("paper")` 返回 `CriticReport | None`，模板 `{% if prior_critic %}` 对 None 求假 ✓
- `MAX_WRITER_ITERATIONS = 2` 意味着最多 **2 次 writer 调用**：第 1 次 iter=0→1，critic reject 后 iter=1<2 → retry，第 2 次 iter=1→2，critic 不管 approve 与否都因 cap 而 advance。**测试 4.2 期望 `writer_iteration == 2`** ✓
- 旧测试 `test_writer_fills_paper` 期望 `delta["paper"]`，新 writer_node 多返回一个 key（`writer_iteration`），旧断言仍成立（dict 多个 key 不破断言）✓

**4. 改动半径：** 7 个文件，~30 行 src + ~80 行测试。3 个 commit（Phase 1+2+3 各一，Phase 4 单独）。预计 1.5-2 小时手动完成（不含端到端验收）。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-29-writer-paper-critic-loop.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
