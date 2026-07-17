# 数学建模多智能体系统 — 内容深度突破层实现计划（Plan D）

> **历史方案记录（非现行事实源）**：≥30 页、≥10 图是阶段目标，不代表当前真题产物已达到；现行状态见 [`../../README.md`](../../README.md)。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **状态：DRAFT — 关键架构决策点已标出，需在执行前 brainstorm 拍板。**

**Goal:** 把当前单次 LLM 调用产出的 ~16 页 / 1-2 张图 / 扁平方程列表的论文产出，提升到 ≥ 30 页 / ≥ 10 张图 / 多步推导 / 真实文献的国一级别。

**Architecture:** 把"一次性梭哈"的 `writer_node` / `coder_node` / `modeler_node` 拆为**多次 LLM 调用**——writer 按章节生成、coder 按图任务多次调用、modeler 输出推导链而非方程列表。Plan D 不动 Plan B 的 graph 拓扑（analyst → modeler ⇄ critic → coder → sensitivity → figure_pipeline → writer ⇄ paper_critic → evaluation → human_review → latex），只**重构这 3 个节点的内部实现**，让每个节点在其内部多次调用 LLM 完成深度产出。

**Tech Stack:** 不引入新依赖。Plan A/B/C 的栈不变：LangGraph / Pydantic / LiteLLM / jinja2 / pytest-mock。

> **前置条件**：Plan A/B 已完成且 `pytest -q` 全绿。**Plan C 已完成 ✅**（2026-07-02 验证：
> `retry.py` / `tracing.py` / `rag/` 全部落地，`SqliteSaver` 已接入 cli `run`/`resume`，
> `retrieved_context` 参数已注入 analyst / modeler / writer 的 `build_prompt` 签名与节点入口）。
> Plan D Phase 2 拆 writer per-group 时**直接沿用 `retrieved_context` 参数**到每个 group
> 模板，不需要重复注入。
>
> 所有 5 个决策点（#1 writer 粒度、#2 coder 模式、#3 modeler 推导链、#4 references 来源、
> #5 失败回滚）均可从默认推荐值开始执行，无需等待任何外部依赖。

---

## ⚠️ 待 brainstorm 决策点（执行前必拍板）

### 决策 #1：writer 拆章节的粒度

| 选项 | LLM 调用数 | 单次 token | 总成本 vs 当前 | 风险 |
|---|---|---|---|---|
| **A. 9 段独立调用**（abstract / problem_restatement / assumptions / notation / model_section / solution / sensitivity / conclusion / references）| 9 | 1-2k | 4-5x | 段间风格漂移、abstract 写时其它段还没写 |
| **B. 4 大块**（abstract + problem / model + solution / sensitivity + conclusion / references）| 4 | 2-4k | 2-3x | 中等 |
| **C. 阶段化**（pass 1 写大纲 → pass 2 按大纲逐段填充）| 1 + 9 = 10 | 1-2k | 5-6x | 最深但最贵 |

**默认推荐**：**C（阶段化：大纲 + 逐段填充）**。Plan D 的目的是从国二到国一，质量优先于成本。
pass1 全局大纲保证章节间逻辑连贯；pass2 每段单独聚焦写深。多花 6 次调用买的是论文骨架
完整性——abstract 写时有 model_section 大纲可引，不会泛泛而谈。如果只要国二品质，选 B 省一半。

### 决策 #2：coder 多任务模式怎么定

| 选项 | 实现 |
|---|---|
| **A. 显式 figure plan**：modeler 输出 `final_model` 时附带 `figure_purposes: list[str]`（如 "需求时序图 / 调度路径图 / 成本构成饼图"），coder 按 list 跑 N 次（每次产 1 张图）| 干净；要改 modeler schema |
| **B. coder 自决**：单次 coder prompt 要求"输出 5-8 个独立可运行的代码段，每段产一张图"，节点对每段独立 sandbox 执行 | 实现简单；图数量不可控 |
| **C. plan + execute**：coder pre-pass 产 figure_plan，再 main-pass 按计划逐图调 LLM | 最稳但最贵 |

**默认推荐**：A。理由：modeler 已经有"分阶段输出"模式，把 figure_purposes 加上去 schema 改动最小。

### 决策 #3：modeler 推导链结构

modeler 当前输出 `ModelVersion(stage, description, equations: list[str], variables: dict, notes)`。优秀论文的推导链像：

```
1. 动机：为什么用 SARIMAX
2. 数学陈述：模型族 $y_t = \sum...$
3. 参数估计：MLE / 矩估计
4. 约束推导：定常性 → 差分阶数
5. 等价变换：Markov 形式
6. 求解：Kalman 滤波
```

**架构选项**：

- **A. 加 `derivation_steps: list[DerivationStep]` 字段到 ModelVersion**，每个 step 单独 LLM 调用产出
- **B. 不动 schema，让 modeler prompt 输出 description 时按章节写**（最 ponytail，但 LLM 自律差）
- **C. 新节点 `derivation_node` 插在 modeler 之后**

**默认推荐**：A。每 ModelVersion 增加 LLM 调用次数：**只对 `stage="final"` 加 derivation**，
其余阶段不动 —— 6 步 × 1 阶段 + 1 次 self-consistency = **7 次额外调用**（不是 18 次；basic/improved 阶段维持单次产出）。

### 决策 #4：references 真实化路径

| 选项 | 实现 | 依赖 |
|---|---|---|
| **A. 让 writer 凭训练数据"想"** | prompt 加 "5-15 篇真实文献，年份 ≥ 2010" | LLM 会编 DOI 和卷期号 → **不可接受** |
| **B. RAG 检索（Plan C 范围）** | ingest 历年优秀论文 + Google Scholar 数据集 → writer 引 | 依赖 Plan C 完成 |
| **C. 提供静态 references 库**（手工准备 20 篇 seed，覆盖主流方法论；扩到 200 篇为独立工单）| 一个 json 文件，writer 按 problem_domains 筛选 | 工作量 1-2 小时（20 篇）|
| **D. Semantic Scholar API 实时检索** | writer 按题目关键词调 `api.semanticscholar.org`，返回真实 DOI/卷期号 | 运行时联网（coder 已需联网装包，合理假设）|

**默认推荐**：**D（Semantic Scholar API）**。在所有维度严格优于 C：
- **真实性**：API 返回验证过的元数据，与手工核对同等可靠
- **相关性**：按题目关键词实时检索，远优于 20 篇静态库的覆盖
- **覆盖广度**：千万篇 vs 20 篇
- **维护成本**：零（30 行代码）vs 手工填 + 核对

C 作为离线 fallback 保留（网络不可用时降级），但默认走 D。
长期若需要"历年国一论文"特定语料，叠加 B（RAG）。

### 决策 #5：失败回滚策略

Plan D 后单次跑 50-80 分钟 / API 成本 $5-15。一个节点崩了重跑代价大。

- **A. 每章节独立可恢复**（writer 段落级 checkpoint，需要在 writer 内部多打 checkpoint）
- **B. 复用 Plan B 的 SqliteSaver**（graph 已经在每个节点末尾 commit checkpoint；崩了从最后 commit 处恢复——**零额外工作**）
- **C. 全有全无**（崩了重跑）

**默认推荐**：B。Plan B 的 `SqliteSaver + interrupt_before` 基础设施已经提供节点级 checkpoint；
Plan D 跑长流水线时本来就该 `math-agent run --thread <id>`，崩了 `math-agent resume --thread <id>`
就从最后一个完成的节点继续。**不需要额外开发**。

A 选项只在"writer 一个节点跑 8 分钟、内部 4 次 LLM 调用、第 3 次崩了"这种细粒度场景才有意义；
在那之前 B 已经够用。

---

## 文件结构

新增（all paths relative to `build-agent/`）：

```
src/math_agent/
├── prompts/
│   ├── writer_section.py        # 各章节独立 prompt builder
│   ├── modeler_derivation.py    # 推导链 step prompt
│   ├── coder_figure_plan.py     # figure 任务规划 prompt
│   └── coder_figure_one.py      # 单图代码生成 prompt
├── nodes/
│   ├── writer.py                # 改：拆为多次 LLM 调用 + 章节级 critic loop（可选）
│   ├── modeler.py               # 改：先出 ModelVersion 骨架，再逐步推导填充
│   └── coder.py                 # 改：plan + 多次单图调用
├── state.py                     # 改：ModelVersion + DerivationStep；CodeArtifact 保留兼容
└── references/
    └── builtin_library.json     # 静态参考文献库（决策 #4 选 C 时启用）

tests/
├── nodes/
│   ├── test_writer_per_section.py
│   ├── test_modeler_derivation.py
│   └── test_coder_multi_figure.py
└── test_writer_full_smoke.py    # 端到端 mock 模式
```

修改：
- `src/math_agent/state.py` — 加 `DerivationStep` / `Reference` / `CriticIssue` 模型 + `ModelVersion.derivation_steps`、`figure_purposes`；`CriticReport.issues` 升级为 `list[CriticIssue]`（兼容旧 `list[str]`）
- `src/math_agent/prompts/writer.py` — 拆为 dispatcher + 单段 prompt
- `src/math_agent/templates/writer_prompt.md.j2` — 单段模板（每段一份）
- `src/math_agent/prompts/paper_critic.py` — prompt 输出结构化 `issues: [{section, problem}]`
- `src/math_agent/nodes/paper_critic.py` — 解析结构化 issues

---

## Phase 1：State schema 扩展（绑定决策后才执行）

> ⚠ **不能在 brainstorm 之前执行**。Phase 1 的字段（`figure_purposes` / `DerivationStep`）
> 仅在决策 #2 = A 与决策 #3 = A 时需要；其它分支会让这些字段成为死代码。
> Phase 1 在拍板后随对应 Phase 一起提交，不单独打底。

### Task 1.1：扩展 ModelVersion 加 figure_purposes（先写测试）

**Files:**
- Modify: `tests/test_state.py`
- Modify: `src/math_agent/state.py:26-32`

> ⚠ 决策 #2 = A 时执行；选 B / C 跳过（不需要 schema 改）。

- [ ] **Step 1：写测试**

```python
def test_model_version_has_figure_purposes_default_empty():
    from math_agent.state import ModelVersion
    m = ModelVersion(stage="final", description="d")
    assert m.figure_purposes == []
    m.figure_purposes.append("需求时序图")
    assert m.figure_purposes == ["需求时序图"]
```

- [ ] **Step 2：跑测试确认失败**

Run: `pytest tests/test_state.py::test_model_version_has_figure_purposes_default_empty -v`
Expected: AttributeError

- [ ] **Step 3：改 state.py**

把 `ModelVersion` 改为：

```python
class ModelVersion(BaseModel):
    stage: ModelStage
    description: str
    equations: list[str] = Field(default_factory=list)
    variables: dict[str, str] = Field(default_factory=dict)
    notes: str = ""
    figure_purposes: list[str] = Field(default_factory=list)  # Plan D：modeler 建议要画的图
    derivation_steps: list[DerivationStep] = Field(default_factory=list)
    derivation_notes: str = ""  # Plan D：self-consistency gate 产出的问题标注
```

- [ ] **Step 4：跑测试 + 全量回归**

Run: `pytest tests/test_state.py -v`
Expected: 全绿。

- [ ] **Step 5：commit**

```bash
git add src/math_agent/state.py tests/test_state.py
git commit -m "feat(state): ModelVersion.figure_purposes for coder multi-figure planning"
```

### Task 1.2：加 DerivationStep + ModelVersion.derivation_steps（先写测试）

**Files:**
- Modify: `tests/test_state.py`
- Modify: `src/math_agent/state.py`

> ⚠ 决策 #3 选 A 时执行；选 B 跳过。

- [ ] **Step 1：写测试**

```python
def test_derivation_step_carries_step_metadata():
    from math_agent.state import DerivationStep
    d = DerivationStep(
        title="参数估计",
        motivation="为何用 MLE",
        statement="对数似然 \\ell(\\theta)=...",
        result="\\hat\\theta = ...",
    )
    assert d.title == "参数估计"
    assert d.motivation.startswith("为何")
```

- [ ] **Step 2：跑测试确认失败**

Run: `pytest tests/test_state.py::test_derivation_step_carries_step_metadata -v`
Expected: ImportError

- [ ] **Step 3：改 state.py**

在 `ModelVersion` 之前加：

```python
class DerivationStep(BaseModel):
    """模型推导链中的一步（动机 → 数学陈述 → 结果）。"""
    title: str                       # "参数估计" / "约束推导" / "等价变换"
    motivation: str                  # 为何做这步
    statement: str                   # 数学陈述（含 inline LaTeX）
    result: str = ""                 # 推导结论
```

然后 `ModelVersion` 加：

```python
    derivation_steps: list[DerivationStep] = Field(default_factory=list)
```

- [ ] **Step 4：跑测试**

Run: `pytest tests/test_state.py -v`
Expected: 全绿。

- [ ] **Step 5：commit**

```bash
git add src/math_agent/state.py tests/test_state.py
git commit -m "feat(state): DerivationStep + ModelVersion.derivation_steps for derivation chain"
```

### Task 1.3：升级 CriticReport.issues 为结构化 CriticIssue

**Files:**
- Modify: `tests/test_state.py`
- Modify: `src/math_agent/state.py`
- Modify: `src/math_agent/prompts/paper_critic.py`
- Modify: `src/math_agent/nodes/paper_critic.py`
- Modify: `tests/nodes/test_paper_critic.py`
- **Audit:** `tests/test_graph_smoke.py`（构造 `CriticReport(issues=[...])` 的地方）
- **Audit:** `tests/bench/conftest.py`（L129 bench mock 里构造 `CriticReport(issues=["Looks good"])`）
- **Grep:** 全代码库 `CriticReport(` 和 `.issues` ——所有构造/读取处均需适配

> **背景**：Plan D Task 2.3 的 `_sections_to_rewrite` 需要 paper_critic 输出结构化的
> `section` 字段才能精准定位要重写的章节。当前 `CriticReport.issues: list[str]` 是纯文本，
> 靠正则匹配太脆弱（"模型"二字几乎一定出现）。
>
> **破坏性变更声明**：`CriticReport.issues` 从 `list[str]` → `list[CriticIssue]` 会
> 导致所有构造 `CriticReport(issues=["xxx"])` 的测试炸裂。修复策略：**方案 2（显式——
> 旧测试改为构造 `CriticIssue(section="general", problem="xxx")`）**，不采用方案 1
> （魔法单字符串转换会破坏类型系统保护）。受影响文件除上列外，需 `rg "CriticReport\("` 
> 和 `rg "\.issues"` 全代码库确认无遗漏。

- [ ] **Step 1：在 `state.py` 加 `CriticIssue` 模型**

```python
class CriticIssue(BaseModel):
    """结构化评审意见，section 字段限定到固定 enum。"""
    section: Literal[
        "abstract", "problem_restatement", "assumptions", "notation",
        "model_section", "solution", "sensitivity", "conclusion", "references", "general",
    ] = "general"
    problem: str
```

把 `CriticReport.issues` 从 `list[str]` 改为 `list[CriticIssue]`。

> **兼容旧测试**：Plan A/B 中所有 `CriticReport(issues=["xxx"])` 构造均会报类型错误。
> **采用方案 2（显式）**——旧测试改为：
> `CriticReport(issues=[CriticIssue(section="general", problem="xxx")])`。
> 不采用方案 1（魔法单字符串转换），因为会破坏 Pydantic 类型验证的保护作用。
>
> **全代码库审计清单**（执行者需逐项确认）：
> - [ ] `tests/nodes/test_paper_critic.py` — 构造 `CriticReport` 的测试
> - [ ] `tests/test_graph_smoke.py` — mock critic `issues` 的 smoke 测试
> - [ ] `tests/bench/conftest.py` L129 — `install_bench_mocks` 中 `CriticReport(issues=["Looks good"])`
> - [ ] `src/math_agent/graph.py` 路由 / `routing.py` — 读 `critic.issues` 的代码
> - [ ] 其他 `.py` 文件中 `rg "CriticReport\("` 和 `rg "\.issues"` 命中的所有引用处

- [ ] **Step 2：改 paper_critic prompt 输出结构化 issues**

prompt 的 JSON schema 改为：
```
"issues": [{"section": "abstract|model_section|...", "problem": str}, ...]
```

- [ ] **Step 3：跑全测**

Run: `pytest tests/test_state.py tests/nodes/test_paper_critic.py -v`
Expected: 全绿（含兼容后的旧测试）。

- [ ] **Step 4：commit**

```bash
git add src/math_agent/state.py tests/test_state.py
git commit -m "feat(state): DerivationStep + ModelVersion.derivation_steps for derivation chain"
```

---

## Phase 2：writer 拆章节生成（核心改动）

> ⚠ 决策 #1 拍板才能写完整 task 内容。下面假设决策 = C（pass1 大纲 + pass2 逐段填充）。
>
> 决策 = A（9 段）：把 `_WRITER_SECTIONS` 改为 9 个单段，省掉 outline pass；其余结构相同。
> 决策 = B（4 大块）：把 `_WRITER_SECTIONS` 改为 4 组，省掉 outline pass。

### Task 2.1：拆 writer prompt 模板

**Files:**
- Create: `src/math_agent/templates/writer_outline.md.j2`（pass1：全局大纲，含每章主旨 + 章间衔接）
- Create: `src/math_agent/templates/writer_section_abstract_problem.md.j2`（摘要 + 问题重述 + **keywords**）
- Create: `src/math_agent/templates/writer_section_assumptions_notation.md.j2`（假设 + 符号说明）
- Create: `src/math_agent/templates/writer_section_model.md.j2`（模型建立，含 derivation_steps 渲染）
- Create: `src/math_agent/templates/writer_section_solution.md.j2`（模型求解）
- Create: `src/math_agent/templates/writer_section_sensitivity.md.j2`（敏感性分析）
- Create: `src/math_agent/templates/writer_section_conclusion.md.j2`（结论）
- Create: `src/math_agent/templates/writer_section_references.md.j2`（参考文献）

pass1 大纲模板输出一份 `WriterOutline`（每章主旨 2-3 句 + 章间衔接 1 句），存入 state。
pass2 每段模板接收 outline 中对应章节的主旨作为锚点，输出该段文本。

每份模板的核心结构：保留现有 `writer_prompt.md.j2` 的 IRON RULES 1-6 头部，但 JSON schema 收窄到只输出本组段落字段。

> **paper.keywords 字段归属**：Plan B 后期 hotfix 已加 `paper.keywords: list[str]` 字段；
> Plan D 拆 per-group 时 keywords 与 abstract / problem_restatement 一起在
> `writer_section_abstract_problem.md.j2` 输出（schema 加 `"keywords": [str, ...]` 键，3-6 个）。
> 其他 group 模板不动 keywords。

> 此 task 的 4 份模板正文细节，等决策 #1 拍板后展开。需展示每份模板的 prompt 内容 + 模板使用的 Jinja2 变量列表。

- [ ] **Step 1-N**：分模板逐个测试 + 实现 + commit。

### Task 2.2：writer_node 改为 per-group LLM 调用

**Files:**
- Modify: `src/math_agent/nodes/writer.py`
- Modify: `tests/nodes/test_writer_per_section.py`（新建）

- [ ] **Step 1：写测试** —— 等决策 #1 拍板后展开具体测试代码。骨架（决策 C = 1 outline + 7 section 调用）：

```python
def test_writer_makes_outline_then_per_section_calls(mocker):
    from math_agent.nodes.writer import writer_node, _WRITER_SECTIONS
    call_count = {"n": 0}
    def _fake(prompt, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return WriterOutline(...)  # pass1 大纲
        return ...  # pass2 各段 per-section schema 输出
    mocker.patch("math_agent.nodes.writer.complete", side_effect=_fake)
    writer_node(_rich_state())
    # 1 outline + len(_WRITER_SECTIONS) 次 section 调用
    assert call_count["n"] == 1 + len(_WRITER_SECTIONS)
```

- [ ] **Step 2-N**：逐 group 实现 + 合并到完整 PaperSections。

### Task 2.3：合并多 group 输出 → PaperSections

- [ ] **Step 1**：把每个 group 的 LLM 输出 dict 合并入同一个 PaperSections 实例。
- [ ] **Step 2**：保留 `writer_iteration` 自增逻辑（Plan B 已落地的 writer↔paper_critic 闭环）。
- [ ] **Step 3**：**critic loop 第二轮只重写 paper_critic.issues 命中的 section**——避免把 7 段都重跑一遍：

> **前置依赖**：`_WRITER_SECTIONS` 常量在 Phase 2 Task 2.2 中创建（其具体值取决于决策 #1）。
> 本 task 只负责引用它做 section 名匹配 `_sections_to_rewrite`。执行顺序：Task 2.2 → Task 2.3。
> 以下代码假设 `_WRITER_SECTIONS` 已定义。

```python
def _sections_to_rewrite(issues: list[CriticIssue]) -> list[str]:
    """PaperCritic 输出结构化 issues（每条含 section 字段），直接按 section 过滤。
    无 section 字段或 section 未命中已知列表时，回退到全量重写。"""
    known = {s for s in _WRITER_SECTIONS}
    hit = {i.section for i in issues if i.section in known}
    return list(hit) or list(_WRITER_SECTIONS)
```

> **前置改动**：PaperCritic 的 `CriticReport.issues` 需要从 `list[str]` 升级为
> `list[CriticIssue]`，其中 `CriticIssue(section: str, problem: str)`。`section` 限定到
> 固定 enum（`abstract / problem_restatement / assumptions / notation / model_section /
> solution / sensitivity / conclusion / references`）。这个 schema 改动列入 Plan D 范围
> （Phase 3 Task 3.1 PaperCritic prompt 同步更新）。**Plan A/B 的旧 `issues: list[str]`
> 测试需要兼容**：CriticIssue 加 `__init__` 接受单字符串，或旧测试改为构造 CriticIssue。

writer_node 第二轮（critic loop iteration > 0）只对 `_sections_to_rewrite(critic.issues)` 跑 LLM；
其余 section 直接从 state.paper 继承。outline pass 不重跑（大纲第一轮已确定，critic 只评正文）。

- [ ] **Step 4**：测试覆盖"第二轮只调对应 section LLM、其他 section 保留第一轮文本"。
- [ ] **Step 5**：commit。

> **预算估算**（决策 C）：pass1 大纲 1 次 + pass2 七段 7 次 = 8 次/轮。critic loop 第二轮
> 通常只重写 1-2 section，实际 = 8 + 1~2 = 9~10 次 LLM 调用/ writer 节点。

---

## Phase 3：coder 多任务模式（决策 #2）

> ⚠ 决策 #2 拍板才能写完整 task。下面假设决策 = A。

### Task 3.1：modeler 输出 figure_purposes

**Files:**
- Modify: `src/math_agent/prompts/modeler.py`
- Modify: `src/math_agent/nodes/modeler.py`

- [ ] **Step 1：改 modeler prompt schema**

让最终阶段（stage="final"）的 modeler 输出附带 `figure_purposes: list[str]`，5-10 个图任务，例如 `["需求时序图", "调度路径图", "成本构成饼图", "敏感性曲线", "供需热力图"]`。

- [ ] **Step 2：测试 + 实现 + commit**

### Task 3.2：coder_node 拆为 N 次单图调用

**Files:**
- Create: `src/math_agent/prompts/coder_figure_one.py`
- Modify: `src/math_agent/nodes/coder.py`
- Modify: `tests/nodes/test_coder_multi_figure.py`（新建）

> ⚠ 决策 #2 拍板后展开完整实现。以下伪代码补全了变量定义，但仍需按实际
> `CodeArtifact` schema 和 `build_prompt_figure_one` 签名 finalize。

- [ ] **Step 1：写测试**：mock LLM + runner，断言 coder 对每个 figure_purpose 各调一次。

- [ ] **Step 2：改 coder_node**

伪代码：

```python
def coder_node(state):
    model = state.latest_model()
    workdir = Path(state.output_dir or tempfile.mkdtemp(prefix="math_agent_"))
    workdir.mkdir(parents=True, exist_ok=True)
    purposes = model.figure_purposes or [model.description]  # 兼容 Plan B 没有 figure_purposes 的旧 state
    artifacts: list[CodeArtifact] = []
    for i, purpose in enumerate(purposes):
        prev_err: str | None = None
        for attempt in range(MAX_CODE_RETRIES + 1):
            draft: CoderDraft = complete(
                build_prompt_figure_one(model, purpose, prev_err),
                schema=CoderDraft, system=SYSTEM,
                model=MODEL_ROUTING["coder"],
            )
            result = run_python(draft.code, workdir=workdir / f"fig_{i}_attempt_{attempt}")
            artifacts.append(CodeArtifact(
                purpose=purpose, code=draft.code,
                stdout=result.stdout, stderr=result.stderr,
                success=result.success, artifact_paths=result.artifact_paths,
            ))
            if result.success:
                break
            prev_err = result.stderr
    return {"code_artifacts": artifacts}
```

- [ ] **Step 3-N**：测试 + commit

> **重要**：Plan B 的 `figure_pipeline_node` 已经做 critic + analyst 后处理，**不动它**——只把 coder 上游变成多次产图。figure_pipeline 自然遍历所有 PNG 给出 critic。

---

## Phase 4：modeler 推导链（决策 #3）

> ⚠ 决策 #3 拍板才能写完整 task。下面假设决策 = A。

### Task 4.1：modeler_derivation prompt

**Files:**
- Create: `src/math_agent/prompts/modeler_derivation.py`

6 个 DerivationStep，每个一个 prompt。**每次调用把已完成的前序 steps 喂回**，让 LLM 看到
"上一步推导到这里，现在接着推"——与人类写推导的过程一致，保证 step 间逻辑连贯。

```python
DERIVATION_STEPS = [
    "motivation",        # 1. 动机：为什么用这个模型族
    "math_statement",    # 2. 数学陈述：模型族形式化
    "param_estimation",  # 3. 参数估计：MLE / 矩估计
    "constraints",       # 4. 约束推导：定常性 / 可解性 → 参数约束
    "transformation",    # 5. 等价变换：Markov 形式 / 状态空间
    "solution",          # 6. 求解：解析解 / 数值方法 / 滤波
]

def build_derivation_prompt(model: ModelVersion, step_kind: str,
                            completed_steps: list[DerivationStep]) -> str:
    prev = "\n".join(
        f"[{i+1}] {s.title}: {s.statement} → {s.result}"
        for i, s in enumerate(completed_steps)
    ) or "（这是第一步，无前序）"
    return (
        f"# 模型\n{model.description}\n方程：{'; '.join(model.equations)}\n\n"
        f"# 已完成推导步骤\n{prev}\n\n"
        f"# 当前步骤：{step_kind}\n"
        f"请输出 JSON：{{\"title\": str, \"motivation\": str, \"statement\": str, \"result\": str}}。"
    )
```

### Task 4.2：modeler_node 加 derivation 子循环

**Files:**
- Modify: `src/math_agent/nodes/modeler.py`

伪代码：

```python
def modeler_node(state):
    # Plan B 原逻辑：单次 LLM 出 ModelVersion 骨架（含 figure_purposes）
    base = complete(build_prompt_v0(state), schema=ModelVersion, ...)
    # Plan D：只对 final 阶段加 6 步推导链，每步把前序喂回
    if base.stage == "final":
        completed: list[DerivationStep] = []
        for step_kind in DERIVATION_STEPS:  # 6 步
            step = complete(
                build_derivation_prompt(base, step_kind, completed),
                schema=DerivationStep, ...
            )
            completed.append(step)
        # Self-consistency gate：让 LLM 回看整个推导链，检查逻辑连贯性
        # 如果 step 1 推导方向走歪，2-6 会连锁错误。这一次额外调用防止在歪骨架上建楼。
        consistency = complete(
            build_consistency_prompt(base, completed),
            schema=ConsistencyCheck,  # {"coherent": bool, "issues": list[str]}
            ...
        )
        if not consistency.coherent:
            logger.warning(f"Derivation consistency check failed: {consistency.issues}")
            # 保留 steps 但标记——writer 渲染时会看到 issues 标注
            base.derivation_notes = "; ".join(consistency.issues)
        base.derivation_steps = completed
    return {"model_versions": [base]}
```

> **成本**：6 次 derivation step 调用 + **1 次 self-consistency 调用** = 7 次额外 LLM 调用，
> 仅 final 阶段。basic/improved 阶段不触发，维持单次产出。
>
> **Self-consistency prompt**（`build_consistency_prompt`）：
> 输入完整 6 步 `completed` 和 `base.description`，输出 JSON：
> `{"coherent": bool, "issues": ["step 3 的结论与 step 1 假设矛盾", ...]}`。
> `derivation_notes` 写入 `ModelVersion`（需在 state.py 中加 `derivation_notes: str = ""` 字段）。

### Task 4.3：writer 模板渲染推导链

**Files:**
- Modify: `src/math_agent/templates/writer_section_model_solution.md.j2`

在模板里渲染 `model.derivation_steps`，让 writer 看到推导链作为输入素材。

> **Token 预算控制**：6 个 step × full text 可能 800-1500 字，加上 IRON RULES + critic
> feedback + retrieved_context 已经接近 8k 上下文上限。**每个 step 进 prompt 时只取
> `motivation + statement` 的前 100 字 + `result` 的前 100 字**（合计 ≤ 300 字/step，6 step
> ≤ 1800 字）。模板里：
>
> ```jinja
> {% for step in model.derivation_steps %}
> ### {{ step.title }}
> - 动机：{{ step.motivation[:100] }}
> - 陈述：{{ step.statement[:100] }}
> - 结果：{{ step.result[:100] }}
> {% endfor %}
> ```
>
> writer 仍能看到完整推导链 *结构*，但每段省略号截断，避免推爆上下文。最终论文里要不
> 要展开完整 step.statement，由 writer 节点自己决定（它会从 state.model_versions 里再读）。

---

## Phase 5：references 实时检索（决策 #4 = D）

> ⚠ 决策 #4 = B（等 RAG）时跳过本 phase。
> 决策 #4 = C（静态库）时，Task 5.1 改为建 `builtin_library.json`（20 篇 seed），Task 5.2 改为从 json 筛选——作为离线 fallback。

### Task 5.1：Reference 模型 + Semantic Scholar API 封装

**Files:**
- Modify: `src/math_agent/state.py` — 加 `Reference` Pydantic 模型
- Create: `src/math_agent/tools/scholar.py` — Semantic Scholar API 封装
- Create: `tests/tools/test_scholar.py`

- [ ] **Step 1：在 `state.py` 加 `Reference` 模型**

```python
class Reference(BaseModel):
    """真实文献条目。来源：Semantic Scholar API 或静态库。"""
    id: str                        # Semantic Scholar paperId 或静态库自定义 id
    title: str
    authors: list[str] = Field(default_factory=list)
    venue: str = ""
    year: int = 0
    doi: str = ""
    domains: list[str] = Field(default_factory=list)  # problem_domains 交集标记
```

- [ ] **Step 2：写失败的测试**

```python
from math_agent.tools.scholar import search_references, ScholarRateLimitError
from math_agent.state import Reference


def test_search_references_returns_real_references(mocker):
    fake_resp = mocker.MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "total": 1,
        "data": [{
            "paperId": "abc123",
            "title": "Bike sharing demand forecasting",
            "authors": [{"name": "Smith, J."}, {"name": "Lee, K."}],
            "year": 2018,
            "venue": "Transportation Research Part B",
            "externalIds": {"DOI": "10.1016/j.trb.2018.01.001"},
        }],
    }
    mocker.patch("math_agent.tools.scholar.requests.get", return_value=fake_resp)
    refs = search_references("bike sharing demand ARIMA", limit=5)
    assert len(refs) == 1
    assert isinstance(refs[0], Reference)
    assert refs[0].title.startswith("Bike sharing")
    assert refs[0].doi.startswith("10.")
    assert refs[0].year == 2018


def test_search_references_returns_empty_on_network_error(mocker):
    mocker.patch("math_agent.tools.scholar.requests.get", side_effect=ConnectionError("net"))
    refs = search_references("anything", limit=5)
    assert refs == []  # 网络失败优雅降级，返回空（writer 用静态库 fallback 或写"未找到"）


def test_search_references_returns_empty_on_rate_limit(mocker):
    fake_resp = mocker.MagicMock()
    fake_resp.status_code = 429
    mocker.patch("math_agent.tools.scholar.requests.get", return_value=fake_resp)
    refs = search_references("anything", limit=5)
    assert refs == []  # rate limit 优雅降级
```

- [ ] **Step 3：实现 `tools/scholar.py`**

```python
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
    try:
        resp = requests.get(
            _API_URL,
            params={"query": query, "limit": limit, "fields": _FIELDS},
            timeout=timeout,
        )
    except requests.RequestException:
        return []

    if resp.status_code != 200:
        return []  # 429 rate limit / 5xx 都降级

    data = resp.json().get("data", [])
    refs: list[Reference] = []
    for item in data:
        refs.append(Reference(
            id=item.get("paperId", ""),
            title=item.get("title", ""),
            authors=[a.get("name", "") for a in item.get("authors", [])],
            venue=item.get("venue", ""),
            year=item.get("year", 0) or 0,
            doi=(item.get("externalIds") or {}).get("DOI", ""),
        ))
    return refs
```

- [ ] **Step 4：在 `pyproject.toml` 加 `"requests>=2.31"`（若尚未有）**

- [ ] **Step 5：跑测试**

Run: `pytest tests/tools/test_scholar.py -v`
Expected: 3 passed.

- [ ] **Step 6：commit**

```bash
git add src/math_agent/state.py src/math_agent/tools/scholar.py tests/tools/test_scholar.py pyproject.toml
git commit -m "feat(refs): Semantic Scholar API wrapper + Reference model"
```

### Task 5.2：writer 节点检索 + 注入 + 静态库 fallback

**Files:**
- Modify: `src/math_agent/prompts/analyst.py` — schema 加 `problem_domains: list[str]`
- Modify: `src/math_agent/nodes/analyst.py` — `AnalystOutput` 加 `problem_domains`
- Modify: `src/math_agent/state.py` — `MathModelingState` 加 `problem_domains: list[str]`（**覆盖语义字段**，类似 `stage_target` / `iteration`，不需要 `Annotated[..., add]`；analyst 写一次，writer 读一次）
- Modify: `src/math_agent/nodes/writer.py` — references section 调 `search_references`
- Modify: `src/math_agent/templates/writer_section_references.md.j2`
- Create: `src/math_agent/references/builtin_library.json` — 离线 fallback（10 篇 generic seed）

> **problem_domains 来源**：让 analyst 顺手做这事，**零额外 LLM 调用**。
> analyst prompt 加一条："请同时输出 `problem_domains: list[str]`，从下列固定集合中选取
> 1-3 个：`['optimization', 'time_series', 'machine_learning', 'graph', 'probability', 'queueing', 'simulation', 'generic']`。"
> 节点把 `problem_domains` 写回 state，writer 节点从 state 读。
>
> **`problem_domains` 默认空 list，Plan A/B 的 analyst 测试无需改动**（Pydantic 可选字段）。

伪代码：

```python
def _select_references(problem: str, problem_domains: list[str]) -> list[Reference]:
    """优先 Semantic Scholar API；网络失败降级到静态库。"""
    from math_agent.tools.scholar import search_references
    # 用 problem 文本做查询（比 problem_domains 关键词更精准）
    refs = search_references(problem[:200], limit=10)
    if refs:
        return refs
    # fallback：从静态库按 domain 交集筛选
    lib = json.loads((REF_DIR / "builtin_library.json").read_text())
    candidates = [Reference(**r) for r in lib
                  if set(r.get("domains", [])) & set(problem_domains)]
    if not candidates:
        candidates = [Reference(**r) for r in lib if "generic" in r.get("domains", [])]
    return sorted(candidates, key=lambda r: -r.year)[:10]
```

把选中的引文传入 writer references prompt，让 LLM **只能引用这个清单**（逐条给出
title/authors/year/venue/DOI，writer 负责组织成规范的参考文献格式）。

---

## Phase 6：端到端验证（手动跑 + 自动门禁）

⚠️ 此 phase 的 **跑** 是手动 LLM 实测（耗时 ~50 min / 题），但**验收**是 bench 自动门禁。

> **Plan D 对 Plan C 的反向修改声明**：本 Phase Task 6.2 会扩展 Plan C 的
> `src/math_agent/bench/expectations.json`（加新字段）和 `bench/runner.py` 的 `_evaluate`
>（读新字段）。这是 Plan D 显式声明的跨 plan 修改，执行者需同步改 Plan C 文件。

### Task 6.1：清理 + 跑 v10

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
# ⚠ 警告：taskkill -F -IM python.exe 会杀掉机器上所有 python 进程（含 IDE/Jupyter）。
# 执行前确认无其他重要 python 任务，或改用更精准的方式（如按 PID 杀）。
taskkill -F -IM python.exe 2>/dev/null || true
rm -rf runs/eval_v10 runs/eval_v10.log
python -m math_agent.cli run \
  --problem tests/fixtures/sample_problem.json \
  --out runs/eval_v10 \
  --thread v10 \
  --no-interrupt \
  --template gmcm \
  --school "上海交通大学" \
  --team-id "No.20260629" \
  --members "李华,张三,王五" 2>&1 | tee runs/eval_v10.log
```

预计 45-80 分钟。

### Task 6.2：验收点 — 自动化到 bench expectations

所有验收点写进 `src/math_agent/bench/expectations.json`（Plan C Phase 3 已建），bench runner
跑完后自动 assert——**不靠人眼**：

```json
{
  "v10_sample": {
    "min_paper_pdf_pages": 30,
    "min_code_artifacts_success": 3,
    "min_code_artifacts_total": 5,
    "min_figures": 5,
    "min_derivation_steps": 4,
    "min_references": 5,
    "references_source": "semantic_scholar",
    "xelatex_must_have_zero_errors": true,
    "max_paper_critic_issues": 3
  }
}
```

Plan C `bench/runner.py` 的 `_evaluate` 函数需要扩展 expect schema 以读这些字段。**这是
Plan D 对 Plan C 的反向依赖**——Plan D 落地时同步在 Plan C 的 expectations.json /
`_evaluate` 上加字段。

**`_evaluate` 签名扩展**：当前签名 `_evaluate(case_id: str, final_state: dict, expect: dict)`。
v10 验收需要 `output_dir` 路径（读 PDF 页数 + xelatex 编译日志），因此签名扩展为：
`_evaluate(case_id: str, final_state: dict, expect: dict, *, output_dir: Path | None = None)`。
所有已有调用点（`run_bench` 中）加 `output_dir=out_dir / case_id`。
新增字段的检查逻辑如下：

| 新 expect 字段 | `_evaluate` 实现 |
|---|---|
| `min_paper_pdf_pages` | 在 `output_dir` 下找 `paper.pdf`，用 PyPDF2/pdfplumber 读页数 |
| `min_code_artifacts_success/_total` | `len(final_state.code_artifacts)` / 过滤 `success=True` |
| `min_figures` | `len(final_state.figures)` |
| `min_derivation_steps` | `final_state.latest_model().derivation_steps` 长度 |
| `min_references` + `references_source` | 解析 `paper.references` 字符串，按 `\n` 或 `[N]` 分割计数条数；来源校验仅对 `"builtin_library"` 做（遍历每条 ref 在 builtin_library.json 中查 title 匹配），对 `"semantic_scholar"` 仅计数不作来源校验 |
| `xelatex_must_have_zero_errors` | 在 `output_dir` 下读 `compile.log`（latex 节点产出），grep `! ` 和 `Error:` |
| `max_paper_critic_issues` | `len(final_state.latest_critic("paper").issues)` |

人工 sanity check 仍保留（看一眼论文是否能读），但**门禁**完全由 bench 自动跑。

- [ ] paper.pdf 页数 ≥ 30 → expectations.min_paper_pdf_pages
- [ ] code_artifacts 数量 ≥ 5（success=True ≥ 3）→ expectations.min_code_artifacts_*
- [ ] state.figures 数量 ≥ 5 → expectations.min_figures
- [ ] ModelVersion(stage="final").derivation_steps 数量 ≥ 4 → expectations.min_derivation_steps
- [ ] paper.references ≥ 5 条，且来源与 `references_source` 匹配：
  - `"semantic_scholar"`：每条至少含 title + year（DOI 可选），条数 ≥ 5
  - `"builtin_library"`：每条在 builtin_library.json 中可匹配 → expectations.references_source
- [ ] xelatex 编译 0 missing-char、0 errors → expectations.xelatex_must_have_zero_errors
- [ ] paper_critic.issues ≤ 3 → expectations.max_paper_critic_issues

---

## 已知非本计划范围

- **多问题题目**（A 题 4 问）：当前 sample_problem.json 是单 problem。多问题支持需要再拆 writer 章节、graph 加 per-question 循环——独立工单
- **真实仿真器**（如优秀论文里的 MATLAB 离散事件仿真器）：需要 simulator 节点，独立 plan
- **自研求解器算法**（Bianchi 模型稳态解 / 自定义 Kalman 滤波）：当前 coder 跑不动，依赖 LLM 在 4000 token 内写出复杂算法——不在本 plan
- **跨题目 RAG**：Plan C 的 RAG 集成与本 plan 正交，可独立完成

---

## 与 Plan C 的关系

| 维度 | Plan C | Plan D |
|---|---|---|
| 焦点 | 工程化基础设施 | 论文内容深度 |
| 改动文件 | 新建 errors/retry/rag/bench/tracing | 改 writer/modeler/coder 节点 + 加 prompts |
| 单题成本 | 同 Plan B | ~6x Plan B（质量优先：writer 10 + coder N + modeler +6） |
| 依赖关系 | 独立 | **Plan D 依赖 Plan C 完成**（顺序强约束，见下） |
| 串行建议 | **C 先（必须）→ D 后** | 同左 |

**为什么 C 必须先于 D？**

1. **Plan C Task 2.6** 在 analyst / modeler / writer 三个节点的 `build_prompt` 上加了
   `retrieved_context: str = ""` 参数，并在节点内调用 `rag.retrieve.search`。
2. **Plan D Phase 2** 把 writer 拆为 per-group 多次调用，每个 group 有自己的模板与
   `build_prompt`。如果 Plan D 先做，Plan C 后做就要在每个 group 模板上再做一次相同的参数
   注入——**改动半径翻倍**。
3. 反过来：先 C 后 D，`build_prompt` 已经支持 `retrieved_context`，Plan D Phase 2 的每个
   group prompt 函数继续接受这个参数即可（详见 Phase 2 Task 2.1 模板要求）。

**Plan D Phase 2 对 retrieved_context 的具体处理**（绑定 Plan C 已落地）：
- 每个 group 的 `build_prompt(state, retrieved_context: str = "")` 保留 Plan C 的签名
- writer_node 在 group 循环之前**只查一次** RAG，把 snippet 复用到每个 group，避免 N 次查询
- group 模板在末尾 `{{ retrieved_context }}` 位置渲染检索结果

---

## 自我审查

**0. Brainstorm 缺口**

本 plan 是 draft，**5 个核心决策点**未拍板：
1. writer 章节粒度（4 段 / 9 段 / 大纲+填充）
2. coder 多任务模式（显式 plan / 自决 / pre+main）
3. modeler 推导链结构（schema 扩字段 / prompt 自律 / 独立节点）
4. references 真实化（编 / RAG / 静态库 / Semantic Scholar API）— **默认 D，API 实时检索**
5. 失败回滚策略（节级 checkpoint / 复用 Plan B SqliteSaver / 全有全无）— **默认 B，零额外工作**

需在 brainstorm 阶段逐个拍板，然后展开 Phase 2/3/4/5 对应 task 的 inline 代码。Phase 1（state 扩展）的字段是决策 #2=A、#3=A 的产物——**只在对应决策落地时随 Phase 一起提交**，不单独打底，避免堆死代码。

**1. Spec 覆盖**

| 你列的 4 个内容深度问题 | 解决在哪个 Phase |
|---|---|
| 页数 ≥ 30 卡在 16 页 | Phase 2 writer 拆章节 ✓ |
| 图量 ≥ 10 卡在 1-2 张 | Phase 3 coder 多任务 ✓ |
| 多章节深度推导 vs 方程列表 | Phase 4 modeler 推导链 ✓ |
| 真实参考文献 5-15 篇 | Phase 5 Semantic Scholar API（+ 静态库 fallback）✓ |

**2. Placeholder 扫描**

本 plan 是 DRAFT，含 5 个标记 `⚠ 决策 #N` 的拍板点和"等决策 #N 拍板后展开"的占位。**这不是隐式 TODO，是显式 brainstorm 出口**——执行前必须 brainstorm 拍板，然后把对应 Phase 的 task 展开为 final 版本。

执行 final 版本时，无任何 TBD / TODO / "类似 Task N" / "实现 X 即可" 占位。

**3. 类型一致性**

- `ModelVersion.figure_purposes: list[str]`、`derivation_notes: str`、`DerivationStep` 字段（title/motivation/statement/result）— Phase 1 定义，Phase 3/4 使用 ✓
- `CriticIssue(section, problem)` — Phase 1.3 定义，替换 `CriticReport.issues: list[str]`；Phase 2 Task 2.3 `_sections_to_rewrite` 按 `section` 过滤 ✓
- `_WRITER_SECTIONS`（list[str]）— Phase 2 定义，但具体内容依赖决策 #1
- `CodeArtifact` schema 不变（与 Plan B 完全兼容）✓
- `Reference(id/title/authors/venue/year/doi/domains)` — Phase 5 Task 5.1 定义，`search_references` 和 `_select_references` 使用一致 ✓
- `search_references(query, limit)` 返回 `list[Reference]`，在 `tools/scholar.py` 与 `test_scholar.py` 中签名一致 ✓
- `problem_domains: list[str]` — 覆盖语义字段（analyst 写一次 / writer 读一次），不需要 `Annotated[..., add]` ✓
- `DERIVATION_STEPS` 6 项（motivation/math_statement/param_estimation/constraints/transformation/solution），Task 4.1 定义、Task 4.2 循环、Task 4.3 渲染一致 ✓

无类型/命名漂移。

**4. 改动半径估算**

- Phase 1：~40 行 state（含 CriticIssue）+ 测试，安全
- Phase 2：writer 节点 + outline 模板 + 7 个 section 模板，~200 行（决策 #1=C 时）
- Phase 3：coder 节点 + figure_one prompt，~80 行
- Phase 4：modeler 节点 + derivation prompt（6 步 + 前序喂回 + self-consistency），~140 行
- Phase 5：scholar.py API 封装 ~40 行 + Reference 模型 + 10 篇 fallback json + writer 注入 ~30 行

总改动 ~470 行 src + ~300 行测试。预计 1-2 周（含 brainstorm + 执行 + v10 验证）。

**5. 质量最大化决策汇总**

本 plan 的 5 个决策默认推荐已对齐"质量优先"：
1. writer = C（大纲 + 逐段填充，10 次调用）— 买章节间逻辑连贯
2. coder = A（N 次单图调用）— 买 retry 隔离 + 每图定制 prompt
3. modeler = A（6 次推导 + 前序喂回 + 1 次 self-consistency）— 买推导深度 + step 间连贯 + 逻辑自检
4. references = D（Semantic Scholar API）— 真实性 + 相关性 + 覆盖度全优，静态库仅作 fallback
5. 回滚 = B（复用 Plan B SqliteSaver）— 零额外工作，节点级 checkpoint 已够用

总成本 ~6x Plan B，单次跑 $10-25。对冲国一的竞赛论文可接受。

---

## Execution Handoff

**本 plan 是 DRAFT，不可直接执行**。Execution 路径：

1. **Brainstorm**：用 `superpowers:brainstorming` 逐个拍板 5 个决策点，更新本文档为 final 版本
2. **执行 final 版本**：用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`

哪个先动？或者你希望我现在直接 brainstorm 这 5 个决策点（推荐先做 #1 因为它影响最大）。
