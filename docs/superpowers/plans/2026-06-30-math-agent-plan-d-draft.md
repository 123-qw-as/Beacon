# 数学建模多智能体系统 — 内容深度突破层实现计划（Plan D）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **状态：DRAFT — 关键架构决策点已标出，需在执行前 brainstorm 拍板。**

**Goal:** 把当前单次 LLM 调用产出的 ~16 页 / 1-2 张图 / 扁平方程列表的论文产出，提升到 ≥ 30 页 / ≥ 10 张图 / 多步推导 / 真实文献的国一级别。

**Architecture:** 把"一次性梭哈"的 `writer_node` / `coder_node` / `modeler_node` 拆为**多次 LLM 调用**——writer 按章节生成、coder 按图任务多次调用、modeler 输出推导链而非方程列表。Plan D 不动 Plan B 的 graph 拓扑（analyst → modeler ⇄ critic → coder → sensitivity → figure_pipeline → writer ⇄ paper_critic → evaluation → human_review → latex），只**重构这 3 个节点的内部实现**，让每个节点在其内部多次调用 LLM 完成深度产出。

**Tech Stack:** 不引入新依赖。Plan A/B/C 的栈不变：LangGraph / Pydantic / LiteLLM / jinja2 / pytest-mock。

> **前置条件**：Plan A/B 已完成且 `pytest -q` 全绿。Plan C（工程化层：retry / RAG / bench / tracing）建议**先于** Plan D 完成——理由：(a) Plan D 单次跑成本翻 3-5 倍（多个 LLM 调用），retry 不到位时跑失败成本大；(b) RAG 是 Plan D modeler 推导链 / writer 文献引用质量的依赖。Plan D 在 Plan C 未完成时**仍可执行**，但 bench 不可用、单次跑失败需手工恢复。

---

## ⚠️ 待 brainstorm 决策点（执行前必拍板）

### 决策 #1：writer 拆章节的粒度

| 选项 | LLM 调用数 | 单次 token | 总成本 vs 当前 | 风险 |
|---|---|---|---|---|
| **A. 9 段独立调用**（abstract / problem_restatement / assumptions / notation / model_section / solution / sensitivity / conclusion / references）| 9 | 1-2k | 4-5x | 段间风格漂移、abstract 写时其它段还没写 |
| **B. 4 大块**（abstract + problem / model + solution / sensitivity + conclusion / references）| 4 | 2-4k | 2-3x | 中等 |
| **C. 阶段化**（pass 1 写大纲 → pass 2 按大纲逐段填充）| 1 + 9 = 10 | 1-2k | 5-6x | 最深但最贵 |

**默认推荐**：B（折中）。但需要看你想跑国一品质还是国二品质。

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

**默认推荐**：A。但要权衡每 ModelVersion 增加多少 LLM 调用（6 步 ×3 阶段 = 18 次）。

### 决策 #4：references 真实化路径

| 选项 | 实现 | 依赖 |
|---|---|---|
| **A. 让 writer 凭训练数据"想"** | prompt 加 "5-15 篇真实文献，年份 ≥ 2010" | LLM 会编 DOI 和卷期号 → **不可接受** |
| **B. RAG 检索（Plan C 范围）** | ingest 历年优秀论文 + Google Scholar 数据集 → writer 引 | 依赖 Plan C 完成 |
| **C. 提供静态 references 库**（手工准备 200 篇常见建模文献）| 一个 json 文件，writer 按问题领域筛选 | 工作量 1 天但可控 |

**默认推荐**：C（不依赖 Plan C，立即可用）+ 长期切 B（RAG）。

### 决策 #5：失败回滚策略

Plan D 后单次跑 50-80 分钟 / API 成本 $5-15。一个节点崩了重跑代价大。

- **A. 每章节独立可恢复**（writer 段落级 checkpoint）
- **B. 全有全无**（崩了重跑）
- **C. 等 Plan C 完成再做**（依赖 retry/tracing）

**默认推荐**：B 短期 + C 长期。

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
- `src/math_agent/state.py` — 加 `DerivationStep` 模型 + `ModelVersion.derivation_steps`、`figure_purposes`
- `src/math_agent/prompts/writer.py` — 拆为 dispatcher + 单段 prompt
- `src/math_agent/templates/writer_prompt.md.j2` — 单段模板（每段一份）

---

## Phase 1：State schema 扩展（最小依赖底座）

### Task 1.1：扩展 ModelVersion 加 figure_purposes（先写测试）

**Files:**
- Modify: `tests/test_state.py`
- Modify: `src/math_agent/state.py:26-32`

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

---

## Phase 2：writer 拆章节生成（核心改动）

> ⚠ 决策 #1 拍板才能写完整 task 内容。下面假设决策 = B（4 大块）。
>
> 决策 = A（9 段）：把 `_WRITER_GROUPS` 改为 9 个单段；其余结构相同。
> 决策 = C（pass1 大纲 + pass2 填充）：另写一个 `_outline_pass` 函数，pass2 复用 B 的 group 结构。

### Task 2.1：拆 writer prompt 模板

**Files:**
- Create: `src/math_agent/templates/writer_section_abstract_problem.md.j2`（摘要 + 问题重述）
- Create: `src/math_agent/templates/writer_section_model_solution.md.j2`（模型建立与求解）
- Create: `src/math_agent/templates/writer_section_sensitivity_conclusion.md.j2`（敏感性 + 结论）
- Create: `src/math_agent/templates/writer_section_references.md.j2`（参考文献）

每份模板的核心结构：保留现有 `writer_prompt.md.j2` 的 IRON RULES 1-6 头部，但 JSON schema 收窄到只输出本组段落字段。

> 此 task 的 4 份模板正文细节，等决策 #1 拍板后展开。需展示每份模板的 prompt 内容 + 模板使用的 Jinja2 变量列表。

- [ ] **Step 1-N**：分模板逐个测试 + 实现 + commit。

### Task 2.2：writer_node 改为 per-group LLM 调用

**Files:**
- Modify: `src/math_agent/nodes/writer.py`
- Modify: `tests/nodes/test_writer_per_section.py`（新建）

- [ ] **Step 1：写测试** —— 等决策 #1 拍板后展开具体测试代码。骨架：

```python
def test_writer_makes_one_call_per_group(mocker):
    from math_agent.nodes.writer import writer_node, _WRITER_GROUPS
    call_count = {"n": 0}
    def _fake(prompt, **kw):
        call_count["n"] += 1
        return PaperSections(...)  # 各 group 的 mock 输出
    mocker.patch("math_agent.nodes.writer.complete", side_effect=_fake)
    writer_node(_rich_state())
    assert call_count["n"] == len(_WRITER_GROUPS)
```

- [ ] **Step 2-N**：逐 group 实现 + 合并到完整 PaperSections。

### Task 2.3：合并多 group 输出 → PaperSections

- [ ] **Step 1**：把每个 group 的 LLM 输出 dict 合并入同一个 PaperSections 实例。
- [ ] **Step 2**：保留 `writer_iteration` 自增逻辑（Plan B 已落地的 writer↔paper_critic 闭环）。
- [ ] **Step 3**：测试 + commit。

> **风险点**：writer↔paper_critic 闭环现在跑两轮 = 2 × 4 = 8 次 LLM 调用。预计 writer 节点 token 成本 8x。

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

- [ ] **Step 1：写测试**：mock LLM + sandbox，断言 coder 对每个 figure_purpose 各调一次。

- [ ] **Step 2：改 coder_node**

伪代码：

```python
def coder_node(state):
    model = state.latest_model()
    artifacts = []
    purposes = model.figure_purposes or [model.description]  # 兼容 Plan B 没有 figure_purposes 的旧 state
    for purpose in purposes:
        for attempt in range(MAX_CODE_RETRIES + 1):
            draft = complete(build_prompt_figure_one(model, purpose, prev_err), ...)
            result = run_python(draft.code, workdir=workdir / f"fig_{i}_attempt_{attempt}")
            artifacts.append(CodeArtifact(...))
            if result.success: break
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

每个 DerivationStep 一个 prompt，输入：之前的 description + equations + 已完成的 steps。

### Task 4.2：modeler_node 加 derivation 子循环

**Files:**
- Modify: `src/math_agent/nodes/modeler.py`

伪代码：

```python
def modeler_node(state):
    # Plan B 原逻辑：单次 LLM 出 ModelVersion
    base = complete(build_prompt_v0(state), schema=ModelVersion, ...)
    # Plan D：对 final 阶段加推导链
    if base.stage == "final":
        for step_kind in ["motivation", "param_estimation", "constraints", "solution"]:
            step = complete(build_derivation_prompt(base, step_kind), schema=DerivationStep, ...)
            base.derivation_steps.append(step)
    return {"model_versions": [base]}
```

### Task 4.3：writer 模板渲染推导链

**Files:**
- Modify: `src/math_agent/templates/writer_section_model_solution.md.j2`

在模板里渲染 `model.derivation_steps`，让 writer 看到推导链作为输入素材。

---

## Phase 5：references 静态库（决策 #4 = C）

> ⚠ 决策 #4 = B（等 RAG）时跳过本 phase。

### Task 5.1：建 builtin_library.json

**Files:**
- Create: `src/math_agent/references/builtin_library.json`

按问题领域分类 200 篇经典文献：
- 优化（线性规划 / 整数规划 / 凸优化）
- 时间序列（ARIMA / SARIMAX / Holt-Winters）
- 机器学习（XGBoost / RF / GBM）
- 图论（最短路 / TSP / MTZ）
- 概率（泊松 / 负二项 / CVaR）
- ...

JSON schema：

```json
{
  "id": "ar2010-arima-bike",
  "domain": ["time_series", "transportation"],
  "title": "Bike sharing demand: ARIMA approach",
  "authors": ["Smith, J.", "Lee, K."],
  "venue": "Transportation Research Part B",
  "year": 2018
}
```

### Task 5.2：writer 节点筛选 + 注入

**Files:**
- Modify: `src/math_agent/nodes/writer.py`
- Modify: `src/math_agent/templates/writer_section_references.md.j2`

伪代码：

```python
def _select_references(problem_domain: list[str], k: int = 10) -> list[Reference]:
    lib = json.loads((REF_DIR / "builtin_library.json").read_text())
    # 按 domain 取交集，再按 year 降序
    candidates = [r for r in lib if set(r["domain"]) & set(problem_domain)]
    return sorted(candidates, key=lambda r: -r["year"])[:k]
```

把选中的引文传入 writer references prompt，让 LLM **只能引用这个清单**。

---

## Phase 6：端到端验证（手动，非门禁）

⚠️ 此 phase 不是 pytest，是手动 LLM 实测，耗时 ~50 min / 题。

### Task 6.1：清理 + 跑 v10

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
taskkill -F -IM python.exe
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

### Task 6.2：验收点

- [ ] paper.pdf 页数 ≥ 30
- [ ] code_artifacts 数量 ≥ 5（其中 success=True ≥ 3）
- [ ] state.figures 数量 ≥ 5
- [ ] ModelVersion(stage="final").derivation_steps 数量 ≥ 4
- [ ] paper.references 至少 5 条且**每条**能在 builtin_library.json 中追溯到
- [ ] xelatex 编译 0 missing-char、0 errors
- [ ] paper_critic 报告中 issues ≤ 3

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
| 单题成本 | 同 Plan B | 3-5x Plan B |
| 依赖关系 | 独立 | **Plan D Phase 5 决策 #4=B 依赖 Plan C Phase 2** |
| 串行建议 | C 先（基础设施）| D 后（内容突破） |

Plan C 与 Plan D **不可同时执行**——会改 writer 节点同一处代码（Plan C Task 2.6 加 retrieved_context 参数 vs Plan D Task 2.2 拆 per-group）。**先 C 后 D**：C 落地后 writer.build_prompt 已支持 retrieved_context，D 把它沿用到每个 group 即可。

---

## 自我审查

**0. Brainstorm 缺口**

本 plan 是 draft，**5 个核心决策点**未拍板：
1. writer 章节粒度（4 段 / 9 段 / 大纲+填充）
2. coder 多任务模式（显式 plan / 自决 / pre+main）
3. modeler 推导链结构（schema 扩字段 / prompt 自律 / 独立节点）
4. references 真实化（编 / RAG / 静态库）
5. 失败回滚策略

需在 brainstorm 阶段逐个拍板，然后展开 Phase 2/3/4/5 对应 task 的 inline 代码。Phase 1（state 扩展）是底座，所有选项都需要——可立即执行。

**1. Spec 覆盖**

| 你列的 4 个内容深度问题 | 解决在哪个 Phase |
|---|---|
| 页数 ≥ 30 卡在 16 页 | Phase 2 writer 拆章节 ✓ |
| 图量 ≥ 10 卡在 1-2 张 | Phase 3 coder 多任务 ✓ |
| 多章节深度推导 vs 方程列表 | Phase 4 modeler 推导链 ✓ |
| 真实参考文献 5-15 篇 | Phase 5 references 静态库 ✓ |

**2. Placeholder 扫描**

本 plan 是 DRAFT，含 5 个标记 `⚠ 决策 #N` 的拍板点和"等决策 #N 拍板后展开"的占位。**这不是隐式 TODO，是显式 brainstorm 出口**——执行前必须 brainstorm 拍板，然后把对应 Phase 的 task 展开为 final 版本。

执行 final 版本时，无任何 TBD / TODO / "类似 Task N" / "实现 X 即可" 占位。

**3. 类型一致性**

- `ModelVersion.figure_purposes: list[str]`、`DerivationStep` 字段（title/motivation/statement/result）— Phase 1 定义，Phase 3/4 使用 ✓
- `_WRITER_GROUPS`（list[GroupSpec]）— Phase 2 定义，但具体内容依赖决策 #1
- `CodeArtifact` schema 不变（与 Plan B 完全兼容）✓
- `_select_references()` 返回 `list[Reference]`，Reference 类型在 Task 5.1 定义

**4. 改动半径估算**

- Phase 1：~30 行 state + 测试，安全
- Phase 2：writer 节点 + 4 个 prompt 模板，~150 行（决策 #1=B 时）
- Phase 3：coder 节点 + figure_one prompt，~80 行
- Phase 4：modeler 节点 + derivation prompt，~100 行
- Phase 5：builtin_library.json 200 条手工 + 选择器 ~50 行

总改动 ~400 行 src + ~300 行测试。预计 1-2 周（含 brainstorm + 执行 + v10 验证）。

---

## Execution Handoff

**本 plan 是 DRAFT，不可直接执行**。Execution 路径：

1. **Brainstorm**：用 `superpowers:brainstorming` 逐个拍板 5 个决策点，更新本文档为 final 版本
2. **执行 final 版本**：用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`

哪个先动？或者你希望我现在直接 brainstorm 这 5 个决策点（推荐先做 #1 因为它影响最大）。
