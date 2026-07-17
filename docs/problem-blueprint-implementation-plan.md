# Problem Blueprint 主线改造实施方案

> 历史实施记录。ProblemBlueprint 与 blueprint critic 已进入现行代码；文件中的行号、待办和“当前”描述保留当时上下文，现行状态见 [`README.md`](README.md)。

## 1. 结论

本轮改造的核心目标不是增加更多 Agent，而是把 Beacon 的主链条打稳：

```text
题目理解 -> 问题蓝图 -> 蓝图审查 -> 建模 -> 模型审查 -> 代码生成 -> 模型代码一致性审查 -> 写作与评估
```

当前 Beacon 已有完整 LangGraph 流程，继续堆 DataAuditor、ModelSelector、ResultVerifier 等节点会放大复杂度。更优先的事情是让“题目怎么拆、变量是什么、目标是什么、约束是什么、数据需要什么、代码是否实现模型”变成可追踪的结构化资产。

因此，本方案只新增两个关键审查点：

```text
Blueprint Critic
Model-Code Consistency
```

并让现有 `Modeler`、`Model Critic`、`Coder`、`Writer` 消费同一个 `ProblemBlueprint`。

## 2. 当前仓库约束

本地项目当前主流程为：

```text
analyst -> modeler -> model_critic -> coder -> sensitivity -> figure_pipeline -> writer -> writer_section loop -> paper_critic -> table_assembler -> evaluation -> human_review -> latex
```

需要特别注意以下实现细节：

1. `MathModelingState` 中很多 list 字段使用 `Annotated[..., add]`，节点返回增量时会追加，不会覆盖。
2. 当前 `analyst_node` 只输出 `assumptions` 和 `problem_domains`。
3. 当前 `coder_node` 已经按 `figure_purposes` 生成多段独立绘图代码，并额外生成 baseline 代码。
4. 当前 `writer` 是 prep + section loop 结构，不应一次性重写成单节点大 prompt。
5. 重试计数必须明确在节点返回值里递增，否则 routing 中的“最多重试一次”不会生效。

## 3. 目标架构

改造后的主流程：

```text
analyst
  -> blueprint_critic
  -> modeler
  -> model_critic
  -> coder
  -> model_code_consistency
  -> sensitivity
  -> figure_pipeline
  -> writer
  -> writer_section loop
  -> paper_critic
  -> table_assembler
  -> evaluation
  -> human_review
  -> latex
```

其中：

- `analyst` 负责生成 `ProblemBlueprint`。
- `blueprint_critic` 负责进入建模前审查题目理解。
- `modeler` 必须沿 `ProblemBlueprint` 建模。
- `model_critic` 增加 blueprint 对齐审查。
- `coder` 必须输出稳定、可追踪的指标。
- `model_code_consistency` 审查代码是否真正实现 final model。
- `writer` 必须覆盖 blueprint 中的全部小问、目标、约束、指标和验证计划。

## 4. P0：Problem Blueprint 主链落地

P0 只做地基，不做 UI、不做 benchmark、不做数据审计。

### 4.1 修改 `state.py`

新增以下结构：

```python
class SubQuestionBlueprint(BaseModel):
    id: str
    original_text: str
    task_type: Literal[
        "evaluation",
        "prediction",
        "optimization",
        "simulation",
        "classification",
        "explanation",
        "strategy",
        "generic",
    ]
    depends_on: list[str] = Field(default_factory=list)
    expected_output: str = ""
    success_criteria: list[str] = Field(default_factory=list)


class DecisionVariable(BaseModel):
    name: str
    meaning: str
    domain: str = ""
    unit: str = ""


class ObjectiveSpec(BaseModel):
    direction: Literal["minimize", "maximize", "evaluate", "predict", "explain"]
    description: str
    math_hint: str = ""


class ConstraintSpec(BaseModel):
    description: str
    math_hint: str = ""
    source: Literal["given", "derived", "assumed"] = "assumed"


class MetricSpec(BaseModel):
    name: str
    meaning: str
    direction: Literal["higher_better", "lower_better", "target", "neutral"] = "neutral"
    unit: str = ""


class DataRequirement(BaseModel):
    field: str
    meaning: str
    needed_for: str
    given_or_missing: Literal["given", "missing", "derived", "unknown"] = "unknown"
    handling_strategy: str = ""


class ModelingCandidate(BaseModel):
    name: str
    route: str
    suitable_for: list[str] = Field(default_factory=list)
    data_required: list[str] = Field(default_factory=list)
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    risk: str = ""
    recommendation_score: int = 0


class RecommendedRoute(BaseModel):
    route: str
    reason: str
    baseline: str = ""
    fallback: str = ""


class ValidationPlanItem(BaseModel):
    target: str
    method: str
    pass_criteria: str = ""


class ProblemBlueprint(BaseModel):
    core_task: str
    subquestions: list[SubQuestionBlueprint] = Field(default_factory=list)
    decision_variables: list[DecisionVariable] = Field(default_factory=list)
    objectives: list[ObjectiveSpec] = Field(default_factory=list)
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    metrics: list[MetricSpec] = Field(default_factory=list)
    data_requirements: list[DataRequirement] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    problem_domains: list[str] = Field(default_factory=list)
    modeling_candidates: list[ModelingCandidate] = Field(default_factory=list)
    recommended_route: RecommendedRoute | None = None
    validation_plan: list[ValidationPlanItem] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
```

在 `MathModelingState` 中新增：

```python
problem_blueprint: ProblemBlueprint | None = None
blueprint_iteration: int = 0
```

注意：

- `problem_blueprint` 是覆盖语义，不要用 `Annotated[..., add]`。
- `assumptions` 和 `problem_domains` 继续保留，兼容旧节点。
- Analyst 重试时不能无脑追加 `assumptions`，否则会重复。P0 阶段可以接受旧字段仅同步一次；更稳妥做法是在 writer/modeler 中优先读取 `problem_blueprint.assumptions`。

### 4.2 修改 Analyst 输出

`src/math_agent/nodes/analyst.py` 中将 `AnalystOutput` 替换为 `ProblemBlueprint`。

节点返回：

```python
return {
    "problem_blueprint": blueprint,
    "assumptions": blueprint.assumptions,
    "problem_domains": blueprint.problem_domains,
}
```

如果存在上一轮未通过的 analyst critic，则把 issues 和 suggestions 注入 prompt。

实现注意：

- `problem_blueprint` 覆盖。
- `assumptions` 是追加字段，重试时要避免重复。可在 P0 中用 `blueprint_iteration == 0` 时才同步旧字段。
- `problem_domains` 是覆盖字段，可以直接同步。每次 analyst 重试会覆盖 `problem_domains`，这是预期行为：题目理解调整后，参考文献方向（writer 中 `select_references()` 消费此字段）也应随之更新。

### 4.3 新增 Blueprint Critic

新增文件：

```text
src/math_agent/prompts/blueprint_critic.py
src/math_agent/nodes/blueprint_critic.py
```

输出复用 `CriticReport`：

```python
CriticReport(
    target="analyst",
    score=0,
    approved=False,
    issues=[...],
    suggestions=[...],
)
```

审查重点：

- 是否遗漏小问。
- 任务类型是否判断错误。
- 变量、目标、约束是否空泛。
- 数据需求是否虚构。
- 推荐路线是否能回答全部小问。
- 是否缺少 baseline 和验证计划。

`blueprint_critic_node` 返回时必须递增：

```python
return {
    "critic_reports": [report],
    "blueprint_iteration": state.blueprint_iteration + 1,
}
```

> TODO: `latest_critic("analyst")` 通过 `target` 字段匹配报告。如果将来新增另一个也以 `"analyst"` 为 target 的 critic（如 DataAuditor），可能会匹配到错误的报告。当前只有一个 blueprint_critic 以 `"analyst"` 为 target，这是安全的。若后续扩展，可在 `CriticReport` 上增加 `critic_type: str` 字段用于区分同 target 的不同 reviewer。

### 4.4 新增 Blueprint routing

在 `src/math_agent/config.py` 新增常量：

```python
MAX_BLUEPRINT_ITERATIONS = 2  # 首次审查 + 最多一次 retry
```

在 `src/math_agent/routing.py` 新增：

```python
def after_blueprint_critic(state: MathModelingState) -> str:
    report = state.latest_critic("analyst")
    if report is None:
        return "retry"
    if report.approved:
        return "advance"
    if state.blueprint_iteration >= MAX_BLUEPRINT_ITERATIONS:
        return "advance_with_warning"
    return "retry"
```

语义约定：

- `blueprint_iteration == 0`：尚未审查。
- `blueprint_iteration == 1`：analyst 首轮输出已被审查一次。若未通过，允许一次 retry。
- `blueprint_iteration >= 2`：第二次审查仍未通过，则带警告前进（最多一次 retry）。

注意：`blueprint_critic_node` 返回时已递增 `blueprint_iteration`（参见 4.3），routing 中的阈值 `>= MAX_BLUEPRINT_ITERATIONS` 与此递增顺序一致。

### 4.5 修改 `graph.py`

将：

```python
g.add_edge("analyst", "modeler")
```

改为：

```python
g.add_edge("analyst", "blueprint_critic")
g.add_conditional_edges(
    "blueprint_critic",
    after_blueprint_critic,
    {
        "retry": "analyst",
        "advance": "modeler",
        "advance_with_warning": "modeler",
    },
)
```

### 4.6 让 Modeler 消费 Blueprint

在 `ModelVersion` 中新增覆盖映射字段：

```python
class ModelQuestionCoverage(BaseModel):
    question_id: str
    how_answered: str
    related_equations: list[str] = Field(default_factory=list)
    related_metrics: list[str] = Field(default_factory=list)
```

并扩展 `ModelVersion`：

```python
question_coverage: list[ModelQuestionCoverage] = Field(default_factory=list)
objective_mapping: list[str] = Field(default_factory=list)
constraint_mapping: list[str] = Field(default_factory=list)
validation_mapping: list[str] = Field(default_factory=list)
```

修改 `src/math_agent/prompts/modeler.py`：

- 注入 `problem_blueprint`。
- 要求模型覆盖所有 `subquestions`。
- 要求 final 阶段必须给出 baseline 和 validation mapping。
- 如果新增变量、目标或约束，必须说明为什么不在 blueprint 中。
- **约束 `question_coverage.how_answered`**：必须引用具体的 equation 名称或 variable 名称（如 `"由公式 E_dispatch 求解"`），不允许纯自然语言描述（如 `"the model answers this question"`），以便 model_critic 交叉验证。

修改 `src/math_agent/nodes/modeler.py`：

- `build_prompt(...)` 增加 `blueprint=state.problem_blueprint`。
- 如果 `state.problem_blueprint is None`，返回错误，不进入自由建模。

```python
if state.problem_blueprint is None:
    return {"errors": ["modeler: missing problem_blueprint"]}
```

## 5. P1：模型与代码一致性

P1 用来解决“模型写得像一回事，代码跑的是另一回事”的问题。

### 5.1 增强 Model Critic

修改 `src/math_agent/prompts/model_critic.py`，新增 blueprint 对齐检查：

- 是否覆盖所有小问。
- 变量是否对应 blueprint。
- 目标函数是否对应 blueprint。
- 约束是否对应 blueprint。
- final model 是否包含 baseline。
- validation plan 是否可执行。
- **交叉验证 `question_coverage.how_answered`**：如果模型声称覆盖了小问 X，但 equations 中找不到对应的公式 → 记为 issue。

这一步不新增节点，只增强已有 `model_critic`。

### 5.2 调整 Coder 指标输出

当前单图 prompt 已经要求：

```text
RESULT: baseline=ours ...
```

建议统一成稳定指标协议：

```text
RESULT: baseline=ours total_cost=... service_rate=...
RESULT: baseline=greedy total_cost=... service_rate=...
```

短期不建议改成 `METRIC`，因为现有 baseline 和表格汇总已经围绕 `RESULT:` 工作。可以在后续统一抽象解析层。

Coder prompt 需要增加：

- 代码必须实现 final model 的核心变量、目标函数、约束。
- 至少输出 blueprint.metrics 中的关键指标。
- baseline 代码必须沿用同一指标名。
- stdout 不允许只输出自然语言总结。

### 5.3 新增 Model-Code Consistency 类型

> 设计说明：不复用 `CriticReport`，因为一致性审查需要细粒度的变量/目标/约束对齐信息（`implemented_variables`、`missing_constraints` 等），这些结构化字段无法无损嵌入 `CriticReport` 的通用 `issues`/`suggestions` 列表。相比之下，blueprint_critic 的审查内容可以充分表达为 `issues` 和 `suggestions`，复用 `CriticReport` 即可。

在 `state.py` 新增：

```python
class ModelCodeConsistencyReport(BaseModel):
    score: int
    approved: bool
    implemented_variables: list[str] = Field(default_factory=list)
    missing_variables: list[str] = Field(default_factory=list)
    implemented_objectives: list[str] = Field(default_factory=list)
    missing_objectives: list[str] = Field(default_factory=list)
    implemented_constraints: list[str] = Field(default_factory=list)
    missing_constraints: list[str] = Field(default_factory=list)
    output_metric_alignment: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
```

在 `MathModelingState` 中新增：

```python
model_code_reports: Annotated[list[ModelCodeConsistencyReport], add] = Field(default_factory=list)
code_verify_iteration: int = 0
```

### 5.4 新增一致性审查节点

新增：

```text
src/math_agent/prompts/model_code_consistency.py
src/math_agent/nodes/model_code_consistency.py
```

**前置修正：CodeArtifact 增加 batch 字段**

`code_artifacts` 是 `Annotated[list[CodeArtifact], add]`（追加语义），coder 重试时会追加新 artifact，旧 artifact 不会被清除。为避免一致性审查在 retry 场景下看到脏数据，需在 `CodeArtifact` 上增加 `batch` 字段：

```python
class CodeArtifact(BaseModel):
    # ... 现有字段 ...
    batch: int = 0  # 新增：标识属于第几轮代码生成
```

Coder 节点每次运行时递增 batch：

```python
max_batch = max((a.batch for a in state.code_artifacts), default=0)
current_batch = max_batch + 1
# 所有新生成的 artifact 写入 batch=current_batch
```

重要修正：当前 `coder_node` 可能生成多段主方案代码，不应只取最后一个 `category == "figure"` 的 artifact。应汇总所有**最新批次**中的成功主方案 artifact：

```python
max_batch = max((a.batch for a in state.code_artifacts), default=0)
main_artifacts = [
    a for a in state.code_artifacts
    if a.success and a.category == "figure" and a.batch == max_batch
]
baseline_artifacts = [
    a for a in state.code_artifacts
    if a.category.startswith("baseline:") and a.batch == max_batch
]
```

审查输入应包含：

- `problem_blueprint`
- final model
- 所有成功主方案代码摘要
- 所有主方案 stdout
- baseline stdout
- 失败 artifact 的 stderr 摘要

如果没有成功主方案代码，则直接返回未通过报告。

### 5.5 新增一致性 routing

新增：

```python
def after_model_code_consistency(state: MathModelingState) -> str:
    if not state.model_code_reports:
        return "retry_coder"

    report = state.model_code_reports[-1]
    if report.approved and report.score >= 7:
        return "advance"
    if state.code_verify_iteration >= 2:
        return "advance_with_warning"
    return "retry_coder"
```

`model_code_consistency_node` 返回时递增：

```python
return {
    "model_code_reports": [report],
    "code_verify_iteration": state.code_verify_iteration + 1,
}
```

### 5.6 修改 graph

将：

```python
g.add_edge("coder", "sensitivity")
```

改为：

```python
g.add_edge("coder", "model_code_consistency")
g.add_conditional_edges(
    "model_code_consistency",
    after_model_code_consistency,
    {
        "retry_coder": "coder",
        "advance": "sensitivity",
        "advance_with_warning": "sensitivity",
    },
)
```

## 6. P2：Writer 和质量 Benchmark

P2 在主链稳定后做。

### 6.1 Writer 消费 Blueprint

当前 writer 是分节生成。不要重写结构，只修改 prompt 构造：

```text
src/math_agent/prompts/writer_section.py
```

在 `build_outline_prompt` 和 `build_section_prompt` 中注入：

- `problem_blueprint`
- `question_coverage`
- `objective_mapping`
- `constraint_mapping`
- `validation_mapping`
- `model_code_reports`

写作约束：

- 摘要必须覆盖 `core_task`。
- 问题重述必须按 `subquestions` 展开。
- 符号说明优先使用 `decision_variables`。
- 模型建立必须对应 `objectives` 和 `constraints`。
- 求解与结果必须对应 `metrics`、`validation_plan` 和 baseline。
- 如果某个小问没有被模型或代码支持，必须写成局限性，不能编造结果。

### 6.2 质量 Benchmark

新增目录：

```text
src/math_agent/bench/problems/
src/math_agent/bench/rubrics/
```

新增脚本：

```text
src/math_agent/bench/quality_eval.py
```

评分维度：

```text
BlueprintScore
ModelScore
CodeScore
PaperScore
```

最低门槛：

```text
BlueprintScore >= 7
ModelScore >= 7
CodeScore >= 7
PaperScore >= 7
overall >= 7.2
```

其中 `overall = (BlueprintScore + ModelScore + CodeScore + PaperScore) / 4`（算术平均）。要求 `overall >= 7.2` 而非 `>= 7.0` 的原因：确保至少有一项高于 7 分（最小差值 = 0.2），避免”每项刚好踩线”的及格即放行情况——即要求至少一个维度达到良好水平。

P2 的 benchmark 先做”最低质量门槛”，不要追求自动判断一等奖级别。

### 6.3 CLI report 展示

`math-agent report --out runs/demo` 增加：

```text
Blueprint Score: 8/10
Model-Code Score: 7/10
Question Coverage: 3/3
Unresolved Issues: 1
```

## 7. P3：UI 和后续能力

P3 最后做：

- Web UI 展示 Problem Blueprint。
- Web UI 展示 Blueprint Critic 和 Model-Code Consistency。
- DataPackage / DataAuditor。
- 更强 RAG 对齐。
- 云端沙箱。

现阶段不要先做可编辑表单。先展示，后交互。

## 8. 测试计划

### 8.1 P0 测试

新增或修改：

```text
tests/test_problem_blueprint_schema.py
tests/nodes/test_analyst.py
tests/nodes/test_blueprint_critic.py
tests/test_routing.py
tests/test_graph_smoke.py
tests/test_graph_full_smoke.py
tests/nodes/test_modeler.py
```

重点断言：

- `ProblemBlueprint` 能解析完整 JSON。
- 缺失可选字段时有默认值。
- `MathModelingState.problem_blueprint` 可覆盖写入。
- analyst 能返回 blueprint。
- blueprint critic 能生成 `target="analyst"` 的报告。
- routing 不会无限 retry。
- graph 中包含 `blueprint_critic`。
- modeler 没有 blueprint 时显式报错。

### 8.2 P1 测试

新增：

```text
tests/nodes/test_model_code_consistency.py
```

重点断言：

- 缺少 final model 时未通过。
- 缺少成功代码 artifact 时未通过。
- 代码缺少关键变量时 `missing_variables` 非空。
- 成功代码和指标对齐时 `approved=True`。
- graph 中 `coder` 后接 `model_code_consistency`，再进入 `sensitivity`。

### 8.3 P2 测试

新增：

```text
tests/bench/test_quality_eval.py
tests/test_report.py
```

重点断言：

- rubric 能加载。
- 评分结果包含四类分数。
- report 能展示 blueprint 和一致性摘要。

## 9. 推荐实施顺序

建议按以下顺序开工：

```text
1. state.py 增加 ProblemBlueprint 与字段
2. analyst prompt 和 analyst_node 改为输出 ProblemBlueprint
3. 新增 blueprint_critic prompt 和 node
4. routing.py 增加 after_blueprint_critic + config.py 增加 MAX_BLUEPRINT_ITERATIONS
5. graph.py 接入 blueprint_critic
6. modeler prompt 和 node 消费 ProblemBlueprint
7. model_critic 增加 Blueprint 对齐检查
8. coder prompt 强化 RESULT 指标输出
9. state.py 增加 CodeArtifact.batch + ModelCodeConsistencyReport
10. coder_node 增加 batch 递增逻辑
11. 新增 model_code_consistency prompt 和 node
12. routing.py 和 graph.py 接入 model_code_consistency
13. writer_section 注入 ProblemBlueprint（须在 benchmark 之前完成，benchmark 的 PaperScore 依赖 writer 按 blueprint 覆盖全部小问）
14. 增加 quality benchmark（依赖第 13 项 writer 完成，建议先手动检查样例论文质量确认输出稳定后再启动）
15. 增加 CLI report 展示
16. 最后做 Web UI 展示
```

## 10. 验收标准

完成 P0 后，应满足：

- 每次运行都有 `problem_blueprint`。
- blueprint 能覆盖所有小问。
- blueprint critic 会阻止明显偏题的题目理解直接进入建模。
- modeler 的 final model 能说明每个小问如何被回答。

完成 P1 后，应满足：

- final model 的核心变量、目标和约束能在代码中找到对应实现。
- stdout 有稳定 `RESULT:` 行。
- baseline 与主方案使用同一指标名。
- 不一致时最多重试一次，然后带警告继续。

完成 P2 后，应满足：

- 论文按 blueprint 覆盖全部小问。
- 论文数字来自代码输出或敏感性分析。
- benchmark 能给出 blueprint、model、code、paper 四类质量分数。

## 11. 最终原则

本方案的核心原则：

```text
先把题目理解变成结构化资产，再让后续每个节点围绕这个资产工作。
```

Beacon 下一阶段真正需要的不是更多节点，而是更强的一致性：题目理解一致、模型一致、代码一致、论文一致。

## 12. 第一性原理审查问题清单

以下问题基于对当前代码库的逐文件验证（`state.py`、`graph.py`、`nodes/*.py`、`routing.py`、`config.py`）发现，按严重程度排序。核心架构方向正确，但在实施前需修正阻塞级问题。

### 12.1 问题概览

| 优先级 | 编号 | 问题 | 修正量 | 涉及文件 |
|--------|------|------|--------|----------|
| **阻塞** | #1 | `code_artifacts` 追加语义导致一致性审查脏数据 | 改 CodeArtifact schema + coder 节点 + 一致性节点 | `state.py`, `nodes/coder.py`, `nodes/model_code_consistency.py` |
| **阻塞** | #2 | 路由语义自相矛盾 | 删除方案 A，选定方案 B，新增常量 | `routing.py`, `config.py` |
| 中等 | #3 | `latest_critic("analyst")` 的未来冲突风险 | 加 TODO 或新增 `critic_type` 字段 | `state.py` |
| 轻微 | #4 | ModelVersion 新增字段的 LLM 产出风险 | 加约束规则 | `prompts/modeler.py`, `prompts/model_critic.py` |
| 轻微 | #5 | `problem_domains` 覆盖语义未明确说明 | 文档补充 | 本文档 |
| 轻微 | #6 | ModelCodeConsistencyReport 不复用 CriticReport 的理由缺失 | 文档补充 | 本文档 |
| 轻微 | #7 | `overall >= 7.2` 阈值缺少推导 | 文档补充 | 本文档 |
| 轻微 | #8 | P2 实施项之间的依赖顺序不明确 | 文档补充 | 本文档 |

### 12.2 阻塞问题

#### #1：code_artifacts 追加语义导致一致性审查脏数据

**根因**：

方案 5.4-5.5 设计了 `model_code_consistency -> (retry_coder) -> coder -> model_code_consistency` 的 retry 循环。但 `code_artifacts` 字段定义为 `Annotated[list[CodeArtifact], add]`（追加语义），每次 coder 重试会**追加**新的 artifact，旧 artifact **不会被清除**。

方案 5.4 的过滤逻辑：

```python
main_artifacts = [
    a for a in state.code_artifacts
    if a.success and a.category == "figure"
]
```

这会将第一轮失败前成功的主方案代码、第二轮生成的新代码、第三轮生成的新代码**全部混在一起**。一致性审查器无法区分哪些是"最新版本"。

**影响**：

如果第一次 coder 生成的代码关键变量正确，但第二次 coder 生成的代码关键变量缺失，一致性报告可能基于旧代码给出 `approved=True`，导致应该被拦截的问题被放行。第 2+ 轮 retry 时会产生**系统性误判**。

**修正方案**（二选一）：

**方案 A — 增加批次号**（推荐）：

1. 在 `CodeArtifact` 上增加 `batch: int = 0` 字段。
2. Coder 节点每次运行时读取 `max_batch = max((a.batch for a in state.code_artifacts), default=0) + 1`，所有新 artifact 写入此 `batch`。
3. Model-code consistency 节点只检查 `batch == max_batch` 的 artifact：

```python
max_batch = max((a.batch for a in state.code_artifacts), default=0)
main_artifacts = [
    a for a in state.code_artifacts
    if a.success and a.category == "figure" and a.batch == max_batch
]
```

**方案 B — 新增 code_generation_batch 状态字段**：

1. 在 `MathModelingState` 上增加 `code_generation_round: int = 0`（覆盖语义）。
2. Coder 节点每次运行时递增并写入 `code_generation_round`。
3. `CodeArtifact` 上增加 `generation_round: int`，一致性节点按 `generation_round == state.code_generation_round` 过滤。

推荐方案 A，因为不需要额外状态字段，且 `batch` 信息属于 artifact 自身的元数据。方案 4.1 正确识别了 `assumptions` 的追加语义风险，但在 5.4 遗漏了同一个问题在 `code_artifacts` 上的影响。

---

#### #2：路由语义自相矛盾

**根因**：

方案 4.4 给出了两套互斥的路由规则，但没有给出决策依据：

```python
# 方案 A：建议 max retry = 0（retry 变为死代码）
if state.blueprint_iteration >= 1:
    return "advance_with_warning"

# 方案 B：建议 max retry = 1（符合"最多重试一次"的设计意图）
if state.blueprint_iteration >= 2:
    return "advance_with_warning"
```

按照 `blueprint_critic_node` 先递增再进入 routing 的设计（4.3）：

- **方案 A**：analyst (iter=0) → critic (iter=1) → routing 发现 iter>=1，直接 `advance_with_warning`，**一次都不 retry**。`"retry"` 分支成为死代码。
- **方案 B**：analyst (iter=0) → critic (iter=1，未通过) → retry → analyst (iter=1) → critic (iter=2) → advance_with_warning。**最多一次 retry**。

**修正方案**：

1. 删除方案 A 的描述。
2. 在 `config.py` 中新增常量：

```python
MAX_BLUEPRINT_ITERATIONS = 2  # blueprint critic 允许的评估次数（首次 + 一次 retry）
```

3. `after_blueprint_critic` 最终实现：

```python
def after_blueprint_critic(state: MathModelingState) -> str:
    report = state.latest_critic("analyst")
    if report is None:
        return "retry"
    if report.approved:
        return "advance"
    if state.blueprint_iteration >= MAX_BLUEPRINT_ITERATIONS:
        return "advance_with_warning"
    return "retry"
```

### 12.3 中等问题

#### #3：latest_critic("analyst") 的未来冲突风险

**根因**：

方案 4.3 让 blueprint_critic 输出 `CriticReport(target="analyst")`，4.4 的 routing 用 `state.latest_critic("analyst")` 查找最新报告。

`latest_critic` 从 `critic_reports` 末尾反向扫描，返回第一个 `target` 匹配的报告。如果将来有人新增另一个也以 `"analyst"` 为 target 的 critic（如 DataAuditor），`latest_critic` 会拿到最后一个匹配的报告，不一定是 blueprint critic 的。

**修正方案**：

P0 阶段不做结构性修改，但应在文档中标注为已知限制。后续可在 `CriticReport` 上增加 `critic_type: str` 字段用于区分同 target 的不同 reviewer，或在分析报告中增加：

```python
# TODO: 如果将来有多个 target="analyst" 的 critic，需要按 critic_type 筛选
#       当前只有一个 blueprint_critic，latest_critic("analyst") 是安全的
```

---

### 12.4 轻微问题

#### #4：ModelVersion 新增字段的 LLM 产出风险

**根因**：

方案 4.6 在 `ModelVersion` 上新增 `question_coverage`、`objective_mapping`、`constraint_mapping`、`validation_mapping`。这些都是让 LLM 输出的结构化"元信息"字段（说明模型如何对应 blueprint），而非模型本身的内容。

现有 `ModelVersion` 的 schema 已经较大（`equations`、`variables`、`figure_purposes`、`derivation_steps`）。增加 4 个列表字段会让 prompt 更长，且 LLM 可能在这些字段里写空泛内容（如 `"this model covers all questions"`），与实际 equations/variables 不一致。

**修正方案**：

1. `question_coverage.how_answered` 字段约束为：必须引用具体的 `equation` 名称或 `variable` 名称（如 `"由公式 E_dispatch 求解"`），不允许纯自然语言描述（如 `"the model answers this question"`）。Prompt 中显式要求。
2. Model_critic 的 blueprint 对齐检查（方案 5.1）须交叉验证：如果模型声称覆盖了小问 X，但 equations 中找不到对应的公式 → 记为 issue。

---

#### #5：analyst 返回 problem_domains 的语义说明缺失

**根因**：

方案 4.2 让 analyst 返回 `"problem_domains": blueprint.problem_domains`。`problem_domains` 是覆盖语义字段（非 `Annotated[..., add]`），每次 analyst 重试会**覆盖**之前的值。

当前 `problem_domains` 在 `writer` 节点中被传给 `select_references()` 用于参考文献选择。覆盖语义意味着每次 analyst 重试都会更新参考文献方向。这是合理的（题目理解改了，参考文献也应跟着变），但方案未说明这是**预期行为**。

**修正方案**：

在方案 4.2 处增加说明：

> `problem_domains` 跟随最新 analyst 输出变化是预期行为：当 blueprint critic 触发 analyst 重试时，题目理解可能调整，参考文献方向也应随之更新。

---

#### #6：ModelCodeConsistencyReport 不复用 CriticReport 的理由缺失

**根因**：

方案 4.3 让 blueprint_critic 复用 `CriticReport`，但 5.3 为 model-code consistency 新增了独立类型 `ModelCodeConsistencyReport`。两者不一致的理由未说明。

**判断**：新增独立类型是**合理的**。`CriticReport` 只有通用的 `issues` 和 `suggestions` 列表，而 `ModelCodeConsistencyReport` 需要细粒度字段（`implemented_variables`、`missing_variables`、`implemented_objectives` 等），这些无法嵌入 `CriticReport` 而不改变其 schema。但应在方案中显式说明理由，避免后人误以为这是一个不一致的设计缺陷。

**修正方案**：

在方案 5.3 处增加设计说明：

> 注：不复用 `CriticReport`，因为一致性审查需要细粒度的变量/目标/约束对齐信息（`implemented_variables`、`missing_constraints` 等），这些结构化字段无法无损嵌入 `CriticReport` 的通用 `issues`/`suggestions` 列表。相比之下，blueprint_critic 的审查内容可以充分表达为 `issues` 和 `suggestions`，复用 `CriticReport` 即可。

---

#### #7：P2 benchmark 的 overall >= 7.2 阈值缺少推导

**根因**：

方案 6.2 要求 `overall >= 7.2`，同时各项子分数 `>= 7`。如果各项都是 7，算术平均 `overall` 是 7.0，会低于 7.2 的门槛。这意味着**即使四项都在及格线上，overall 仍不通过**。

**修正方案**：

在方案 6.2 处增加推导说明，明确是算术平均还是加权平均，以及为什么 7.2 而非 7.0：

> `overall = (BlueprintScore + ModelScore + CodeScore + PaperScore) / 4`。要求 `overall >= 7.2` 而非 `>= 7.0` 的原因：确保至少有一项高于 7 分（最小差值 = 0.2），避免"每项刚好踩线"的及格即放行情况。即要求至少一个维度达到良好水平。

---

#### #8：P2 实施项之间的依赖顺序不明确

**根因**：

方案第 9 节推荐实施顺序：

```text
12. writer_section 注入 ProblemBlueprint    ← P2 内容
13. 增加 quality benchmark                  ← P2 内容
14. 增加 CLI report 展示
15. 最后做 Web UI 展示
```

第 12 项（writer）依赖 P0/P1 的 blueprint 和一致性报告。第 13 项（benchmark）又依赖 writer 的质量才能评分（PaperScore）。两者都在 P2 但未明确依赖顺序。writer 必须先完成，benchmark 才能度量。

**修正方案**：

在第 9 节中补充依赖说明：

> 第 12 项（writer 注入 blueprint）必须在第 13 项（benchmark）之前完成：benchmark 的 `PaperScore` 维度依赖论文按 blueprint 覆盖全部小问。建议在 writer 完成后先用少量样例手动检查论文质量，确认输出稳定后再启动 benchmark 开发。
