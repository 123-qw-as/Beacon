# 论文质量提升设计文档

**日期**：2026-07-03  
**状态**：待审批  
**背景**：Plan D 内容深度突破已完成（≥30 页 / ≥10 图 / 多步推导 / 真实参考文献）。当前瓶颈从"内容量"转向"内容质量"——论文结构、对比、表格密度、语言清洁度、数字可追溯性、附录精炼度，6 项全部是评委评阅时的隐性扣分点。

**6 大提升方向**：
1. 重构论文逻辑——从建模迭代内视图转为评委友好的 8 段呈现结构
2. 增加基准对比——代码真跑至少 3 个对照方案
3. 强化表格——代码强制 5 张表而非靠 LLM 自发
4. 清理生成痕迹——禁止 PaperCritic/Claim/Evidence/占位等词进入正文
5. 让每个结论可追溯——stdout 结构化和数字--来源映射
6. 把附录从"全部代码堆叠"改为"关键代码 + 说明"

**分两期交付**（方案 2）：
- **Phase 1**（无新代码运行，低风险高 ROI）：呈现层重写、表格强制生成、禁用词清洗、附录精选。
- **Phase 2**（引入新代码运行，较高风险）：基准对照实验、数字结构化提取与可追溯。

---

## A. 整体架构变更

### 原则

**建模层零触碰**：`analyst → modeler ⇄ model_critic` 的 `basic/improved/final` 三阶段迭代、`_advance_stage` 节点、`routing.py` 中的 `MAX_MODEL_ITERATIONS` 上限全部不动。8 段评委结构是针对**论文模板**（paper.tex.j2 / gmcm.tex.j2 / paper.md.j2）和 **writer 分节提示词** 的重写，不是对建模过程的改动。

### graph 拓扑

```
现状（Plan D）:
  figure_pipeline → writer ⇄ paper_critic →[advance]→ evaluation → human_review → latex → END

Phase 1 后:
  figure_pipeline → writer ⇄ paper_critic →[advance]→ table_assembler → evaluation → human_review → latex → END
                                                         ↑ 新增节点
```

**理由**：
- `table_assembler` 在 writer/critic 循环结束后运行——此时 section 文本已稳定，表格和清洗不会在 critic 重写中被覆盖。
- evaluation 能看到含表格的最终版本，评分更准。
- evaluation 的 `build_prompt`（`prompts/evaluation.py:24-26`）当前对 `model_section`、`solution`、`sensitivity` 各截断 800 字符。`table_assembler` 将表格追加在每个 section 的**末尾**，因此 prose 部分仍在截断窗口内可见。截断预算不变。

### 具体改动

**`graph.py`**（改 3 行）：
- L55 后新增 `g.add_node("table_assembler", _wrap(table_assembler_node, "table_assembler"))`
- L78 `after_paper_critic` 的 `"advance"` 目标从 `"evaluation"` 改为 `"table_assembler"`
- L80 `g.add_edge("evaluation", "human_review")` 之前新增 `g.add_edge("table_assembler", "evaluation")`

**`routing.py`**（改 1 行）：
- L32 `after_paper_critic` 的 `"advance"` 返回值保持 `"advance"` 不变（graph.py 的 conditional_edges 映射到 "table_assembler"，此处无需改返回值，仅改 graph.py 的目标映射）。

纠正：`after_paper_critic` 返回的是 `"advance"` / `"retry"` 字符串，而 `graph.py:78` 中的映射 `{"retry": "writer", "advance": "evaluation"}` 需要把 `"advance"` 目标改为 `"table_assembler"`。`routing.py` 自身不动。

### 已知限制：表格无 critic 评审闭环

`table_assembler` 位于 `paper_critic` **之后**，因此 PaperCritic 看不到注入的表格——表格格式问题无法被 critic 自动捕获并触发重写。这是有意取舍：

- 放在 critic **之前**：表格会在 critic→writer 重写循环中被覆盖丢失（writer 产出的是纯文本，不含表格）。
- 放在 critic **之后**（当前选择）：表格稳定保留，但无 critic 评审。

代价是表格质量只能靠 evaluation 节点或人工 review 发现。考虑到表格内容来自结构化数据（Phase 1 的 3 张强制表），格式由代码生成而非 LLM，格式出错概率低；语义错误（如敏感性评级算错）靠单元测试覆盖。此取舍合理，记为已知限制。

### 改动文件一览

| 文件 | Phase | 改动性质 |
|---|---|---|
| `state.py` | 1 | +`BaselineResult`、`NumericResult`、`CuratedSection`、`table_warnings` 字段 |
| `graph.py` | 1 | 注册 table_assembler 节点 + 改 1 条边 |
| `nodes/table_assembler.py` | 1 | **新建**（~250 行） |
| `templates/gmcm.tex.j2` | 1 | 8 段标题 + 附录精选 + members 默认值 |
| `templates/paper.tex.j2` | 1 | 同上 |
| `templates/paper.md.j2` | 1 | 同上 |
| `templates/writer_iron_rules.md.j2` | 1 | 补禁用词表（从 legacy writer_prompt.md.j2 同步） |
| `templates/writer_section_model.md.j2` | 1 | 8 段写作指令 |
| `templates/writer_section_solution.md.j2` | 1 | 8 段写作指令中的求解/对比部分 |
| `templates/writer_section_conclusion.md.j2` | 1 | 追加优缺点评价表要求 |
| `nodes/coder.py` | 2 | 对照方案运行逻辑 |
| `tools/runner.py` | 2 | 通用 `RESULT:` 提取函数（`extract_numeric_results`） |
| `nodes/sensitivity.py` | 2 | 复用通用提取函数（小重构，`_RESULT_RE` → 共享模块） |

### 数据流

```
paper_critic 输出 "advance"
  ↓
table_assembler（新节点）
  ├── 读：state.model_versions[-1].variables
  ├── 读：state.sensitivity_runs
  ├── 读：state.code_artifacts[-1].stdout
  ├── 读：state.paper（model_section / solution / sensitivity / conclusion）
  │
  ├── 生成表 1：参数说明表（从 model_versions[final].variables + equations）
  ├── 生成表 2：模型变量表（从 model_versions[final].variables，含单位/类型列）
  ├── 生成表 3：各模型结果对比表（Phase 1: 从 code_artifacts stdout 语义提取；Phase 2: 从 BaselineResult + NumericResult）
  ├── 生成表 4：敏感性结果表（从 sensitivity_runs）
  ├── 生成表 5：优缺点评价表（Phase 1: 从 paper.conclusion 语义抽取；Phase 2: 结构化要求 writer 产出）
  │
  ├── 注入表格到对应 section 字段末尾（# 约定：追加到各 field 值末尾，在 LaTeX 小节标题之前）
  │
  ├── 正则扫描 7 个 section 字段的禁用词：
  │     PaperCritic / issue / 回应 / Claim / Evidence / Reasoning / 代码[数字] / 超时 / 占位 / 李华 / 张三 / 王五
  │     → 替换或删除 + 写入 state.table_warnings
  │
  └── 输出：{"paper": 修改后的 PaperSections, "table_warnings": [...]}
    ↓
evaluation → human_review → latex → END
```

---

## B. 8 段呈现结构重写

### 设计决策

**不新增 PaperSections 字段**。现有 9 个字段（abstract / problem_restatement / assumptions / notation / model_section / solution / sensitivity / conclusion / references）保持不变。8 段评委结构通过 **writer 要求在字段内用 markdown 标题组织** 实现，因为：

- PaperCritic 的 `CriticIssue.section` 枚举（`state.py:67-70`）绑定到这 9 个字段名，拆字段会破坏 critic→rewrite 的映射体系。
- `_SECTION_FIELD_TO_GROUP`（`writer_section.py:53-63`）和 `_sections_to_rewrite`（`writer_section.py:218-240`）强依赖这 9 个字段名。
- 用 markdown 标题（`## 小节名`）组织字段内内容，LaTeX 模板将其渲染为小节标题即可。

### 映射关系

| 评委看到的 8 段 | 对应 PaperSections 字段 | 字段内 markdown 结构 |
|---|---|---|
| 问题分析 | `problem_restatement` | 单段论述，带「决策变量/目标/约束」三要素 |
| 基础预测模型 | `model_section`（前半） | `## 基础预测模型` + 理论推导 |
| 优化调度模型 | `model_section`（中段） | `## 优化调度模型` + 优化建桩 |
| 动态/不确定性扩展模型 | `model_section`（后半） | `## 动态与不确定性扩展` + 鲁棒/随机规划 |
| 求解算法 | `solution` | `## 求解算法与流程` + 算法描述、复杂度分析、数值结果 |
| 实验与对比 | `solution`（后半） | `## 实验设计与对比分析` + 对照表/结果解读 |
| 敏感性与鲁棒性 | `sensitivity` | 保持现有，新增强制表格 |
| 模型评价与推广 | `conclusion` | 改进四段式：优点/缺点/推广/方法论总结 |

### 前置依赖：修正 IRON RULE 4 与 8 段结构的冲突

**冲突**（核查 `writer_iron_rules.md.j2:10` 确认）：当前 IRON RULE 4 第二点明确"**不要用 markdown 标题**（`#`、`##`、`###`）。但 8 段结构要求 writer 在 `model_section` 写 `## 基础预测模型`、在 `solution` 写 `## 求解算法与流程`。两者直接矛盾——不修正则 writer 会拒绝写标题，8 段结构垮掉。

**修正**：`_md_headings_to_latex`（`latex.py:198-212`）已经把 `## xxx` → `\subsection{xxx}`、`### xxx` → `\subsubsection{xxx}`。因此 markdown 标题技术上完全支持，只需放宽铁规措辞。Phase 1 改 `writer_iron_rules.md.j2:10` 为：

> "**章节标题**：`model_section` 和 `solution` 内的子节用 `## xxx` 标题；更深层子节用 `### xxx`。**禁止顶层 `#`**——那会生成与模板 `\section{}` 冲突的 `\section{}`。其他章节（abstract / assumptions / notation / sensitivity / conclusion）不用标题。"

这是 Phase 1 的硬前置依赖，不是可选项。

### writer 模板改动

#### `writer_section_model.md.j2`（核心改动）

当前模板要求 `model_section` 按 `basic→improved→final` 三小节组织（L71）。重写后改为：

```
| `model_section` | 3000–5000 | 必须按以下三段组织，每段 ≥800 字：
  ## 基础预测模型
  - 预测问题的形式化定义（输入/输出/目标）
  - 特征工程与变量选择依据
  - 预测模型（XGBoost / STGNN / 时序预测等）的选择与数学定义
  - 关键推导步骤（从 model_versions[final].derivation_steps 取相关项）
  
  ## 优化调度模型
  - 从预测结果到调度决策的映射关系
  - 目标函数形式化（min 成本 / max 服务率 / 多目标）
  - 约束条件逐一推导（从题目条件 + model_versions[final].variables）
  - 模型类型声明（MILP / MINLP / 随机规划）
  
  ## 动态与不确定性扩展
  - 从静态到动态的扩展逻辑（滚动窗口 / 反馈修正）
  - 不确定性来源分析（需求波动 / 预测误差 / 外部冲击）
  - 鲁棒性/随机性处理方法（CVaR / 机会约束 / 场景树）
  - 扩展模型的数学形式
```

这个三段结构的写作指令告诉 writer 把 `model_versions[final]` 的推导链（`derivation_steps`）按主题分配到三段中，而不是按迭代阶段（basic/improved/final）平铺。同时要求每段首句点明该段相对于更简单方案的**增量价值**（为对比章节提供铺垫）。

#### `writer_section_solution.md.j2`（核心改动）

当前要求 solution 包含 4 部分（求解总述 / 分步过程 / 结果验证 / 结果讨论）。重写后分为两大节：

```
| `solution` | 1500–2500 | 必须按以下两段组织：
  ## 求解算法与流程
  - 算法选择动机（对比候选算法：精确求解器/启发式/元启发式/强化学习）
  - 算法伪代码（文字描述，不输出代码块）与复杂度分析
  - 关键步骤的公式代入（把通用方程填入题目具体参数）
  - 收敛性/最优性保证（若有）
  
  ## 实验设计与对比分析
  - 实验环境（数据规模 / 时间窗口 / 硬件）
  - 至少 3 个对照方案的描述与设置理由
  - 各方案定量结果（从代码运行结果提取，必须与 stdout 一致）
  - 结果分析（用 CER：Claim→Evidence→Reasoning，这里 Evidence 来自代码 stdout）
```

### LaTeX 模板改动

#### `gmcm.tex.j2`

```latex
% 现状
\section{模型的建立与求解}
{{ paper.model_section }}
\subsection{求解过程与结果}
{{ paper.solution }}

% 改后 —— model_section 中用 ## 标题自然分节
\section{模型的建立与求解}
{{ paper.model_section }}
\section{模型的求解与实验}
{{ paper.solution }}
```

`paper.model_section` 内部写的 `## 基础预测模型` 等 markdown 标题在 `_md_to_latex`（`latex.py:` 现有的 markdown 标题转换）中会被转换为 `\subsection{基础预测模型}` 等 LaTeX 命令，所以 8 段结构和 LaTeX 小节的映射关系由 markdown 标题自然驱动。

`gmcm.tex.j2:37` 的 `\subsection{求解过程与结果}` 保持不变（gmcmthesis 要求结构一致），但 `paper.solution` 内部的 `## 求解算法与流程` / `## 实验设计与对比分析` 由 writer 语义内容填充，不再需要硬编码的 subsection 标题。

#### `paper.md.j2`

```
% 现状
## 4. 模型的建立与演化
{{ paper.model_section }}
## 5. 模型的求解
{{ paper.solution }}

% 改后
## 4. 模型的建立
{{ paper.model_section }}
## 5. 模型的求解与实验
{{ paper.solution }}
## 6. 敏感性分析
{{ paper.sensitivity }}
## 7. 模型评价与推广
{{ paper.conclusion }}
```

### members 默认值

`state.py:167` 的 `members` 字段默认值为 `None`，gmcm 模板中用 `member_a | default("队员A")` 兜底。Phase 1 改动：确认模板中已使用 "队员A/B/C" 而非 "张三/李四/王五"（`gmcm.tex.j2:12-14` 已如此，无需修改）。

---

## C. table_assembler 节点 + 5 张表格

### 关于"强制"的诚实分层

5 张表中，**Phase 1 真正代码强制的只有 3 张**（表 1/2/4），它们的数据源是结构化字段，不依赖 LLM 或语义提取。表 3（对比表）和表 5（优缺点表）在 Phase 1 是"尽力填充"——从 stdout / conclusion 文本语义提取，能填几条填几条，提取不到的行填 `—` 并记入 warnings。这两张表要等 **Phase 2 的 `BaselineResult` / `NumericResult` 结构化数据到位后才是真强制**。

| 表 | Phase 1 | Phase 2 | 数据源 |
|---|---|---|---|
| 表 1 参数说明表 | ✅ 强制 | — | `model_versions[-1].variables` |
| 表 2 模型变量表 | ✅ 强制 | — | `model_versions[-1].variables` |
| 表 3 各模型结果对比表 | ⚠️ 尽力 | ✅ 强制 | P1: stdout 语义提取 / P2: `baseline_results` + `numeric_results` |
| 表 4 敏感性结果表 | ✅ 强制 | — | `sensitivity_runs` |
| 表 5 优缺点评价表 | ⚠️ 尽力 | ✅ 强制 | P1: conclusion 语义抽取 / P2: writer 结构化产出 |

### 节点职责

`nodes/table_assembler.py`（新建）提供以下能力：

1. **从 state 提取结构化数据**（不靠 LLM，纯代码）
2. **生成 markdown 表格**（每个表格是独立的 markdown pipe-table 字符串）
3. **注入到 PaperSections 对应字段**
4. **返回修改后的 paper + table_warnings**

### 5 张表格详设

#### 表 1：参数说明表

- **来源**：`state.model_versions[-1].variables`（final 阶段模型的变量字典）
- **注入位置**：`paper.model_section` 末尾（`## 模型参数表` 标题 + 表格）
- **格式**：

```
| 参数 | 含义 | 单位 | 类型 |
|---|---|---|---|
| ... | ... | ... | 决策变量 / 已知参数 / 预测输出 |
```

- **Phase 1 实现**：从 `ModelVersion.variables` 的 `{name: description}` 字典逐行生成。如果 description 中包含"单位"信息（如"需求量(件)"），尝试拆分单位和含义列；否则单位列填 `—`。类型从变量名按规则推断（下划线含字母→决策变量，否则→已知参数）。

#### 表 2：模型变量表

- **来源**：`state.model_versions[-1].variables`
- **注入位置**：`paper.notation` 字段末尾（追加到现有符号说明后面）
- **格式**：与表 1 相同但省略类型列，强化为

```
| 符号 | 含义 | 单位 |
|---|---|---|
```

- **Phase 1 实现**：这是对现有 `notation` 字段的增强——writer 已要求输出符号表（`writer_iron_rules.md.j2` 和 `writer_section_assumptions_notation.md.j2`），但格式不一定完整。`table_assembler` 检查 `paper.notation` 中是否已含 pipe-table；若已存在，跳过（避免重复表格）；若不存在，从 `variables` 生成一张补表。

#### 表 3：各模型结果对比表

- **来源**：
  - Phase 1：`state.code_artifacts[-1].stdout`（从 stdout 文本末尾找 `RESULT: solution=...` 行），以及 writer 在 `solution` 字段中已写出的对比数据。
  - Phase 2：`state.baseline_results`（结构化 BaselineResult 列表）+ `state.numeric_results`（从所有 code_artifacts stdout 提取）。
- **注入位置**：`paper.solution` 末尾（`## 实验结果对比` 标题 + 表格）
- **格式**：

```
| 方案 | 目标值 | 约束满足率 | 计算时间 | 备注 |
|---|---|---|---|---|
| 无调度 | ... | ... | ... | 基准：不做任何调度调整 |
| 简单平均预测 | ... | ... | ... | 基准：用历史均值代替预测 |
| 贪婪启发式 | ... | ... | ... | 基准：局部最优贪心分配 |
| 本文方案 | ... | ... | ... | XGBoost+随机规划滚动优化 |
```

- **Phase 1 实现**：从 `code_artifacts` 的 stdout 逐行匹配多种 `RESULT:` 格式（`sensitivity.py:47` 的 `RESULT: parameter=...` 是一个格式，还需要支持 `RESULT: scenario=X objective=Y` 等）。能提取几条填几条；提取不到的方案行填 `—` 并在 `table_warnings` 中记录。
- **Phase 2 实现**：从 `BaselineResult` 列表直接填入。

#### 表 4：敏感性结果表

- **来源**：`state.sensitivity_runs`
- **注入位置**：`paper.sensitivity` 末尾（`## 敏感性结果汇总` 标题 + 表格）
- **格式**：

```
| 参数 | 取值范围 | 指标 | 指标变化范围 | 敏感性评级 |
|---|---|---|---|---|
| ... | [x, y] | MAE / 成本 / 服务率 | [a, b] | 高/中/低 |
```

- **Phase 1 实现**：从 `SensitivityRun` 列表直接生成。敏感性评级：`(max(results) - min(results)) / abs(mean(results))`，>30% 标记为"高"，10-30% 为"中"，<10% 为"低"。

#### 表 5：优缺点评价表

- **来源**：`paper.conclusion`（语义抽取） + `state.model_versions[final].notes`
- **注入位置**：`paper.conclusion` 末尾（`## 模型综合评价` 标题 + 表格）
- **格式**：

```
| 维度 | 评价 |
|---|---|
| 预测准确性 | ... |
| 调度效率 | ... |
| 鲁棒性 | ... |
| 可解释性 | ... |
| 计算复杂度 | ... |
| 推广性 | ... |
```

- **Phase 1 实现**：用简单规则从 `paper.conclusion` 文本中提取"优点/缺点/推广"段落的关键短语填入对应维度。如果文本中找不到足够信息，跳过该表并在 `table_warnings` 中记录。
- **Phase 2 实现**：`writer_section_conclusion.md.j2` 新增结构化要求，让 writer 直接输出 6 维评价，方便 parser 提取。

### 表格批量注入逻辑

`table_assembler` 的 `_inject_table(section_text, table_md)` 函数：

1. 检查 `section_text` 中是否已含同名表格（通过 `## 表格标题` 匹配）
2. 若不存在：`section_text + "\n\n" + title + "\n" + table_md`
3. 若已存在但格式不完整：替换旧表格为新的标准格式（Phase 2）
4. 返回修改后的文本

---

## D. 确定性禁用词清洗

### 问题分析

现状（探索确认）：
- 禁用词表仅在 `writer_prompt.md.j2:125-139`（legacy 单次写入 prompt）中定义。
- 新的 per-section 模板（`writer_section_*.md.j2`）只 `{% include 'writer_iron_rules.md.j2' %}`——铁规 6 条但不含禁用词表。
- PaperCritic 的提示词（`prompts/paper_critic.py:1-14`）反复使用 "问题 (issue)" 等词——这些词可能从 prompt 泄露到 LLM 输出中。
- **没有任何确定性后处理**去清洗这些词。

### 解决方案

在 `table_assembler` 节点中，生成/注入表格之后，对所有 7 个 section 字段做确定性正则清洗。

### 禁用词清单与处理规则

```python
FORBIDDEN_PATTERNS = [
    # 元对话痕迹（从 PaperCritic/Writer 的 prompt 可能泄露到输出）
    (r"(?i)papercritic", "[内部评审]"),      # 改为无害标记
    (r"(?i)claim", "结论"),
    (r"(?i)evidence", "依据"),
    (r"(?i)reasoning", "推理"),
    (r"代码\s*\d+", "代码"),                 # "代码1" "代码[1]" → "代码"
    (r"\bissue\b(?!s)", "问题"),            # 只替换单数 "issue"，保留复数 "issues"
    (r"(?i)\bissues?\s*[:：]", "问题:"),    # "issues: ..." → "问题: ..."
    (r"回应\s*[:：]", "处理:"),              # "回应：..." → "处理:..."
    (r"回应", "处理"),                       # 孤立"回应"

    # 元信息泄露
    (r"超时", "运行"),                       # "代码超时" → "代码运行"
    (r"占位", "--"),                        # 任何形式占位 → 删除

    # 人名为占位名
    (r"李华", "队员A"),
    (r"张三", "队员A"),
    (r"王五", "队员B"),
]
```

### 实现细节

`_clean_forbidden_words(text: str, section: str) -> tuple[str, list[str]]`：

1. 按顺序应用 `FORBIDDEN_PATTERNS`，每次替换后记录 `warnings.append(f"[{section}] 替换: {old} → {new}")`
2. 统计替换次数
3. 返回 (cleaned_text, warnings)

### 补充到 writer prompt

为减少清洗负担，Phase 1 同时：
- 在 `writer_iron_rules.md.j2` 末尾追加铁规 7："**禁泄露**：正文中不得出现 PaperCritic、Claim、Evidence、Reasoning、issue（作为英文词）、回应、代码[数字]、超时、占位、李华、张三、王五。"
- `writer_section_*.md.j2` 通过 `{% include 'writer_iron_rules.md.j2' %}` 自动继承这条铁规。

---

## E. 基准对比实验（Phase 2）

### 目标

每篇论文至少包含 3 个定量对比：
1. **无调度方案 vs 本文方案**（上限对比：调度是否产生价值）
2. **简单平均预测 vs XGBoost/分位数预测**（预测方法对比）
3. **贪婪启发式 vs 随机规划/滚动优化**（求解方法对比）

### 数据模型

```python
class BaselineResult(BaseModel):
    """一次对照实验的结构化结果。"""
    name: str                             # "无调度" / "简单平均预测" / "贪婪启发式"
    category: Literal["no_schedule", "simple_pred", "greedy"]
    metrics: dict[str, float]             # {"total_cost": 1245.3, "service_rate": 0.82, ...}
    stdout: str = ""                      # 原始输出（可追溯）
    success: bool = False
    stderr: str = ""
```

`MathModelingState` 新增：
```python
baseline_results: Annotated[list[BaselineResult], add] = Field(default_factory=list)
```

### 实现方式

#### 方式：coder 节点内多方案运行

在 `coder_node`（`nodes/coder.py`）中，主方案代码运行完成后，为每个对照方案写一段独立的 Python 代码并沙箱执行：

```python
# Phase 2 伪代码（coder.py 增量）
for baseline_spec in BASELINE_SPECS:  # 3 个预设方案模板
    baseline_prompt = build_baseline_prompt(
        problem=state.problem,
        main_solution_code=last_successful_code,
        spec=baseline_spec,
    )
    # LLM 生成对照方案代码
    baseline_code: str = complete(baseline_prompt, ...)
    # 沙箱执行
    result = run_python(baseline_code)
    # 提取 RESULT: 行
    extracted = extract_numeric_results(result.stdout)
    baseline = BaselineResult(
        name=baseline_spec.name,
        category=baseline_spec.category,
        metrics=extracted,
        stdout=result.stdout,
        success=result.success,
        stderr=result.stderr,
    )
    state.baseline_results.append(baseline)
```

### 对照方案模板（3 个）

#### 方案 1：无调度基准
- **生成指令**：用与主方案相同的数据和预测输入，但完全不执行调度优化——所有决策变量取默认值 / 不调整。运行后输出 `RESULT: baseline=no_schedule total_cost=... service_rate=...`。
- **LLM 提示词关键**："把主方案代码中的优化求解步骤全部删除，改为'不调整/保持现状'。保留数据加载和评估逻辑。"

#### 方案 2：简单平均预测
- **生成指令**：把主方案中的 XGBoost/STGNN/分位数回归预测器替换为简单历史均值预测。其余逻辑（调度/评估）不变。输出 `RESULT: baseline=simple_avg_pred total_cost=... service_rate=...`。
- **LLM 提示词关键**："把预测函数替换为 `np.mean(historical_data, axis=0)`。保持调度代码不变。"

#### 方案 3：贪婪启发式
- **生成指令**：把主方案中的随机规划/滚动优化求解器替换为贪心策略（如"每次取当前最大需求的节点优先满足"）。输出 `RESULT: baseline=greedy_heuristic total_cost=... service_rate=...`。
- **LLM 提示词关键**："把优化求解器替换为 while 循环 + 贪心选择规则。保持数据加载和评估逻辑不变。"

### 失败处理

- 任一对照片运行失败 → `BaselineResult.success = False`，`table_assembler` 表格中该行填 `"运行失败"` + warnings 记录。
- 3 个方案串行运行（不并行，避免沙箱竞争），每个方案限时 90 秒。
- 任一方案失败不阻断主流程（graph 不中断）。

---

## F. 数字结构化提取与可追溯（Phase 2）

### 目标

让论文中的每个数字都能追溯到具体的代码运行输出行，而不是靠 LLM 自觉 + PaperCritic 事后抽查。

### 现状

- `sensitivity.py:47` 已有 `_RESULT_RE = re.compile(r"RESULT:\s*parameter=(\S+)\s+values=(\[[^\]]+\])\s+results=(\[[^\]]+\])")` ——但只用于 sensitivity 节点的内部解析。
- 其他 code_artifacts 的 stdout 没有被结构化提取——writer 拿到的是 `a.stdout[-500:]` 文本块，靠 IRON RULE 1（"禁编造"）让 LLM 从中取数字。
- PaperCritic 做事实核查是基于语义匹配（"正文数字是否在 stdout 中出现"），而非结构化校验。

### Phase 2 方案

#### 通用 RESULT: 提取器

在 `tools/runner.py` 新增函数：

```python
# 扩展 RESULT 匹配格式
_RESULT_GENERIC_RE = re.compile(
    r"RESULT:\s*"
    r"(?:baseline|scenario|method|config)=(\S+)\s+"  # 可选标识前缀
    r"(.*?)\s*$",                                      # 剩余部分为 key=value 对
    re.MULTILINE
)

_RESULT_PAIR_RE = re.compile(r"(\w+)=([0-9.eE+\-]+|\[[^\]]+\])")

def extract_numeric_results(stdout: str) -> dict[str, dict[str, float]]:
    """从 stdout 提取所有 RESULT: 行，返回 {identifier: {metric: value}} 映射。
    
    支持格式：
      RESULT: parameter=alpha values=[0.1,0.5,1.0] results=[10,20,30]
      RESULT: baseline=no_schedule total_cost=1245.3 service_rate=0.82
      RESULT: scenario=high_demand objective=9876 solve_time=12.3
    """
    ...
```

#### 在 coder 节点中调用

`coder_node` 完成代码运行后（主方案 + Phase 2 的 3 个对照方案），对每个 `CodeArtifact.stdout` 调用 `extract_numeric_results()`，产出 `state.numeric_results`。

#### 在 writer prompt 中注入数字→来源映射

Phase 2 的 writer 分节模板新增：

```
## 可追溯数字清单（以下数字可直接引用，标注来源即可）
{% for nr in numeric_results %}
- {{ nr.identifier }}: {{ nr.metrics }}
  （来源：代码运行结果第 {{ loop.index }} 组）
{% endfor %}

铁律（强化版 IRON RULE 1）：正文中的所有定量结果必须来自上方「可追溯数字清单」。
如果某个数字不在清单中，改为定性表述。数字后面标注来源编号，如 `总成本 1245.3（来源 1）`。
```

#### 在 PaperCritic 中检查

Phase 2 不改变 PaperCritic 的现有实现（LLM 语义匹配），但加上结构化预警：如果 `paper.model_section` 或 `paper.solution` 中包含不在 `numeric_results` 中的数字，生成一条 `CriticIssue(section="general", problem="正文数字 X 不在代码运行结果中，可能为编造")`。

这个检查在 `table_assembler` 节点中做（数字提取在这里），不在 PaperCritic 中做。

---

## G. 精选附录（两期均涉及）

### 现状

- `gmcm.tex.j2:76-94`：附录 = 所有 `code_artifacts`（`success=True`）的完整代码 + 完整 stdout（`verbatim`），不截断。
- `paper.md.j2:36-51`：附录 = **所有** artifacts（含失败）的完整代码 + 完整 stdout + stderr。
- 两个模板都无摘要、无说明、无精选。

### Phase 1 改动

#### gmcm.tex.j2 改动

```latex
{% if code_artifacts %}
\newpage
\appendix
\section{附录：关键算法代码与说明}

{% for art in code_artifacts %}
{% if art.success %}
\subsection{ {{- art.purpose -}} }
{{ art.purpose | pymarkdown_as_latex }}  % 目的说明转为 LaTeX 段落

\textbf{核心代码段：}
\begin{lstlisting}[language=Python]
{{ art.code | truncate_lines(80) }}  % 最多 80 行
\end{lstlisting}

{% if art.stdout %}
\subsubsection*{关键输出摘要}
\begin{verbatim}
{{ art.stdout | extract_key_output }}
\end{verbatim}
{% endif %}

{% endif %}
{% endfor %}
{% endif %}
```

**需要新增的 Jinja2 过滤器**（在 `latex.py` 的 `_env` 中注册，Phase 1 实现，各 ~15 行）：
- `truncate_lines(lines)`: 取代码前 N 行（优先保留 import + 核心函数定义，跳过连续空行和纯注释块）
- `extract_key_output`: 从 stdout 中挑关键行——`RESULT:` 行 + 最后 5 行

这两个过滤器当前**不存在**，是 Phase 1 的新增工作项。另外 `paper.tex.j2`（默认模板，L66 直接 `\end{document}`）当前完全没有附录区块——附录对它是纯新增，不是改动。`gmcm.tex.j2:76-94` 已有附录区块，是改动。

注意：附录的目的说明用纯文本，不走 markdown→LaTeX 转换（避免引入不存在的 `pymarkdown_as_latex` 过滤器）。

#### paper.md.j2 改动

```
## 附录 A. 关键算法代码

{% for a in code_artifacts %}
{% if a.success %}
### A.{{ loop.index }} {{ a.purpose }}

**说明**：{{ a.purpose }}

**核心代码**（关键算法段，非完整运行）：
```python
{{ a.code | truncate_lines(60) }}
```

**关键输出摘要**：
```
{{ a.stdout | extract_key_output }}
```
{% endif %}
{% endfor %}
```

### Phase 2 增强

Phase 2 使用 `extract_numeric_results()` 提取的 `NumericResult` 做输出摘要——附录中 stdouthighlight 的是结构化提取出来的数字行，不是简单的尾部截断。这样"关键输出摘要"和正文中的"可追溯数字清单"对应同一组结构化数据。

---

## H. 测试与验证方式

### Phase 1 验证

#### 单元测试

| 测试项 | 文件 | 方法 |
|---|---|---|
| `_clean_forbidden_words` | `tests/test_table_assembler.py` | 输入含 PaperCritic/Claim/Evidence 的文本，验证被替换后的输出，断言 warnings 列表正确 |
| `_generate_param_table` | 同上 | 给定模拟 variables dict，验证输出的 markdown 表格行数、列名 |
| `_generate_sensitivity_table` | 同上 | 给定 3 个 SensitivityRun，验证表格格式和敏感性评级计算 |
| `_inject_table`（去重） | 同上 | 给定已含同类表格的文本，验证不重复注入 |
| `truncate_lines` 过滤器 | `tests/test_latex.py` | 给定 200 行代码，验证截断到指定行数且保留 import + 核心逻辑 |

#### 集成测试（e2e）

- 使用现有的 `scripts/e2e_plan_d.py` 模式（mock LLM，真实 graph）
- 新增 `scripts/e2e_table_assembler.py`：
  - 用 mock 数据构造 `MathModelingState`（含 variables, sensitivity_runs, code_artifacts, paper sections）
  - 运行 `table_assembler_node`
  - 验证输出 `paper` 中 7 个字段含预期的表格（正则匹配 `| --- |` 行）
  - 验证 `table_warnings` 非空（含清洗记录）
  - 验证禁用词已被替换

#### 回归测试

- 运行 `scripts/e2e_plan_d.py` 确保 graph 拓扑变化（新增 table_assembler 节点）不破坏现有流程
- `bench/runner.py` 的 `run_bench()` 应该仍然通过（`table_assembler` 不改变 evaluator 打分基准）

### Phase 2 验证

#### 单元测试

| 测试项 | 文件 | 方法 |
|---|---|---|
| `extract_numeric_results` | `tests/test_runner.py` | 给定多种 stdout 格式（RESULT: parameter=..., RESULT: baseline=..., RESULT: scenario=...），验证提取出的 dict 结构正确 |
| baseline prompt 构建 | `tests/test_coder.py` | 给定 mock problem + main code，验证 3 个 baseline prompt 各自包含正确的替换指令 |

#### 集成测试

- mock coder 节点的对照方案执行路径（在 mock sandbox 中返回预设 stdout），验证 `BaselineResult` 被正确写入 state
- 验证 table_assembler 的对比表（表 3）从 `baseline_results` 正确填入数据

---

## 分阶段交付时间线

| 阶段 | 内容 | 预估改动量 | 风险等级 |
|---|---|---|---|
| **Phase 1** | 呈现层重写（8段）+ table_assembler（4张表+清洗）+ 附录精选 + writer模板重写 | ~500 行新代码 + ~200 行模板改动 | 低（无新代码运行） |
| **Phase 2** | 基准对照实验 + 数字结构化提取 + writer/critic 追溯增强 + table_assembler 扩展表 3 | ~400 行新代码 + ~100 行模板改动 | 中（LLM 自动生成对照代码可能失败） |

**Phase 1 可在 Phase 2 之前独立验证并合并**：改动范围限定为呈现层 + writer prompt，所有现有 e2e 和 bench 测试仍应通过。

---

## 风险评估

| 风险 | 严重度 | 缓解措施 |
|---|---|---|
| 8 段结构在某些题目类型上不适用（非预测+调度型的纯评价/分类题） | 中 | `writer_section_model.md.j2` 的结构指令加条件："若题目无调度需求，将'优化调度'段改为'评价/分类模型优化'；若题目无不确定性需求，将'动态扩展'段改为'模型泛化性讨论'" |
| LLM 对照方案代码跑不通 | 高（仅 Phase 2） | 对照方案失败不中断流程；表格中填 `"运行失败"`；最多 2 次重试；3 种方案至少 1 种成功即算达标 |
| 表格追加在 section 末尾可能被 LaTeX 渲染到错误位置 | 低 | 约定：表格前加 `\clearpage` 或分页提示（LaTeX 模板层面） |
| `extract_numeric_results` 的正则格式覆盖不全 | 中（仅 Phase 2） | 先定义 3 种主要格式模板，在 sensitivity.py 的现有基础上扩展；新增格式通过分析实际 runs/eval_v11 的 stdout 逐条测试 |
| 禁用词清洗太激进误伤正常内容 | 低 | 清洗只在 `table_assembler` 节点做（不在 writer 生成中拦截）；warnings 日志可审阅；每个替换都会记录以便回滚 |
| writer prompt 大幅重写导致内容质量反向下降 | 中 | Phase 1 通过手动对比新旧 prompt 在同一题目上的输出（A/B 测试）来验证 |

---

## 开放问题

1. **`writer_section_model.md.j2` 的 8 段指令是否该按题目类型条件化？** 当前设计是固定的三段式（基础预测/优化调度/动态扩展）。如果题目是纯分类/评价问题，可能需要不同的段模板。建议：Phase 1 先用固定三段，从 2-3 个真实题目运行中观察适用性，Phase 2 时决定是否需要 `analyst` 节点额外输出"推荐论文章节结构"来驱动条件化。

2. **对照方案代码的 LLM 生成成功率到底有多高？** 这是 Phase 2 最大的不确定性。建议：在正式纳入 coder_node 之前，先用 3 个现有 bench 题目（2022_A, 2023_B, 及一个新题）脱机测试 baseline prompt 生成能力，评估成功率。

3. **是否需要新增一个"对比分析写手"子节点？** Phase 2 的对照表格生成是纯代码（从 `BaselineResult` 到 markdown 表格），但表格之前的 prose 段落（"从对比结果可以看出..."）仍需要 writer 在 `solution` 字段中生出来。考虑在 Phase 2 的 writer prompt 中增强对对比结果的要求，还是在 writer/critic 循环之后新增一个 "baseline_writer" 节点？建议：Phase 2 先走 prompt 增强路线，避免 graph 拓扑继续膨胀。

---

## 审批检查清单

审批通过后，下一步：
- [ ] 确认设计文档中所有改动范围无遗漏
- [ ] 确认两期边界清晰（Phase 1 可独立合并）
- [ ] 确认建模层零改动
- [ ] 调用 `writing-plans` skill 生成实现计划
