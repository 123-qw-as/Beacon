# Beacon 真实数学建模全流程故障根因、修复与验收报告

日期：2026-07-17  
真实题目：`C:\Users\lwh86\Downloads\A题：城市绿色物流配送调度_1776993150913`  
最终运行：`runs/green-logistics-rootfix-v4-20260717`  
验收结论：本次运行通过公开 `math-agent start` 正常入口达到 `completed`，不是“文件生成了但证据不足”的 `degraded`。主方案、三个基线、三组敏感性、模型—代码一致性、论文评审、综合评价、两遍 LaTeX 编译和 PDF 逐页视觉检查均已完成。

历史说明：旧运行 `green-logistics-rootfix-v3-20260716` 的 PDF 使用最新正式敏感性证据，但备查 `paper.md` 曾混入 append-only 历史。代码修复 Markdown/TeX 正式证据选择和 finalizer 门禁后，v4 从新目录、同一真实附件重新走完整流程，正式 Markdown、TeX、PDF 均不再含旧数值。视觉复审发现的半句图注与成本图图例拥挤又经确定性重排修复；该重排只重跑原 latex/finalizer 节点，不调用 LLM、不重算或替换主方案指标。

## 1. 调查结论摘要

本次故障不是 Ollama 单点故障，也不是把某个 timeout 调大即可解决。真实原因是多个边界问题在长链路、连续节点和代码生成负载下叠加：

1. 上游确实会返回 502、429、断连或不返回；旧实现又把不可取消的 SDK 调用留在主进程中，导致“请求已经超时，但执行线程仍在后台占用连接和内存”。
2. 9router 能在账号间 fallback，但上游连接失败、单账号限流和并发配额仍会暴露给客户端；Beacon 的几十次连续长请求把单次低概率故障放大成流程级高概率故障。
3. 监督状态是持久化快照，不等同于活进程；旧状态没有同时校验 PID 身份、创建时刻、心跳和已验证完成标记。
4. coder 旧重试只传 stderr，不传上一版代码、错误类型和真实附件 schema，导致每轮从零生成，重复引入字段、路径、时间和索引错误。
5. 原城市物流脚本把订单按客户聚合，却不拆分超过单车容量的客户；当一轮没有可服务客户时，外层集合不减少，形成无进展循环。
6. 退出码 0 不能证明数值成功；只检查“存在 RESULT”也不能证明脚本读过附件或指标口径一致。
7. 历史主求解、补充图、基线和敏感性 artifact 混在一个列表中；若不按证据角色和最新批次选择，旧值或补图近似值会污染论文。

修复原则是给每一层建立硬边界和可验证契约，而不是无限增加 timeout、重试数或降低质量门槛。

## 2. 完整调用链

### 2.1 CLI / Web UI 到上游模型

调用链如下：

`Web UI frontend/app.js:296`  
→ `POST /api/run`，`frontend/server.mjs:530`  
→ 以子进程执行 `math_agent.cli supervise`，`frontend/server.mjs:598,624`  
→ CLI `supervise` / `start` / `recover`，`src/math_agent/cli.py:470,611,696`  
→ 进程监督器，`src/math_agent/supervisor.py:84,361,431`  
→ worker 子进程执行 `run` 或 `recover`，`src/math_agent/supervisor.py:327,385,407`  
→ SQLite checkpoint 与恢复，`src/math_agent/checkpointing.py`、`src/math_agent/run_lock.py`  
→ LangGraph 节点，`src/math_agent/graph.py:77-102`  
→ coder 或其他业务节点调用 `complete`，例如 `src/math_agent/nodes/coder.py:1134`、`src/math_agent/nodes/writer.py:39`  
→ LLM 逻辑调用的单次/总预算，`src/math_agent/llm.py:257,303-401`  
→ 可强杀 worker transport，`src/math_agent/transport.py:219-274`  
→ worker 内 LiteLLM OpenAI 兼容请求，`src/math_agent/llm_worker.py:61`  
→ `http://localhost:20128/v1`  
→ 9router 的 `/v1/* → /api/v1/*` rewrite，`9router/app/server.js:6`  
→ chat completions route，`9router/app/.next-cli-build/server/app/api/v1/chat/completions/route.js:1`  
→ provider/credential 选择、账号 fallback 与上游调用，`9router/app/.next-cli-build/server/chunks/112.js:1` 的模块 `79489`  
→ 实际上游模型。

CLI 直接运行从 `src/math_agent/cli.py:327` 进入相同链路；Web UI 没有另一套求解实现，因此 CLI、Web 和公开状态结构保持兼容。

### 2.2 生成代码到论文与 PDF

`coder_prepare`  
→ `coder_generate`，`src/math_agent/nodes/coder.py:1134`  
→ `coder_execute`，`src/math_agent/nodes/coder.py:1242`  
→ `run_python` 创建隔离子进程，`src/math_agent/tools/runner.py:292,318`  
→ 监控整个后代进程树 RSS、120 秒期限和 2 GB 内存上限，`src/math_agent/tools/runner.py:251-270,282,312-353`  
→ stdout/stderr 与实际文件读取审计  
→ `validate_numeric_results`，`src/math_agent/tools/runner.py:450`  
→ `CodeArtifact(evidence_role=primary|baseline|supporting)`  
→ 模型—代码一致性  
→ 基线独立重跑  
→ 敏感性代码、中心点对齐与 `SensitivityRun`  
→ writer 分节 checkpoint，`src/math_agent/nodes/writer.py:39,646`  
→ paper critic / evaluation / human review  
→ table assembler 只消费已验证结构化字段  
→ LaTeX 正式证据选择与编译，`src/math_agent/nodes/latex_node.py:24-204`、`src/math_agent/tools/latex_compile.py`  
→ finalizer 原子提交并写入带哈希的 `completion.json`，`src/math_agent/nodes/finalizer.py:236`。

## 3. 分项根因、证据与修复

### 3.1 间歇性 502、429、断连和长时间不返回

运行证据：

- `runs/full_pipeline_run4.log:94`：`502 fetch connect timeout`。
- `runs/full_pipeline_run11.log:100`：`502 ... ECONNRESET`。
- `runs/huazhong-green-logistics-a-live-flash/supervisor.log:4`：上游 socket 被关闭。
- `runs/ui-server/ui-1783352427189/run.log:4`：`429 concurrency reached, current: 9, limit: 8`。
- `runs/huazhong-green-logistics-a-live-fixed-model/trace.json`：3 次 LLM timeout。
- 最终运行 `trace.json` 共 75 个逻辑调用、83 个 attempt；最长单次记录 120144 ms，`coder_generate` 累计 155578 ms。即使单次成功率较高，长链路仍会显著放大尾延迟和偶发失败。

9router 代码证据：`chunks/112.js:1` 模块 `79489` 会选择 provider credential，失败后调用 `shouldFallback`，若允许则把 connection 加入已尝试集合并换下一个账号；所有账号 rate limited 时返回上次错误与 retry-after。它能做账号级容错，但不能保证所有上游都健康，也不能消除外部并发配额。

Beacon 旧根因：SDK 调用和主图执行处在同一长寿命进程中，Python 线程无法可靠中止底层阻塞网络调用；超时后重复调用会叠加“幽灵请求”。同时旧重试没有统一总预算，节点级重试、SDK 重试和 router fallback 可能相乘。

修复：

- `src/math_agent/llm.py:303-401` 为每个逻辑调用设置 profile 级单次期限和总硬期限；总预算包含退避等待，预算不足时不再发新请求。
- timeout 不自动在 fallback 模型重放，避免上游仍执行时产生重复长请求；明确 5xx、429、断连、timeout 的分类。
- `src/math_agent/transport.py:219-274` 把 LiteLLM 放到 spawn worker；父进程 `poll(timeout)`，超时后 close → terminate → kill → join，下一次惰性启动新 worker。
- `src/math_agent/llm_worker.py:61` 把 SDK 自身 retry 设为 0，防止隐藏重试与 Beacon 总预算相乘。
- `src/math_agent/llm.py` 把模型失败计数和 wall-clock 冷却期限原子写入当前 run 的 `llm_model_health.json`；supervisor 从 checkpoint 启动新 worker 后先加载该状态，把仍在冷却的模型移到候选末尾。旧版仅用进程内字典，恢复 worker 会遗忘熔断并再次命中同一挂起模型，这是 v4 初次 blocked 暴露出的最后一个 transport 根因。
- supervisor 只对可恢复故障在同一 checkpoint 恢复，并有同节点失败上限；不会通过无限重试掩盖持续故障。

v4 `trace.json` 记录 67 个逻辑调用、83 个物理 attempt、6 次失败，其中 3 次 `LLMTimeoutError` 均约 124 秒，另有 3 次 `LLMServerError`（约 7.5 秒、234.4 秒、234.7 秒）。流程仍完成，证明修复目标不是假设上游永不失败，而是让失败受硬期限、分类、跨 worker 模型冷却、回收和定点恢复约束。

### 3.2 supervisor 显示 running，但进程已经退出

根因：`supervisor.json` 是最后一次原子写入的状态快照。进程被外部终止、系统重启或 PID 被复用后，文件不会自动变化；只看字符串 `running` 会产生陈旧状态。

修复：

- `src/math_agent/supervisor.py:203-220` 校验 supervisor PID 是否仍存活，并结合 `started_at` 校验进程身份，兼容旧状态文件。
- `src/math_agent/cli.py:696-738` 的只读 `status` 同时读取 checkpoint、supervisor 和经过哈希验证的 completion；若监督状态与已验证完成状态冲突，标记 `stale`，不篡改历史文件。
- worker 与 supervisor 各有目录锁，避免同一输出目录被两个活进程并发写入。

最终现场证据：

- `supervisor.json` 记录 supervisor PID 22708、worker PID 3004、最后心跳 `2026-07-17T01:59:14.616309+00:00`，状态和结束时间均已原子收敛为 completed。
- supervisor、worker 及 transport 后代 PID 6732、27000、27576 当前均不存在。
- `math-agent status` 输出 `checkpoint: yes`、`next_node: -`、`supervisor_status: completed`、`completion: completed`；没有把死 PID 误报为仍在运行。

### 3.3 coder 重试反复从头生成

根因：旧队列只保留 stderr；模型看不到上一版代码、失败阶段、真实字段和已经完成的工作，因而每次重写整段脚本。代码生成输出更长，重写又提高超时概率，并反复引入新错误。

修复：

- checkpoint 保存 `coder_pending_draft`、`coder_previous_code`、`prev_kind`、当前 work item 和已提交 artifacts，恢复只重做未完成项；状态字段在 `src/math_agent/state.py:346` 附近。
- `src/math_agent/nodes/coder.py:862-1001` 按 timeout/runtime/output_validation 分类；优先对上一版代码做安全局部修补，提示真实附件 schema、路径和失败 stdout/stderr。
- 对已知安全修复（整数距离索引、Windows 路径、聚合超容量、服务时间等）本地 patch，不再额外调用 LLM；无法安全修补时才让模型基于上一版代码修复。
- writer、sensitivity 和 figure 也拆成可 checkpoint 的细粒度步骤，避免一个后段失败重做整个论文。

### 3.4 脚本超时、无限循环和内存增长

城市物流直接根因：订单按客户聚合后，部分需求超过单车容量；候选筛选永远拒绝这些客户，而外层 `while remaining` 没有检查集合是否减少，导致 120 秒后才被杀。

修复：

- 正式求解器按题面最小车型容量拆分聚合需求；每个 task 都能装入至少一种车型。
- 每次选车后断言 `len(remaining)` 严格减少；没有候选或车队耗尽立即抛出 `no progress/fleet exhausted`。
- 使用题面五类有限车队 `60/50/50/10/15`，不再把车辆上限随意放大到 10000。
- `runner.py:334-353` 同时检查 wall-clock 和整个后代树 RSS；默认 120 秒、2048 MB。越界即回收整树，而不是只杀父 Python。
- `runner.py:251-270` 使用 psutil 遍历 descendants，先 terminate 再 kill，并等待回收。

最终主脚本在 `final_state.json` 的最新 primary artifact 中保留完整代码；其中超容量拆分、无进展异常和严格减少断言均可直接审计。最终 stdout 无 stderr，`service_rate=1.0000`。

### 3.5 数据字段、路径和类型错误

根因包括：模型猜字段名、Windows 反斜杠转义、子进程 cwd 改变后相对路径失效、Excel 时间值有 `datetime.time` 与 `HH:MM` 字符串两种形态、距离矩阵 index 为整数而生成代码用字符串索引。

修复：

- prompt 数据提示读取真实列名、dtype 和少量样本，相关实现为 `src/math_agent/prompts/_data_hint.py`。
- 使用 `Path` 与经过转义的绝对数据目录；runner 先 `resolve()` Python 和工作目录。
- 时间转换同时支持有 `hour/minute/second` 属性的对象和字符串。
- 订单、客户 ID、距离矩阵行列显式转为整数；数值字段使用 `to_numeric(..., errors=...)`。
- actual read audit 通过注入读取记录器保存真实打开路径，不能仅凭源码出现 `read_excel` 判断。

最终 primary 的 `read_paths` 精确包含四个原始附件：订单信息、距离矩阵、时间窗、客户坐标信息。附件只读，四个文件的最后修改时间仍为 `2026-07-11 21:31:10`。

### 3.6 硬编码、全零、负数和口径不一致结果

根因：退出码 0 只表示 Python 进程正常退出。脚本可能捕获异常后打印“失败”，也可能不读取附件却打印表面合理的常量；只做语法检查无法发现这些情况。

修复后的 RESULT 契约：

- stdout 出现 traceback、error、failed、无法求解等失败声明时，即使 exit 0 也失败。
- primary 至少要求成本、车辆数、服务率、碳排放等关键指标；拒绝 NaN/Inf、全零、负成本/时间/距离、越界比例和超过输入规模的车辆数。
- `validate_code_data_usage` 同时做静态数据使用检查和实际 read-path 审计；“先读文件再打印常数”也会被拒绝。
- 核对成本分项、服务覆盖、有限车队、动态距离变化等领域边界。
- `final_state.json` 保留了被拒绝的历史证据，例如旧 greedy 的 `dynamic_distance_change=-28.2828` 对应 stderr 为“指标不应为负数”；正式选择只取后续通过校验的 artifact。
- finalizer 要求当前主证据、有效基线、非空敏感性、评分、当前编译日志和 PDF 哈希全部成立，历史失败不会被删除，但也不会进入正式结果。

### 3.7 主求解、补充图和基线职责混淆

根因：旧 `code_artifacts` 只按时间追加；补图脚本可能重新计算或近似主指标，基线可能复用主 stdout，旧轮次也可能比新轮次更早被选中。

修复：

- `primary`：唯一可定义论文主指标的正式求解；必须读附件并通过 RESULT、约束和一致性校验。
- `baseline`：在同一附件和同一指标口径下独立执行，不得复制 primary stdout；本次为 no_schedule、simple_pred、greedy。
- `supporting`：只能从已验证 primary 的 RESULT/BREAKDOWN 或 SensitivityRun 数组渲染图，不能重新求解或覆盖主指标。
- `temporary` / attempt 目录：保留诊断价值，不进入正式论文。
- `latex_node.py:24-57,131-204` 按参数取最新有效敏感性，按 evidence role 选择正式图；`table_assembler.py` 和 writer 同样只消费最新验证值。

最终论文的成本图来自主 RESULT 的 BREAKDOWN，敏感性图来自三个通过中心点对齐的数组；补充图没有替换主方案指标。

## 4. 稳定复现与回归测试矩阵

| 场景 | 复现方法与断言 | 自动化证据 |
|---|---|---|
| 连续请求下 502 / 模型切换 | fake transport 连续注入 502、成功和候选模型；断言分类、预算和选路 | `tests/test_llm.py`、`tests/test_supervisor.py:17`、`tests/test_router_error_classification.py:9` |
| supervisor 恢复后的模型冷却 | 第一 worker 超时并持久化健康状态，模拟新 worker 清空内存后读取同一 run，断言优先 backup | `tests/test_llm.py::test_timeout_cooldown_survives_worker_recovery_and_promotes_fallback` |
| 429 | 注入 rate-limit，断言不会当普通 runtime；遵守 retry-after 与总预算 | `tests/test_llm.py`、`tests/test_retry.py`；真实证据 `runs/ui-server/ui-1783352427189/run.log:4` |
| 上游永不返回 | fake HTTP server 不返回；断言单次硬期限、总硬期限和下一请求可成功 | `tests/integration/test_llm_transport.py:164,193`、`tests/test_llm_phase_a.py:37,106` |
| worker 整树回收 | hang 后检查旧 worker 退出且新 worker 可用 | `tests/integration/test_llm_transport.py:164` |
| 脚本及后代进程回收 | 脚本再 spawn 睡眠子进程，超时后确认 descendants 全部消失 | `tests/test_runner.py:316` |
| 超容量客户无进展 | 构造聚合需求超过容量，断言拆分且 remaining 严格减少 | `tests/nodes/test_coder.py:256,389,445` |
| 2 GB / 120 秒 | 分配内存或无限睡眠，断言 resource/timeout 分类和整树终止 | `tests/test_runner.py:337`、`tests/nodes/test_coder.py:295,304` |
| exit 0 但 stdout 声明失败 | `print('FAILED...')` 后 exit 0，必须重试/拒绝 | `tests/test_runner.py:116`、`tests/nodes/test_coder.py:625`、`tests/nodes/test_sensitivity.py:100` |
| 表面合理但未读附件 | 打印正常常量 RESULT，或读文件后仍打印常量，必须拒绝 | `tests/test_runner.py:205,226`、`tests/nodes/test_coder.py:687` |
| 负时间、越界比例、异常车辆数 | 参数化构造非法指标并断言具体拒绝原因 | `tests/test_runner.py:143,179,192`、`tests/nodes/test_coder.py:651,724` |
| checkpoint 定点恢复 | 在 modeler/coder/sensitivity/figure/writer 中断，已完成项不重做 | `tests/integration/test_granular_recovery.py`、`tests/nodes/test_writer_recover.py:116` |
| 陈旧 supervisor | 写入 running + dead PID，或 completion 与 supervisor 冲突 | `tests/test_cli.py:174,196` |
| 正式/临时 artifact 隔离 | 旧 supporting 值、旧 sensitivity、baseline code 不得进入论文 | `tests/nodes/test_figure_pipeline.py:113`、`tests/nodes/test_writer.py:91,123`、`tests/nodes/test_latex_node.py:137,169` |
| 最终提交完整性 | 修改已提交 artifact 后 completion 哈希验证失败 | `tests/nodes/test_finalizer.py:77,94,135,149,157` |

全量命令：`uv run pytest -q`。最终结果为 `574 passed, 4 skipped`；4 个 skip 是显式可选的 live 外部集成测试，不是失败。Web UI 另以 `npm.cmd test -- --run` 验证 13 项，全部通过。测试结束后没有遗留 Python/pytest worker。

## 5. 真实题目最终运行证据

### 5.1 checkpoint、监督状态和 LLM 耗时

初始 v4 运行在 `analyst` 连续三次触发约 120 秒硬期限并进入 blocked。新增 run-scoped 模型健康文件后，从同一 checkpoint 以正常 supervisor 恢复；恢复 worker 先选健康 fallback，完成 analyst 和后续全部节点，没有重建输出目录或从头伪造状态。coder 首版真实读附件后因超容量客户失败，第二版仍拒绝客户 4；checkpoint 明确保留 `prev_code` 与 stderr，后续正常 coder 批次生成可拆分、带无进展断言的正式主求解器。

最终状态：

- checkpoint 存在，`next_node=-`，说明图没有待执行节点。
- `completion.json.status=completed`，issues/warnings 均为空。
- `supervisor.json.status=completed`；记录的 supervisor/worker PID 和 transport 后代均已退出。
- `trace.json`：67 次逻辑 LLM 调用、83 次 attempt、6 次失败、3 次 timeout；3 次 timeout 均来自初始 `deepseek-v4-pro` analyst 请求，恢复后出现的 3 次 server error 由模型切换在总预算内消化。
- 最慢成功 long attempt 为 180957 ms；最长 `modeler` 节点 418142 ms，仍受 420 秒逻辑总预算约束。多个 `coder_generate` 节点为 48—122 秒，说明代码生成仍是尾延迟高发阶段，但没有突破 240 秒总预算。

### 5.2 主方案与数据血缘

正式主方案 RESULT：

```text
RESULT: baseline=ours total_cost=144586.99 vehicles=159 service_rate=1.0000 total_carbon=14634.14 total_distance=22054.37 fuel_vehicles=134 ev_vehicles=25 avg_delivery_time=601.95 timewin_rate=0.8944 fuel_ratio=0.8428 response_time=0.027778 dynamic_reinserted=1 dynamic_distance_change=28.2828 dynamic_distance_improved=1
BREAKDOWN: Z_fix=63600.00 Z_wait=20819.51 Z_late=6718.39 Z_energy=43936.90 Z_carbon=9512.19
```

主 artifact 的 `evidence_role=primary`、`success=true`，stderr 为空，实际读取审计包含四个附件。原始附件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| 客户坐标信息.xlsx | `570d846d2cc85f6d45df21161a1df015cd5320cb22ac2885fc42931dd24d7db7` |
| 时间窗.xlsx | `5f20976494bf618847251dec221042c8d984b5844614ad71c2a62546da3f53c9` |
| 订单信息.xlsx | `ff6f512fd4b2bd2d13fd9cbaccc98b9354be80c229ac394008dbc7c73cdf8e5d` |
| 距离矩阵.xlsx | `198f69aef8f9145a5e548f2a015e08cc7a84d8b761decf3cc663dea6ef579a7e` |

### 5.3 基线与敏感性

三个有效、独立执行的基线：

| 方案 | 总成本 | 车辆数 | 碳排放 | 燃油/新能源 | 服务率 | 时间窗满足率 |
|---|---:|---:|---:|---:|---:|---:|
| no_schedule | 148625.41 | 159 | 15705.40 | 143/16 | 1.0000 | 0.9155 |
| simple_pred | 144786.33 | 159 | 14580.08 | 134/25 | 1.0000 | 0.8979 |
| greedy | 155919.78 | 165 | 16767.86 | 160/5 | 1.0000 | 0.8697 |

三个最新正式敏感性数组，中心点均严格复现 144586.99：

- 速度比例 `[0.8, 1.0, 1.2]` → `[146017.04, 144586.99, 145204.85]`。
- 限行开始时间 `[7, 8, 9]` → `[143559.50, 144586.99, 144975.48]`。
- 晚到惩罚 `[0.625, 0.833333, 1.041667]` → `[142907.40, 144586.99, 146266.59]`。

### 5.4 质量评分

- 最新模型—代码一致性：8，approved。
- 最新论文评审：8，approved。
- 综合评价：假设 8、创新 7、结果 7、写作 7、深度 6，overall 7.1。
- human decision：approved。
- finalizer：completed，issues/warnings 为空。

### 5.5 LaTeX 与 PDF

`compile.log` 第 2 行 `[pass 1] exit=0`，第 117 行 `[pass 2] exit=0`；视觉修复后的两遍均输出 9 页 PDF。日志没有 Overfull、Missing character 或 LaTeX Error。存在少量 Underfull 提示，不造成截断或不可读内容。

最终文件：

- `paper.md`：23562 bytes，SHA-256 `fa48d2935bd74e5c35921d158e1e62785f0fb533dcbfa6aea353c82e13416f59`。
- `paper.tex`：24496 bytes，SHA-256 `c7c51f7023484104d5b6d68a6184cd9bee205eeb6f315b077f06ffd35d52cc3a`。
- `paper.pdf`：1767661 bytes，SHA-256 `1fdf5b2af3d797642511b8b4184c471cc3d114dda112b18ca28c7d4451a7ad85`。
- `compile.log`：10941 bytes，SHA-256 `1a2d74ea30b6f9599316103fe87b006de407dd891c713905903dd5b4420f29df`。

按 PDF skill 的要求，最终 PDF 已用 Poppler 以 140 DPI 渲染全部 9 页到 `tmp/pdfs/green-logistics-v4-fixed/page-1.png` 至 `page-9.png`，并逐页人工视觉检查。初次视觉检查发现成本饼图图例拥挤、两条 caption 在小数中间硬截；修复图注边界和 `cost_pie` 正式重绘匹配后再次检查全部页面。最终结果：无空白页、裁切、遮挡、乱码、重叠或半句图注；主路线图、成本图、时间窗图、三张敏感性图、表格和附录代码均可读。

## 6. 完成标准逐项核对

- [x] 两条完整调用链及源文件位置已确认。
- [x] 502/429/断连/不返回、陈旧监督状态、重写式 retry、脚本无进展、字段路径类型、伪数值和证据职责混淆均有独立根因。
- [x] checkpoint、PID/心跳、LLM 耗时、脚本与 stderr、RESULT 拒绝原因、artifact 角色均有代码或运行证据。
- [x] 用户列出的全部复现场景均有自动化测试；Python 全量 574 通过、4 跳过，Web UI 13/13 通过。
- [x] 使用指定真实题目附件完成完整运行，未修改附件。
- [x] 主方案真实读取四个附件，关键 RESULT 指标齐全。
- [x] 至少两个有效基线；实际为三个。
- [x] 三组敏感性非空且中心点对齐。
- [x] 模型—代码一致性 8，论文评审 8，综合 7.1。
- [x] 最终状态 `completed`，不是 `degraded`。
- [x] LaTeX 两遍编译成功，最终 PDF 9 页已逐页渲染和视觉检查。
- [x] 新运行的 Markdown、TeX 与 PDF 共用同一正式证据视图；旧敏感性解释和图路径未重新进入正式产物。

未使用 `scripts/repair_final_run.py`，未修改原始附件，未通过无限 timeout/retry、全零或硬编码结果绕过正常流程，也没有让补充图重新计算并替换主方案指标。

## 7. 复审新增根因：Markdown 与 TeX 证据视图不一致

`state.sensitivity_runs` 和 `state.figures` 为 append-only 历史。`latex_node.py` 已按参数选择最新敏感性、按 evidence role 选择正式图；但旧 `render_markdown()` 直接把完整历史传给 `paper.md.j2`。因此 PDF/TeX 是最新证据，Markdown 却可能附带旧轮次解释和重复图。旧 `paper.md` 中可见失效的 `238754.47`、`251132.99` 数值，而最终 PDF 不包含这些数值。

修复后，`latex_node` 把同一组 `formal_figures` 和最新 `formal_sens` 同时传给 Markdown 与 TeX；绿色物流安全求解器不再重复追加自由文本敏感性解释。finalizer 还会机械检查旧轮次解释文本和图路径是否重新进入 `paper.md`/`paper.tex`。对应回归测试位于 `tests/nodes/test_latex_node.py` 和 `tests/nodes/test_finalizer.py`。

v4 正常运行已验证该修复：正式 `paper.md` 与 `paper.tex` 均包含当前中心值 144586.99，均不含旧值 238754.47、251132.99；completion 摘要对最终重新排版后的四个文件重新计算并验证哈希。因此本复审项已闭环，不再保留“等待新运行”的未完成状态。
