# Beacon 文档索引与当前状态

最后核对：2026-07-17。

## 当前结论

`runs/green-logistics-rootfix-v4-20260717` 仍是本轮完整公开 CLI 闭环的基准运行；入口为
`math-agent start`，调用链经过 supervisor、checkpoint recover、全部 LangGraph 节点、runner、
结果校验、writer、LaTeX 和 finalizer。本次又针对 `runs/wodecesium` 暴露出的“总计 10 页、正文
只有 8 页”问题补齐了论文篇幅质量门禁和实质内容生成。未使用 `scripts/repair_final_run.py`，未修改
原始附件，也未放宽 120 秒脚本期限、2 GB 内存限制或既有数值质量门槛。

现行默认门禁要求 PDF 在附录前至少有 20 个非空正文页，且正文至少有 15000 个非空白字符；两个
阈值可分别通过 `MATH_AGENT_MIN_PAPER_BODY_PAGES` 与 `MATH_AGENT_MIN_PAPER_BODY_CHARS` 配置。
门禁不通过时 finalizer 必须把运行标记为 degraded，并给出实际页数或字符数，不能把附录代码页、
空白页或重复段落当成正文篇幅。

最终证据如下：

- `completion.json.status=completed`，issues/warnings 均为空；checkpoint 无下一节点。
- supervisor PID 22708、worker PID 3004 及其 transport 后代均已退出；公开 `status` 返回 completed。
- LLM trace 共 67 个逻辑调用、83 个物理 attempt；真实出现 3 次 120 秒 timeout、3 次 server error，
  其中两次 long 请求约 234 秒后切换上游成功。run-scoped 模型冷却状态可跨恢复 worker 继承。
- 主方案真实读取订单、距离矩阵、时间窗和坐标四个附件，RESULT 为成本 144586.99 元、车辆
  159 辆、服务率 1.0000、碳排放 14634.14 kg。
- 三个有效基线、三组非空敏感性；模型—代码一致性 8、论文评审 8、综合评分 7.1。
- 完整 CLI 基准运行的两遍 XeLaTeX 均 exit 0，历史产物为 9 页；该产物证明调用链闭环，不再代表
  现行论文篇幅验收标准。
- `runs/wodecesium-lengthfix-v3-20260717` 是从 `wodecesium` 已验证 checkpoint 状态进行的独立、
  确定性论文再生成验证，不是一次新的全流程求解。其 PDF 共 23 页，其中正文 20 页、附录 3 页，
  正文 21228 个非空白字符；23 页均已渲染检查，正文连续且均有实质内容，附录从第 21 页开始。
- Python 回归为 577 passed / 4 skipped，Web UI 为 13/13。

系统并不承诺外部上游永远不出现 502、429、断连或超时；“已解决”的含义是这些故障现在受
单次/总硬期限、错误分类、worker 回收、跨 worker 熔断、同节点恢复上限和 checkpoint 定点恢复
约束，不再无限挂起、无限重试或污染正式结果。

## 为什么旧 PDF 只有 8--10 页，当前如何保证 20 页正文

`runs/wodecesium/paper.pdf` 共 10 页，附录从第 9 页开始，实际正文只有 8 页、6472 个非空白字符；
这不是 XeLaTeX 截断。根因有两层：绿色物流安全分支绕过 writer LLM，直接返回一组偏短的固定章节；
finalizer 又只检查编译、评分和证据质量，没有检查附录前的正文页数与内容量。通用 writer 还会把模型
与敏感性章节替换成绿色物流固定文本，导致提示词中的篇幅要求既不可执行，也可能污染其他题目。

现行 writer 为特殊安全分支补齐问题分析、数据预处理、假设依据、数学模型、分问题求解、基线对比、
复杂度、敏感性、误差边界和部署建议，并只引用已验证 RESULT、BREAKDOWN 与敏感性数组。通用题目的
七个章节组全部交给当前题目的 writer 生成；每组有非空白字符预算，首次过短会带着质量问题重写一次，
再次不足则保留 checkpoint 并失败，不能用其他题目的固定稿兜底。最终 PDF 还需通过 20 个非空正文页
和 15000 个正文字符的双门禁。

最新验证稿页数来自新增的论证链与实验解释，不是通过放大字号、空白页、复制段落或延长代码附录获得。
它仍保留原运行的综合评分 6.80 和分析深度 5 警告，因此 completion 状态为 degraded；这与篇幅门禁已经
通过并不矛盾，也不能被篇幅修复掩盖。

## 现行文档

- [`beacon-resilient-execution.md`](beacon-resilient-execution.md)：后台监督、恢复、硬期限和最终收口。
- [`beacon-full-pipeline-root-cause-20260717.md`](beacon-full-pipeline-root-cause-20260717.md)：完整调用链、
  分项根因、测试矩阵和 v4 真题证据。
- [`llm-timeout-retry-redesign.md`](llm-timeout-retry-redesign.md)：LLM transport、取消和预算机制；
  现行参数以 `.env.example` 与代码为准。
- [`problem-blueprint-implementation-plan.md`](problem-blueprint-implementation-plan.md)：ProblemBlueprint
  落地的历史实施记录。
- [`writer-quality-recovery-development.md`](writer-quality-recovery-development.md)：writer 恢复与质量增强
  的设计记录，包含尚未全部落地的建议。

## 历史方案

`docs/superpowers/plans/` 和 `docs/superpowers/specs/` 保存阶段性计划与设计，不是现行事实源。
旧文件名、旧超时变量和“≥30 页/≥10 图”等声明只按历史上下文阅读。

## 验证入口

```powershell
uv run pytest -q
npm.cmd test -- --run
uv run math-agent status --out runs/green-logistics-rootfix-v4-20260717 --thread green-logistics-rootfix-v4
```

真实运行只有在 completion 摘要校验有效、正式证据门禁通过、LaTeX 两遍编译成功且 PDF 逐页视觉
检查完成后才可验收；“流程走到结尾”本身不算成功。
