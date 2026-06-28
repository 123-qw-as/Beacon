# math-agent

LangGraph + LiteLLM 多智能体数学建模助手。

**当前流水线（Plan B）：**
Analyst → Modeler(basic→improved→final) ⇄ ModelCritic → Coder → **Sensitivity → FigurePipeline →
Writer → PaperCritic → Evaluation → HumanReview → LaTeX**（带 Markdown 兜底）

## 安装

```bash
pip install -e ".[dev]"
cp .env.example .env  # 填入 LLM API key（或本地 OAI 兼容端点）
```

## 跑示例（带 HITL）

```bash
math-agent run --problem tests/fixtures/sample_problem.json --out runs/demo
# 流水线在 human_review 前停下；查看 runs/demo/checkpoints.sqlite 与中间产物
math-agent resume --out runs/demo --approve --notes "ok"
# 或一次跑到底：
math-agent run --problem ... --out runs/demo2 --no-interrupt
```

## 测试

```bash
pytest -q
```

## 已完成（Plan B）

- 强制 **Sensitivity** 节点：plan→sweep→interpret 三段式
- **Figure pipeline**：扫描 PNG → 多模态 FigureCritic 评分 → FigureAnalyst 写图说
- **PaperCritic** + 独立 **Evaluation Module**（确定性 overall）
- **Human-in-the-loop**：`interrupt_before=["human_review"]` + `SqliteSaver`
- **LaTeX** 渲染 + `xelatex` 编译；失败回退 Markdown

## 下一步（Plan C）

- 统一的 LLM/沙箱/编译错误重试与限流
- 历年国一题回归基准
- RAG（历年论文 / 经典模型库）
- LangSmith / OTel 链路追踪
