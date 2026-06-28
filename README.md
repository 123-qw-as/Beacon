# math-agent (MVP)

LangGraph + LiteLLM 多智能体数学建模助手。本 MVP 实现：
**Analyst → Modeler(basic→improved→final) ⇄ ModelCritic → Coder(沙箱) → Writer → paper.md**

## 安装

```bash
pip install -e ".[dev]"
cp .env.example .env  # 填入 LLM API key
```

## 跑示例

```bash
math-agent run --problem tests/fixtures/sample_problem.json --out runs/demo
cat runs/demo/paper.md
```

## 跑测试

```bash
pytest -q
```

## 下一步（Plan B / Plan C）

- 强制 Sensitivity 节点 + Figure 流水线
- PaperCritic + Evaluation Module
- LaTeX 生成与编译
- HITL + Checkpointer
- 错误处理与重试统一化
- RAG（历年国一论文）
