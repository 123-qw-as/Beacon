# Plan C Runbook

## 1. RAG 索引

```bash
# 把历年国一论文 PDF/MD 放进 corpus/
math-agent ingest --src corpus --db runs/rag.sqlite \
  --embedding-model text-embedding-3-small --dim 1536

# 运行时开启 RAG 注入
export MATH_AGENT_RAG_ENABLED=1
export MATH_AGENT_RAG_DB=runs/rag.sqlite
math-agent run --problem tests/fixtures/sample_problem.json --out runs/demo
```

可调环境变量：
- `MATH_AGENT_RAG_ENABLED` (0/1)
- `MATH_AGENT_RAG_DB` (sqlite 文件路径)
- `MATH_AGENT_RAG_EMBED` (默认 `text-embedding-3-small`)
- `MATH_AGENT_RAG_DIM` (默认 1536)
- `MATH_AGENT_RAG_TOPK` (默认 4)

prompt 注入字符上限（防爆 8k token）：analyst 1500、modeler 1500、writer 800（见 `config.py`）。

## 2. 回归基准

```bash
# mock 模式：不调 LLM，校验流水线结构 + 关键词覆盖（CI / 本地快速检查）
pytest tests/bench/ -v
# 期望：每条 PASS

# live 模式（贵；建议手动逐题跑）—— 真调 LLM
math-agent bench --out runs/bench

# 或针对单题 debug
math-agent run --problem src/math_agent/bench/problems/2022_A.json --out runs/live/2022_A
```

`expectations.json` 字段：
- `min_overall`：`EvaluationReport.overall` 的下限
- `must_contain_keywords`：拼接 abstract+model_section+solution+conclusion 必须含的关键词
- `must_have_sensitivity` / `must_have_figures`：流程产物完整性

## 3. 链路追踪

```bash
math-agent report --out runs/demo
# 显示每模型调用数 / token 数 / 每节点耗时
```

`trace.json` 字段：
- `llm_calls`：总 LLM 调用次数
- `tokens.prompt` / `tokens.completion`：总 token 数
- `per_model.<model>.{calls,prompt_tokens,completion_tokens,latency_ms}`
- `nodes[].{name,duration_ms}`

可选远端追踪（自动启用，无需改代码）：
- 设置 `LANGSMITH_API_KEY` → litellm 写入 LangSmith
- 设置 `OTEL_EXPORTER_OTLP_ENDPOINT` → litellm 写入 OTel

## 4. 错误处理速查

| 现象 | error_kind / 类型 | 是否自动重试 | 排查 |
|---|---|---|---|
| LLM 429 | `LLMRateLimitError` | 是（tenacity 指数退避） | 默认 `MAX_LLM_RETRIES + 1` 次，可改 config |
| LLM JSON 解析失败 | `LLMValidationError` | 否（`complete` 内部喂回错误重试） | 看 schema 是否过严 |
| LLM 超时/连接错误 | `LLMTransportError` | 是 | 检查 `LITELLM_LOG=DEBUG` 输出 |
| runner 超时 | `RunResult(error_kind="timeout")` | Coder/Sensitivity 内部允许一次重试 | 加大 `timeout` 或简化代码 |
| runner 运行错误 | `RunResult(error_kind="runtime")` | 同上 | 看 `stderr` |
| xelatex 缺失 | `LatexResult(error_kind="missing_binary")` | 否 | 安装 TeX Live 或回退 Markdown |
| xelatex 编译失败 | `LatexResult(error_kind="compile")` | 否 | 看 `paper.log`；常见为字体缺失 |
| xelatex 超时 | `LatexResult(error_kind="timeout")` | 否 | 加大 `compile_latex(timeout=...)` |

## 5. 关键模块速览

- `src/math_agent/errors.py`：所有自定义错误（MathAgentError → LLM/Runner/Latex 三大族）
- `src/math_agent/retry.py`：`llm_retry` / `runner_retry` tenacity 装饰器
- `src/math_agent/tracing.py`：`Tracer` + `set_current`/`get_current`/`reset_current`
- `src/math_agent/rag/{chunking,embeddings,store,retrieve,ingest}.py`：RAG 五件套
- `src/math_agent/bench/runner.py`：纯 live runner；mock 在 `tests/bench/conftest.py`
