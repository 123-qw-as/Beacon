# LLM Timeout、取消与重试机制重构方案

> 历史设计记录。主要机制已经落地；现行参数以 `.env.example`、`src/math_agent/llm.py` 和 [`beacon-resilient-execution.md`](beacon-resilient-execution.md) 为准。

## 1. 背景与结论

当前 LLM 调用链通过 `threading.Thread` 执行 `litellm.completion()`，主线程使用
`Thread.join(LLM_TIMEOUT)` 实现硬超时。该实现能够让调用方在指定时间后返回，
但不能终止底层线程或 HTTP 请求。

诊断已稳定复现以下行为：

- 调用传入 `timeout=0.05`，模块级 `LLM_TIMEOUT=0.30` 时，实际约 0.30 秒才返回，
  说明调用级 timeout 没有控制硬超时。
- 连续 5 次超时后存在 5 个仍然存活的请求线程。
- 当前 `.env` 配置 `MATH_AGENT_LLM_TIMEOUT=600`，默认传输尝试次数为 5，单个逻辑
  调用的理论下界约为 `5 × 600 + 15 = 3015` 秒，即约 50 分钟，并可能同时留下
  5 个未完成请求。
- 历史运行记录显示 writer 的部分章节超过 120 秒，说明原超时值对长文本偏短；
  但把全局超时直接提高到 600 秒只会放大最坏等待，不会解决取消和重试问题。
- timeout 发生在成功追踪之前，`trace.json` 可能显示 `llm_calls=0`，但实际上已经向
  router 发出了一个或多个请求。

因此，本次重构的目标不是单纯调整秒数，而是重新定义完整的调用预算、取消语义、
重试边界和可观测性。

## 2. 目标与非目标

### 2.1 目标

1. LLM 调用超过单次尝试时限后，客户端请求必须收到取消信号，不能因应用层重试持续
   累积后台请求。
2. 一个逻辑调用必须有统一总 deadline，传输重试、退避等待和 JSON 修复都消耗同一预算。
3. timeout、连接失败、限流、5xx、鉴权失败和结构化输出失败必须具有不同重试策略。
4. 所有节点继续通过一个小接口调用 LLM，不把传输细节扩散到各节点。
5. 成功、失败、超时和重试均可追踪，能够回答“哪个节点、哪个模型、哪次尝试、耗时多久”。
6. 保留现有 checkpoint/recover 行为，失败后可以从当前 LangGraph 节点恢复。
7. 配置升级兼容现有 `.env`，并明确配置只对新启动的运行生效。

### 2.2 非目标

- 不在本次重构中更换 LiteLLM 或 router。
- 不通过无限增大 timeout 掩盖慢模型问题。
- 不自动并行发送相同 prompt 进行竞速。
- 不承诺远端 provider 一定停止计费或立即停止生成。Beacon 能保证的是：deadline 到达后
  本地 LiteLLM 工作进程和连接被终止、调用方及时恢复、不会残留本地请求执行体。远端是否
  在 TCP 断开后停止计算取决于 router/provider，因此 timeout 默认不自动重试。
- 不改变结构化 JSON/LaTeX 修复算法本身，只改变其预算和重试编排方式。

## 3. 术语与精确定义

重构后避免继续使用含义模糊的“timeout”和“retry”。统一采用以下术语：

| 术语 | 定义 |
|---|---|
| 逻辑调用（call） | 节点调用一次 `complete()`；可能包含多个物理尝试和 JSON 修复轮次 |
| 物理尝试（attempt） | 向 provider/router 实际发送一次请求 |
| 单次时限（attempt timeout） | 一次物理尝试从开始发送到完整读取响应的最长时间 |
| 总 deadline | 整个逻辑调用的绝对截止时间，覆盖请求、重试、退避和 JSON 修复 |
| 传输重试 | 因 429、连接建立失败或可恢复 5xx 再次发送请求 |
| JSON 修复 | 响应已成功返回，但无法通过目标 schema 校验，携带错误反馈重新请求 |
| 终止宽限（kill grace） | worker 执行 `terminate()` 后等待正常退出的短时间窗口，超出后执行 `kill()` |
| 调用画像（profile） | 一组已命名的时限和预算策略，如 `standard`、`long`、`vision` |

必须满足以下不变量：

1. 任意物理尝试的 timeout 不得超过逻辑调用剩余预算。
2. deadline 到达后不得再创建新请求。
3. 读取超时默认不自动重试，因为服务端可能已经接收请求并继续生成。
4. 仅当上一次请求已完成、失败或确认收到取消后，才允许开始下一次尝试。
5. 每个物理尝试必须产生一条追踪记录，无论成功还是失败。

## 4. 目标模块设计

### 4.1 外部 seam：保持一个深模块

所有节点继续只依赖 `math_agent.llm.complete()`：

```python
def complete(
    prompt: str,
    *,
    schema: type[T] | None = None,
    model: str | None = None,
    system: str | None = None,
    images: list[str] | None = None,
    profile: Literal["standard", "long", "vision"] = "standard",
    temperature: float = 0.3,
    max_validation_repairs: int | None = None,
    **model_kwargs: Any,
) -> str | T:
    ...
```

节点只选择调用画像，不直接传秒数、重试次数、退避参数或 LiteLLM client。这样复杂性集中
在 LLM 模块内；修改取消和重试实现时，20 多个节点调用点不需要同步修改。

不建议把 `hard_timeout`、`retry_attempts`、`retry_base_delay` 继续作为公开接口。测试所需的
策略注入放在模块内部 seam，不暴露给业务节点。

### 4.2 内部 seam：传输 adapter

LLM 模块内部定义一个仅供实现和测试使用的同步传输 port：

```python
class CompletionTransport(Protocol):
    def send(
        self,
        request: CompletionRequest,
        *,
        timeout_s: float,
    ) -> CompletionResponse:
        ...
```

提供两个 adapter：

- `LiteLLMWorkerTransport`：生产 adapter，通过 IPC 调用常驻 LiteLLM 工作进程。
- `FakeCompletionTransport`：测试 adapter，可模拟成功、挂起、取消、429、5xx 和非法 JSON。

这是一个真实 seam：生产和测试各有一个 adapter。外部 `complete()` 接口不暴露该 port；
测试通过模块内部 fixture 替换 adapter。

### 4.3 常驻工作进程与硬终止

当前 LangGraph 节点均为同步函数，因此外部接口保持同步。生产 adapter 维护一个惰性启动的
常驻工作进程：

```python
class LiteLLMWorkerTransport:
    def send(self, request, *, timeout_s):
        worker = self._ensure_worker()
        worker.connection.send(request.to_wire())
        if not worker.connection.poll(timeout_s):
            self._terminate_worker(worker)
            raise LLMTimeoutError(...)
        return CompletionResponse.from_wire(worker.connection.recv())
```

工作进程通过 `multiprocessing` 的 `spawn` 上下文启动，使用 `Pipe` 传输普通 dict，不把
Pydantic 类型、tracer 或调用方对象直接传入子进程。工作进程内部：

1. 进程启动时导入并配置 LiteLLM，摊销其导入和 client 初始化成本。
2. 循环接收请求，调用同步 `litellm.completion()`。
3. 将响应内容、usage 和必要元数据序列化为普通 dict 返回父进程。
4. 将异常序列化为安全的错误描述，包括异常类名、状态码、`Retry-After` 和简短消息。
5. API key、api base 和 callback 配置继续从继承的环境变量读取，不写入 IPC 消息或日志。

父进程使用 `connection.poll(timeout_s)` 实现真正的调用方 deadline。timeout 时依次执行：

1. 关闭 IPC connection，促使 worker 退出正常通信路径。
2. `Process.terminate()` 并等待短暂终止宽限。
3. 若仍存活，执行 `Process.kill()` 并 `join()` 回收句柄。
4. 清空当前 worker 引用；下一次调用惰性启动新 worker。
5. 记录 timeout 和 worker 重启事件，但不自动重放刚才的请求。

正常成功请求复用同一个 worker，因此不会为每次 LLM 调用支付进程启动成本。当前流水线以
顺序调用为主，一个 worker 足够；若未来引入并行节点，再增加一个有明确上限的 worker pool，
不能在本次提前加入无界并发。

### 4.4 为什么不把异步取消作为最终保障

`litellm.acompletion()` 配合 `asyncio.timeout()` 对使用异步 HTTP 的 provider 通常有效，而且开销
更低；但如果某个 provider adapter 在协程内部调用阻塞 SDK、吞掉取消或卡在 C 扩展，事件循环
仍无法保证按 deadline 恢复。当前系统已经明确需要一个不信任第三方 timeout 实现的硬保障，
因此常驻工作进程是默认 production adapter，而不是兼容兜底。

异步 adapter 可以保留为未来经过 provider 白名单验证后的性能优化，但不能成为默认安全路径。

### 4.5 能保证和不能保证的取消范围

工作进程方案保证：

- 父进程在 deadline 后恢复控制；
- 本地 LiteLLM 执行体被终止；
- 本地 HTTP socket 随进程退出而关闭；
- 不存在每次 timeout 新增一个不可回收 daemon thread 的问题。

工作进程方案不能保证远端 provider 已停止计算。HTTP 连接断开通常会传播取消，但具体行为由
router/provider 决定。因此 timeout 后默认不自动重试，避免远端旧请求仍在执行时产生第二份请求。

## 5. 调用预算模型

### 5.1 建议默认画像

| 画像 | 单次时限 | 总 deadline | 适用场景 |
|---|---:|---:|---|
| `standard` | 180 秒 | 300 秒 | analyst、critic、coder、sensitivity、evaluation |
| `long` | 300 秒 | 420 秒 | writer outline、writer section、较长 modeler 输出 |
| `vision` | 180 秒 | 300 秒 | figure critic、figure analysis |

这些默认值基于现有运行记录：多数成功调用在几十秒内完成，历史 writer 章节存在超过
120 秒的情况，因此 writer 使用 300 秒单次时限；不再让所有节点共同使用 600 秒。

### 5.2 剩余预算计算

逻辑调用开始时记录单调时钟 deadline：

```python
deadline = monotonic() + policy.total_timeout_s
remaining = deadline - monotonic()
attempt_timeout = min(policy.attempt_timeout_s, remaining)
```

进入请求、退避或 JSON 修复前都重新计算 `remaining`。如果剩余预算不足以完成下一操作，
直接抛出 `LLMBudgetExceededError`，不得再发送请求。

不使用 wall clock 计算耗时，避免系统校时导致 deadline 前后跳变。

### 5.3 输出长度与 timeout

timeout 只控制异常等待，不能替代输出预算。各 prompt 应继续设置合理的 schema 和输出范围；
必要时给 writer/modeler 配置显式 `max_tokens`。若某画像频繁接近 deadline，应先检查：

- prompt 是否携带了不必要的完整 state；
- schema 是否要求一次生成过多字段；
- reasoning 模型是否把大量 token 消耗在思考过程；
- router 是否存在排队或并发限制。

## 6. 重试策略

### 6.1 错误分类

建议扩展错误类型：

```text
LLMError
├── LLMAuthenticationError       不重试
├── LLMInvalidRequestError       不重试
├── LLMProtocolError             不重试
├── LLMRateLimitError            在 deadline 内退避重试
├── LLMConnectionError           建连失败可重试
├── LLMServerError               部分 5xx 可重试
├── LLMTimeoutError              默认不自动重试
├── LLMBudgetExceededError       不重试
└── LLMValidationError           进入 JSON 修复流程
```

不要继续把 timeout、连接失败、5xx、协议错误都合并成 `LLMTransportError` 后使用同一重试策略。
可以保留 `LLMTransportError` 作为兼容父类。

### 6.2 默认策略矩阵

| 错误 | 默认尝试 | 退避 | 说明 |
|---|---:|---|---|
| 读取/生成 timeout | 1 | 无 | 服务端可能仍在生成，自动重试可能重复计费 |
| 连接建立失败 | 最多 2 | 1 秒 + jitter | 通常可确定请求未被服务端处理 |
| 429 | 最多 3 | 优先 `Retry-After`，否则 2/4 秒 + jitter | 必须受总 deadline 限制 |
| 502/503/504 | 最多 2 | 1/2 秒 + jitter | 短暂 router 故障 |
| 500/协议损坏 | 1 | 无 | 默认不盲目重放 |
| 400/401/403/404 | 1 | 无 | 配置或请求错误 |
| schema 校验失败 | 初次 + 2 次修复 | 无网络退避 | 共享同一个总 deadline |

若业务确实需要 timeout 后重试，应作为显式画像策略，并且只有在取消清理成功后才能执行，
默认最多再试一次。生产默认仍应关闭。

### 6.3 LiteLLM 内部重试

Beacon 必须作为唯一重试编排者，避免 LiteLLM、router 和 Beacon 三层重试相乘：

- LiteLLM client 侧重试设置为 0。
- Beacon 根据错误分类执行上述策略。
- router 是否重试属于部署配置，必须在运行文档中明确；若 router 已重试，Beacon 对 5xx/429
  的尝试次数应进一步降低。

### 6.4 JSON 修复与传输重试分离

现有 `max_retries` 同时容易被理解为传输重试和解析重试。重构后改名为
`max_validation_repairs`，默认 2，并遵守：

- 每次响应成功返回后才进入 schema 校验。
- 修复请求仍可能发生传输错误，但不会重置总 deadline。
- timeout 不会被包装成“JSON 解析失败”。
- 最终错误包含修复轮次、最后一次校验错误和剩余预算信息。

## 7. 节点画像映射

第一版建议映射如下：

| 节点/调用 | 画像 |
|---|---|
| analyst | `standard` |
| blueprint_critic | `standard` |
| modeler 主模型 | `long` |
| modeler derivation / consistency | `standard` |
| model_critic | `standard` |
| coder / baseline coder | `standard` |
| sensitivity plan / code / interpret | `standard` |
| figure critic / figure analysis | `vision` |
| writer outline | `long` |
| writer section | `long` |
| paper_critic | `standard` |
| evaluation | `standard` |

画像映射可以集中在 `config.py` 的 `LLM_CALL_PROFILES` 与 `NODE_LLM_PROFILE` 中，节点仍需在
调用处显式选择画像，以便代码审查时能看到“这是长调用”，但不能传入任意秒数。

## 8. 配置与前端设计

### 8.1 新环境变量

```dotenv
# 普通 LLM 调用
MATH_AGENT_LLM_ATTEMPT_TIMEOUT=180
MATH_AGENT_LLM_TOTAL_TIMEOUT=300

# 长文本调用（writer / 大型 modeler）
MATH_AGENT_LLM_LONG_ATTEMPT_TIMEOUT=300
MATH_AGENT_LLM_LONG_TOTAL_TIMEOUT=420

# 工作进程终止宽限
MATH_AGENT_LLM_WORKER_KILL_GRACE=2

# 结构化输出修复轮次
MATH_AGENT_LLM_VALIDATION_REPAIRS=2
```

不建议继续把传输尝试次数暴露为一个全局环境变量；不同错误类型使用代码中的安全上限。

### 8.2 兼容策略

过渡一个版本：

1. 如果设置了新变量，完全使用新变量。
2. 如果只存在旧 `MATH_AGENT_LLM_TIMEOUT`：
   - 将它解释为 `standard.attempt_timeout_s`；
   - `standard.total_timeout_s = min(old × 1.5, old + 180)`；
   - `long.attempt_timeout_s = max(old, 300)`，但上限 600；
   - 启动时输出一次弃用提示。
3. 下一主版本删除旧变量兼容。

旧的 `MATH_AGENT_LLM_RETRY_ATTEMPTS`、`MATH_AGENT_LLM_RETRY_BASE_DELAY` 和
`MATH_AGENT_LLM_RETRY_MAX_DELAY` 标记为弃用。新重试矩阵不从它们推导。

### 8.3 前端

非技术用户界面只展示两个设置：

- “普通调用最长等待”：默认 180 秒。
- “长文本调用最长等待”：默认 300 秒。

总 deadline 由系统按安全规则推导，并在帮助文字中说明“包含必要的等待与重试”。终止宽限、
错误类型重试次数等内部安全参数不放到 UI。

保存配置后显示：“新设置将在下一次新运行或恢复运行时生效。”当前 `server.mjs` 已在 spawn
子进程时剔除缓存配置，让新子进程重新读取 `.env`，该机制继续保留。

连接测试继续使用独立的 30 秒 AbortController；它只判断端点可达，不复用正式生成画像。

## 9. 可观测性

### 9.1 trace.json 兼容扩展

保留现有字段，并新增：

```json
{
  "llm_calls": 12,
  "llm_attempts": 14,
  "llm_failures": 2,
  "llm_timeouts": 1,
  "llm_attempt_records": [
    {
      "call_id": "...",
      "node": "writer_section",
      "model": "openai/ocg/deepseek-v4-pro",
      "profile": "long",
      "attempt": 1,
      "status": "timeout",
      "latency_ms": 300015,
      "attempt_timeout_ms": 300000,
      "remaining_budget_ms": 119985,
      "error_kind": "timeout"
    }
  ]
}
```

定义：

- `llm_calls`：成功完成的逻辑调用数，保持现有含义。
- `llm_attempts`：实际发送到 provider/router 的物理请求数。
- `llm_failures`：失败的物理尝试数。
- `llm_timeouts`：timeout 的物理尝试数。

每个 attempt 在发送前先登记开始，结束时更新结果。进程异常退出时，未结束记录可标记为
`interrupted`。不记录 prompt、API key、完整响应或图片内容。

### 9.2 用户可见日志

日志采用简短中文格式：

```text
[LLM] writer_section / long / attempt 1：开始，单次上限 300s，总剩余 420s
[LLM] writer_section / long / attempt 1：超时 300.0s，请求已取消，不自动重试
```

429 或 5xx 重试时必须显示原因、等待时间和剩余预算。不能只输出“router 半挂？”，因为该
文案把未经验证的猜测当成结论。

## 10. 测试方案

### 10.1 首要回归测试

在修改实现前先加入以下失败测试：

1. `test_attempt_timeout_terminates_inflight_worker`
   - Fake worker 挂起并记录进程状态。
   - deadline 后返回 `LLMTimeoutError`。
   - 断言 worker 已退出并完成 `join()`，不存在后台请求执行体。

2. `test_timeout_is_not_retried_by_default`
   - 第一次请求挂起到 timeout。
   - 断言 transport 只调用一次。

3. `test_total_deadline_caps_retries_and_backoff`
   - 使用 fake clock 和连续 429。
   - 断言总耗时不突破 deadline，deadline 后不再发送请求。

4. `test_per_profile_attempt_timeout_is_applied`
   - 分别执行 `standard` 和 `long`。
   - 断言 adapter 收到对应时限，而非模块级固定常量。

5. `test_validation_repairs_share_total_budget`
   - 连续返回非法 JSON。
   - 断言修复轮次不会重新获得完整预算。

6. `test_failed_attempt_is_written_to_trace`
   - timeout 后断言 attempts、failures、timeouts 和 attempt record 均正确。

7. `test_worker_timeout_terminates_and_restarts_worker`
   - 第一个 worker 永久挂起，timeout 后断言进程已退出。
   - 下一次正常请求启动新 worker 并成功返回。

8. `test_tracer_stays_in_parent_process`
   - worker 只返回 usage 和状态，父进程 tracer 正确记录 attempt 和逻辑调用。

### 10.2 错误矩阵测试

参数化验证以下输入与尝试次数：

- timeout：1 次；
- connect error：最多 2 次；
- 429 + Retry-After：最多 3 次且不突破 deadline；
- 502/503/504：最多 2 次；
- 400/401/403/404：1 次；
- protocol error：1 次；
- validation error：初次 + 指定修复次数。

### 10.3 本地 HTTP 集成测试

启动可控的本地 OpenAI-compatible fake server，提供：

- `/hang`：接受请求后不返回；
- `/slow`：延迟后返回合法响应；
- `/rate-limit`：先返回 429 再成功；
- `/invalid-json`：返回协议损坏内容；
- `/disconnect`：响应中途断开。

重点断言 `/hang` 超时后 worker 进程退出、客户端连接被关闭，随后正常请求由新 worker 完成，
不受前一个请求影响。

### 10.4 真实 router 冒烟测试

真实调用不进入 CI，通过显式环境变量启用：

```powershell
$env:MATH_AGENT_RUN_LIVE_LLM_TESTS='1'
uv run --extra dev pytest tests/integration/test_llm_live.py -q
```

仅发送极短 prompt，验证普通成功、时限参数和 trace，不测试真实 300 秒 timeout。

### 10.5 完整回归命令

```powershell
uv run --extra dev pytest tests/test_llm.py tests/test_retry.py tests/test_tracing.py -q
uv run --extra dev pytest tests/integration/test_llm_transport.py -q
uv run --extra dev pytest -q
npm test
```

## 11. 分阶段实施计划

### 阶段 A：建立正确反馈闭环

1. 把本次一次性复现转为正式回归测试。
2. 新增 fake transport 和 fake clock fixture。
3. 先观察“超时后活动请求仍为 1”测试失败。

完成标准：有一条数秒内稳定变红的命令，精确捕获幽灵请求问题。

### 阶段 B：替换取消机制

1. 引入内部 `CompletionTransport` port。
2. 实现 `LiteLLMWorkerTransport` 和工作进程入口。
3. 使用 IPC poll、terminate、kill 和 join 实现可验证的硬终止。
4. 正常请求复用 worker，timeout 后下一次请求惰性重启 worker。
5. 删除 `_do_completion()` 中的 daemon thread 和 `Thread.join()`。
6. 禁用 LiteLLM client 内部重试。

完成标准：超时测试变绿，活动请求归零；现有正常 completion 测试通过。

### 阶段 C：预算与分类重试

1. 新增 `CompletionPolicy`、`CallBudget` 和调用画像。
2. 拆分错误类型和分类逻辑。
3. 将 retry 从通用 tenacity 装饰器迁入 LLM 深模块，按错误类型决策。
4. JSON 修复改为共享 deadline。
5. 移除 `_retry_attempts`、`_retry_base_delay` 等测试泄漏参数。

完成标准：错误矩阵和总 deadline 测试全部通过。

### 阶段 D：节点画像与可观测性

1. 给 writer、modeler 和 vision 调用映射画像。
2. 扩展 tracer，记录每个物理尝试。
3. 增加用户可见的开始、重试、取消和失败日志。
4. 确认 recover 从失败节点继续，不重复已 checkpoint 的 writer section。

完成标准：timeout 后 trace 不再显示 0 次请求，recover 行为保持不变。

### 阶段 E：配置迁移与文档

1. 增加新环境变量和旧变量兼容层。
2. 更新前端高级设置和提示文案。
3. 更新 `.env.example`、README 和运行手册。
4. 启动时输出一次旧配置弃用提示。

完成标准：旧 `.env` 能运行，新设置在下一次 run/resume 子进程中生效。

## 12. 文件级改动清单

| 文件 | 改动 |
|---|---|
| `src/math_agent/llm.py` | 移除 thread join；加入调用预算、worker transport、分类重试和画像选择 |
| `src/math_agent/llm_worker.py` | 新增常驻 LiteLLM 工作进程入口与 IPC 协议 |
| `src/math_agent/config.py` | 定义画像配置、新环境变量和旧变量兼容 |
| `src/math_agent/errors.py` | 细分 timeout、连接、server、protocol、budget 等错误 |
| `src/math_agent/retry.py` | LLM 通用装饰器退役或仅保留非 LLM 用途；避免双重编排 |
| `src/math_agent/tracing.py` | 增加 attempt 级成功/失败/超时追踪 |
| `src/math_agent/nodes/*.py` | 仅为少数调用显式选择 `long` / `vision` 画像 |
| `frontend/routes/config.mjs` | 保存、读取普通与长文本时限 |
| `frontend/app.js` | 更新高级设置字段和“下次运行生效”提示 |
| `.env.example` | 增加新变量，标记旧变量弃用 |
| `README.md` | 更新 timeout、retry、recover 和排障说明 |
| `tests/test_llm.py` | 用接口行为测试替换 Thread.join 实现测试 |
| `tests/test_retry.py` | 移除统一 LLM 重试假设，改测错误矩阵 |
| `tests/test_tracing.py` | 增加失败 attempt 与兼容加载测试 |
| `tests/integration/test_llm_transport.py` | 新增本地挂起、取消和协议异常集成测试 |

## 13. 验收标准

必须全部满足：

- [ ] 任意一次超时后，对应 worker 在终止宽限内退出；必要时 kill 后完成回收。
- [ ] timeout 默认不会自动发送第二个相同请求。
- [ ] 任何重试和退避都不能突破逻辑调用总 deadline。
- [ ] `standard`、`long`、`vision` 的单次时限能够独立生效。
- [ ] 429、连接失败、5xx、鉴权失败和 timeout 使用不同策略。
- [ ] JSON 修复轮次与传输重试分开计数，并共享总 deadline。
- [ ] LiteLLM client 侧重试关闭，避免多层重试相乘。
- [ ] timeout 和失败请求出现在 trace 中，包含节点、模型、画像、attempt 和耗时。
- [ ] writer section 超时后可以 resume，已完成 section 不重复调用。
- [ ] 旧 `.env` 在兼容期可以运行，并产生清晰弃用提示。
- [ ] 全量 Python 与前端测试通过。
- [ ] 不再存在 `Thread.join(LLM_TIMEOUT)` 和每次调用创建 daemon thread 的实现。

## 14. 发布、监控与回滚

### 14.1 发布方式

建议分两个版本开关发布：

1. 首先在开发环境默认启用新 transport，保留
   `MATH_AGENT_LEGACY_LLM_TIMEOUT=1` 临时回退开关。
2. 使用 3 个代表性题目各跑一次完整流水线，观察：
   - timeout 数量；
   - attempt/call 比例；
   - P50/P95 延迟；
   - writer section 成功率；
   - router 并发与错误率；
   - recover 后重复调用数量。
3. 验证稳定后删除 legacy 开关和 Thread.join 实现，不长期维护两套路径。

### 14.2 预警阈值

- `llm_attempts / llm_calls > 1.2`：重试过多，需要检查 router 或错误分类。
- 单模型 timeout 比例 > 5%：检查画像、模型路由或上游容量。
- `worker_kill_failed > 0`：本地进程终止异常，必须作为高优先级故障处理。
- writer `long` P95 > 240 秒：先缩 prompt/输出，再考虑调整画像。

### 14.3 回滚

若新 worker adapter 与某 provider 不兼容，可临时启用 legacy 开关，但必须同时：

- 将传输尝试次数限制为 1；
- 禁止 timeout 后自动重试；
- 在日志中明确提示 legacy 模式可能残留底层请求；
- 记录对应 provider，修复 worker IPC 或序列化兼容问题后再关闭 legacy 模式。

回滚不得恢复“600 秒 × 5 次自动重试”的组合。

## 15. 最终决策摘要

1. 删除 daemon thread + `Thread.join()` 的伪硬超时。
2. 默认采用常驻 LiteLLM 工作进程；deadline 到达时终止并回收 worker，下一次调用惰性重启。
3. 通过 `standard / long / vision` 画像表达节点差异，不让秒数散落在节点中。
4. 使用单一总 deadline 管理传输、退避和 JSON 修复。
5. timeout 默认不自动重试；连接失败、429 和部分 5xx 按类别有限重试。
6. Beacon 是唯一 client 侧重试编排者，关闭 LiteLLM 内部重试。
7. 每个物理尝试都进入 trace，成功和失败同样可见。
8. timeout 只能保证本地 worker 和连接被终止，不能保证远端停止计费；因此默认不自动重试。

## 附录 A：落地风险实测核实

以下结论基于对 `src/math_agent/` 现有代码的逐项核实，用于校准分阶段实施计划中
的假设与完成标准。核实环境：Windows 10.0.22631，Python 3.14，本仓库当前 HEAD。

### A.1 worker 冷启动成本（阶段 B 前置基准）

实测 `spawn` 子进程并 import litellm 的耗时：

| 场景 | 耗时 |
|---|---:|
| 父进程首次 import litellm | 5.60s |
| spawn 空进程 | 0.13s |
| spawn + 子进程内 import litellm | 5.42s |

**含义**：正常路径下 worker 常驻复用，5.4s 冷启动只在首次和 timeout 后重启时支付，
可接受。但若一次运行内连续发生 timeout（router 抖动），每次重启都带来 5.4s 卡顿。

**对方案的修正**：阶段 B 的完成标准应补充——worker 启动后先发一个空 Pipe 握手信号，
父进程等到握手成功才把该 worker 视为就绪并开始 `poll(timeout_s)` 计时，避免把 import
时间算进首次调用的 timeout 窗口。worker 入口的 litellm import 放模块顶层，让 spawn
路径更可预测。

### A.2 graph 并行度（排除风险）

`graph.py` 全部为同步节点 + 顺序 `add_edge`，无 `async def` 节点、无 `asyncio.gather`、
无 `Send()` fan-out、无 `ThreadPoolExecutor`。LangGraph 对同步节点默认串行执行。
`sensitivity` 节点内连续三次 `complete()`（sensitivity.py:80-147）也是串行。

**结论**：单 worker 方案完全可行，不会退化并行、不会争抢 Pipe。第 4.3 节“当前流水线
以顺序调用为主，一个 worker 足够”的判断成立，当前不需要 worker pool。

### A.3 测试参数迁移面（轻量）

`_retry_attempts` / `_retry_base_delay` 在测试中的使用面：

| 文件 | 处数 | 用途 |
|---|---:|---|
| `tests/test_llm.py` | 5 | 用 `_retry_attempts=N, _retry_base_delay=0` 注入重试行为 |
| `tests/test_retry.py` | 2 | 测 env override |

**含义**：阶段 C 移除这两个公开参数的迁移量小（7 处）。但这些测试目前靠公开参数控制
重试，移除后需改用阶段 A 新建的 `FakeCompletionTransport` 注入。顺序自洽：先建
FakeTransport（阶段 A），再移除公开参数（阶段 C）。

### A.4 checkpoint 粒度（关键修正）

这是最重要的核实结果。当前架构：

- LangGraph `SqliteSaver` 在**每个节点函数返回后**写 checkpoint，不在节点内部。
- `graph.py:103` 注释“每次 section 完成 = 一个 checkpoint”，`writer.py:80` 同述。
- writer_section_node 内部的 `complete()` timeout → 节点异常退出 → 不写 checkpoint →
  **该 section 从队列头重新执行**（队列弹出发生在节点返回时，异常时未弹）。
- 但**已完成的 section**（之前返回并 checkpoint）不会重复，字段已写进 `paper`。

**对验收标准的修正**：第 13 节“writer section 超时后可以 resume，已完成 section 不重复
调用”成立，但需明确：**当前崩溃的 section 会整段重跑，包括重新调用 LLM**。这与第 6.4
节“timeout 不包装成 JSON 解析失败”配合后是安全的（重跑只产生一次新请求，不叠加）。

**方案未覆盖的副作用**：recover 重跑当前 section 时会再次 `complete()`。若 timeout 由
prompt 过长 / 输出过大（第 5.3 节）引起，recover 会再次 timeout，陷入
“recover → timeout → recover”循环。当前 `cli.py` 的 `recover` 命令（cli.py:366-398）
无失败计数或降级。

**建议补充（阶段 D）**：recover 时检测同一节点连续失败次数，超过阈值（如 2 次）后停止
recover 并提示用户手动干预或切换画像，而不是无限重跑。将此条加入第 13 节验收标准。

### A.5 IPC 大消息传输（实现细节）

`figure_pipeline.py` 的 `complete()` 传 `images=[data_url]`，单张图可达数 MB。Windows
Pipe 传大消息会阻塞，但只要保证 worker 侧先完整 `recv` 请求再调用 `litellm.completion()`，
父进程 `poll(timeout_s)` 从请求发出后计时，就不会把传输时间算进 timeout。

**对方案的补充**：第 4.3 节伪代码的顺序（`send` → `poll` → `recv`）需保证 worker 已
`recv` 完请求才开始计时，或将 poll 改为“等 worker 就绪信号 + 等结果”两阶段。实现细节，
不阻塞方案可行性。

### A.6 核实结论汇总

| 风险 | 状态 | 对方案的影响 |
|---|---|---|
| worker 冷启动 5.4s | 确认 | 阶段 B 加预热握手；写进完成标准 |
| graph 并行度 | 排除 | 单 worker 安全，无需 pool |
| 测试参数迁移 | 轻量 | 7 处，随阶段 A 的 FakeTransport 一起改 |
| checkpoint 粒度 | 关键修正 | 当前 section 整段重跑；阶段 D 加 recover 失败计数防死循环 |
| IPC 大消息 | 实现细节 | worker 先 recv 再计时，不阻塞 |

**总体判断**：方案可行，可按阶段 A → E 实施。唯一需补充的设计缺口是 recover 死循环
防护（阶段 D），其余为实现层面的注意项。
