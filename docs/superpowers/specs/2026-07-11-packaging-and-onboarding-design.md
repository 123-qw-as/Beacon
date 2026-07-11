# Beacon 打包与使用门槛降低设计

> 日期：2026-07-11
> 状态：已确认，待实现

## 1. 背景与目标

Beacon 是一个数学建模自动化系统，当前用户使用需要跨过以下门槛：

1. **安装依赖**：Python ≥3.11、Node.js ≥18、uv 包管理器
2. **配置 API**：手动复制 `.env.example` → `.env`，手填 `OPENAI_API_BASE`、`OPENAI_API_KEY`、模型名
3. **启动项目**：运行 `scripts/start.bat` 或 `npm start`

目标：在**不改变现有架构**（Python 后端 + Node 服务器 + 原生前端）的前提下，通过以下三部分降低使用门槛：

- **引导向导**：首次打开自动进入5步引导，可视化完成环境检测、API 配置、模型选择
- **设置页面**：随时可改 API 密钥、模型、RAG 开关等，不用手动编辑 `.env`
- **启动脚本增强**：`start.bat` 自检 + 引导启动

### 设计约束

- **平台**：仅 Windows 做自动安装（winget），macOS/Linux 保留现有 `start.sh` 不做自动安装
- **架构**：Node 服务器全权负责环境检测/安装/配置/测试（避免"需要 Python 才能检测 Python"的死循环）
- **存储**：继续用 `.env` 文件，不引入加密存储或数据库
- **Python 后端**：完全不改，仍由 CLI 子进程调用

## 2. 架构设计

### 2.1 整体架构

```
浏览器 (引导向导 / 设置页面 / 工作台)
    ↓ HTTP API
Node 服务器 (server.mjs — 路由分发)
    ├── routes/env.mjs       — 环境检测 + winget 安装
    ├── routes/config.mjs    — .env 读写 + LLM 连接测试
    └── routes/onboarding.mjs — 引导向导状态管理
    ↓ child_process.spawn (不变)
Python 后端 (math-agent CLI — 流水线执行)
```

**核心原则**：`server.mjs` 精简为路由分发器，具体逻辑拆到 `routes/` 模块。Python 后端完全不动。

### 2.2 新增文件结构

```
build-agent/
├── frontend/
│   ├── server.mjs              # 主服务器（精简：路由分发 + 现有 run/artifact API 保留）
│   ├── routes/                 # 新增目录
│   │   ├── env.mjs             # 新增：环境检测 + winget 安装
│   │   ├── config.mjs          # 新增：.env 读写 + LLM 连接测试
│   │   └── onboarding.mjs      # 新增：引导状态判断（是否首次/是否已完成）
│   ├── index.html              # 修改：新增引导向导 + 设置页入口
│   ├── app.js                  # 修改：新增引导/设置交互逻辑
│   ├── styles.css              # 修改：新增引导/设置样式
│   └── providers.json          # 新增：API 提供商预设数据
├── scripts/
│   └── start.bat               # 增强：自检 + 引导启动
├── .env.example                # 不变
└── src/math_agent/             # 完全不变
```

### 2.3 数据流

```
① 用户首次打开浏览器
    → Node 服务器检测无 .env（或 .env 缺少 OPENAI_API_KEY）
    → 健康检查返回 onboarding: true
    → 前端自动进入引导向导

② 引导向导5步
    → 逐步调 /api/env/* 和 /api/config/*

③ 完成
    → POST /api/config 写入 .env
    → 标记 onboarding 完成

④ 进入主界面
    → 正常使用流水线，配置已就绪

⑤ 设置页面（随时）
    → GET /api/config 读当前配置
    → 修改后 POST /api/config 保存
```

## 3. API 设计

所有新增 API 挂在 `/api/env/*` 和 `/api/config/*` 路径下，由 `server.mjs` 分发到对应路由模块。

### 3.1 环境检测与安装（`routes/env.mjs`）

| 方法 | 路径 | 功能 | 返回 |
|------|------|------|------|
| GET | `/api/env/check` | 检测 Python、Node.js、uv 是否已安装且版本达标 | `{ python: {installed, version, ok}, node: {installed, version, ok}, uv: {installed, version, ok}, allOk: boolean }` |
| POST | `/api/env/install` | 通过 winget 静默安装指定依赖 | `{ tool: "python"\|"node"\|"uv", status: "installing"\|"done"\|"failed", message: string }` |

**环境检测逻辑**（`GET /api/env/check`）：
- Python：`python --version` 或 `python3 --version`，解析版本号，校验 ≥3.11
- Node.js：`node --version`，解析版本号，校验 ≥18
- uv：`uv --version`，存在即可

**安装逻辑**（`POST /api/env/install`）：
- 请求体：`{ tool: "python" | "node" | "uv" }`
- Python：`winget install Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements`
- Node.js：`winget install OpenJS.NodeJS.LTS --silent --accept-package-agreements --accept-source-agreements`
- uv：`winget install astral-sh.uv --silent --accept-package-agreements --accept-source-agreements`
- 安装是**阻塞式**的：后端用 `child_process.spawnSync` 等待 winget 执行完毕，再返回最终结果
- 前端在调用时显示"安装中..."加载状态（带旋转动画），请求完成后根据返回的 `status` 显示成功/失败
- 设置 5 分钟超时，超时返回 `status: "failed", message: "安装超时"`
- 安装成功后返回 `status: "done"`，前端自动调 `GET /api/env/check` 刷新状态
- 安装失败时返回 winget 的错误输出，前端显示错误信息 + 手动下载链接作为降级方案
- winget 不可用时（旧版 Windows 无 winget），返回 `status: "failed", message: "winget 不可用"`，前端直接显示手动下载链接

### 3.2 配置读写与测试（`routes/config.mjs`）

| 方法 | 路径 | 功能 | 返回 |
|------|------|------|------|
| GET | `/api/config` | 读取当前 .env 配置 | 所有配置项的 JSON（API 密钥返回掩码） |
| POST | `/api/config` | 保存配置到 .env | `{ ok: boolean }` |
| POST | `/api/config/test-llm` | 用给定配置真实调用 LLM 测试连接 | `{ success: boolean, latency_ms: number, error?: string }` |

**GET `/api/config` 返回结构**：
```json
{
  "provider": "deepseek",
  "apiBase": "https://api.deepseek.com/v1",
  "apiKey": "sk-***...***5678",
  "defaultModel": "deepseek-chat",
  "strongModel": "deepseek-reasoner",
  "llmTimeout": 300,
  "maxModelIterations": 3,
  "ragEnabled": false,
  "ragEmbeddingModel": "text-embedding-3-small",
  "ragTopK": 4,
  "port": 5173
}
```

**POST `/api/config` 请求体**：同上结构（apiKey 为明文），后端写入 `.env` 文件。写入策略：保留 `.env` 中的注释和格式，仅替换对应行的值；新增的 key 追加到文件末尾。

**POST `/api/config/test-llm` 逻辑**：
- 请求体：`{ apiBase, apiKey, model }`
- 用 Node 内置 `fetch` 向 `${apiBase}/chat/completions` 发送一个极简请求：
  ```json
  { "model": "<model>", "messages": [{"role":"user","content":"hi"}], "max_tokens": 1 }
  ```
- Header：`Authorization: Bearer <apiKey>`
- 测量响应时间，返回 `success: true` + `latency_ms`
- 失败时返回 `success: false` + `error`（HTTP 状态码或错误消息）

### 3.3 引导状态（`routes/onboarding.mjs`）

| 方法 | 路径 | 功能 | 返回 |
|------|------|------|------|
| GET | `/api/onboarding/status` | 判断是否需要引导 | `{ needed: boolean, reason: "no_env" \| "no_api_key" \| "done" }` |

**判断逻辑**：
- `.env` 文件不存在 → `needed: true, reason: "no_env"`
- `.env` 存在但 `OPENAI_API_KEY` 为空或为默认值 `123456` → `needed: true, reason: "no_api_key"`
- `.env` 存在且 `OPENAI_API_KEY` 已配置 → `needed: false, reason: "done"`

此状态也合并到现有 `GET /api/health` 返回中，前端在 `app.js` 启动时检查，自动决定是否进入引导向导。

## 4. API 提供商预设（`providers.json`）

```json
[
  {
    "id": "deepseek",
    "name": "DeepSeek",
    "apiBase": "https://api.deepseek.com/v1",
    "defaultModel": "deepseek-chat",
    "strongModel": "deepseek-reasoner",
    "website": "https://platform.deepseek.com",
    "description": "性价比高，适合数学建模"
  },
  {
    "id": "openai",
    "name": "OpenAI",
    "apiBase": "https://api.openai.com/v1",
    "defaultModel": "openai/gpt-4o-mini",
    "strongModel": "openai/gpt-4o",
    "website": "https://platform.openai.com",
    "description": "GPT 系列，能力全面"
  },
  {
    "id": "zhipu",
    "name": "智谱 AI",
    "apiBase": "https://open.bigmodel.cn/api/paas/v4",
    "defaultModel": "glm-4-flash",
    "strongModel": "glm-4",
    "website": "https://open.bigmodel.cn",
    "description": "GLM 系列，国产大模型"
  },
  {
    "id": "moonshot",
    "name": "月之暗面",
    "apiBase": "https://api.moonshot.cn/v1",
    "defaultModel": "moonshot-v1-8k",
    "strongModel": "moonshot-v1-32k",
    "website": "https://platform.moonshot.cn",
    "description": "Kimi，长上下文优势"
  },
  {
    "id": "ollama",
    "name": "本地 Ollama",
    "apiBase": "http://localhost:11434/v1",
    "defaultModel": "llama3",
    "strongModel": "llama3",
    "website": "https://ollama.com",
    "description": "本地部署，离线可用"
  },
  {
    "id": "custom",
    "name": "自定义",
    "apiBase": "",
    "defaultModel": "",
    "strongModel": "",
    "website": "",
    "description": "手动填写端点和模型"
  }
]
```

前端在引导向导第2步加载此文件，用户选择提供商后自动填入 `apiBase`、`defaultModel`、`strongModel`。

## 5. 前端设计

### 5.1 引导向导（5步）

首次打开（`/api/onboarding/status` 返回 `needed: true`）时，前端自动渲染引导向导覆盖层，阻止进入主界面。

| 步骤 | 标题 | 交互 |
|------|------|------|
| 1/5 | 环境检测 | 调 `GET /api/env/check`，逐行显示 Python/Node/uv 状态。已装显示 ✓ + 版本号；未装的行显示 ✗ + "一键安装"按钮。点击安装调 `POST /api/env/install`，安装中显示进度提示，安装后可"重新检测"。全部就绪才能下一步。 |
| 2/5 | 选择 API 提供商 | 2×3 网格卡片，6个预设（DeepSeek/OpenAI/智谱/月之暗面/Ollama/自定义）。点击选中高亮，每个卡片显示名称、网站、描述。 |
| 3/5 | API 密钥 & 端点配置 | API 端点输入框（根据步骤2自动填入，可修改）；API 密钥输入框（password 类型 + 显示/隐藏切换按钮）。下方显示获取密钥的链接（跳转提供商网站）。 |
| 4/5 | 模型选择 & 连接测试 | 两个输入框：主力模型、强力模型（根据步骤2自动填入，可修改）。"测试连接"按钮调 `POST /api/config/test-llm`，分别测试两个模型，显示 ✓/✗ + 延迟。两个都成功才能下一步。 |
| 5/5 | 完成 | 展示配置摘要（提供商/端点/模型）。"进入工作台"按钮调 `POST /api/config` 保存配置，关闭引导覆盖层，进入主界面。 |

**UI 样式要求**：
- 输入框文字颜色为白色（`#fff`），字号 14px，确保清晰可读
- 辅助说明文字颜色 `#aaa`
- 深色背景，与现有 Beacon 主题一致
- 步骤指示器显示当前步骤 / 总步骤
- 底部"上一步"/"下一步"导航按钮

### 5.2 设置页面

在现有侧边栏（工作台/流水线/产物/知识库）新增"设置"入口。点击进入设置页面，左侧分类导航 + 右侧对应内容：

| 分类 | 包含配置项 |
|------|-----------|
| API 配置 | 提供商下拉切换、API 端点、API 密钥（显示/隐藏 + 测试按钮） |
| 模型选择 | 主力模型、强力模型 |
| RAG 知识库 | 启用开关、Embedding 模型、检索 Top-K |
| 高级设置 | 模型迭代轮次上限（滑块 1-5）、LLM 超时（秒）、前端端口 |

- 底部固定显示"重置默认"和"保存配置"按钮
- 保存时调 `POST /api/config`，提示保存成功
- 密钥读取时掩码显示，保存时发送明文
- 测试按钮调 `POST /api/config/test-llm`，实时显示连接状态

### 5.3 前端代码组织

在 `app.js` 中新增：
- `renderOnboarding()` — 渲染引导向导覆盖层
- `renderSettings()` — 渲染设置页面
- 步骤状态管理（当前步骤、各步收集的配置数据）
- API 调用封装（`apiGet`、`apiPost` 辅助函数）

在 `index.html` 中新增：
- 引导向导覆盖层容器 `<div id="onboarding-overlay">`
- 设置页面容器 `<div id="settings-view">`
- 侧边栏"设置"导航项

在 `styles.css` 中新增：
- `.onboarding-overlay` — 全屏覆盖层样式
- `.onboarding-step` — 步骤内容样式
- `.settings-page` — 设置页面布局样式
- `.settings-nav` — 左侧分类导航样式
- 输入框样式确保文字清晰可读

## 6. 启动脚本增强（`scripts/start.bat`）

当前 `start.bat` 已有基本的环境检查。增强点：

1. **更友好的提示**：检测到缺失依赖时，不只是报错，还提示"启动后可通过引导向导一键安装"
2. **不阻断启动**：即使 uv 未安装，也允许启动 Node 服务器（前端引导向导会引导安装），而不是直接退出
3. **自动打开浏览器**：保持现有行为，打开 `http://localhost:5173`
4. **Python 检测增强**：同时检测 `python` 和 `python3`，取可用者

具体逻辑：
```
1. 检查 Node.js → 未装则提示下载链接并退出（没有 Node 什么也做不了）
2. 检查 .env → 不存在则从 .env.example 复制（不提示用户手动编辑，引导向导会处理）
3. npm install（如果 node_modules 不存在）
4. uv sync（如果 uv 可用；不可用则跳过，不退出）
5. 启动 npm start + 打开浏览器
```

## 7. server.mjs 重构

当前 `server.mjs` 约 677 行，包含所有逻辑。重构为路由分发模式：

```javascript
// server.mjs（精简后）
import { handleEnvRoutes } from "./routes/env.mjs";
import { handleConfigRoutes } from "./routes/config.mjs";
import { handleOnboardingRoutes } from "./routes/onboarding.mjs";
// 现有的 run/artifact/upload 逻辑保留在 server.mjs 或拆到 routes/run.mjs

async function handleApi(req, res, parsed) {
  const { pathname } = parsed;

  // 新增路由
  if (pathname.startsWith("/api/env/"))      return handleEnvRoutes(req, res, parsed);
  if (pathname.startsWith("/api/config"))     return handleConfigRoutes(req, res, parsed);
  if (pathname.startsWith("/api/onboarding/")) return handleOnboardingRoutes(req, res, parsed);

  // 现有路由保留
  if (pathname === "/api/health") return handleHealth(req, res);
  if (pathname === "/api/run")    return handleRun(req, res);
  // ...
}
```

现有 `GET /api/health` 增强：在返回中合并 `onboarding` 状态，前端启动时一次性获取。

## 8. .env 读写策略

`routes/config.mjs` 中的 `.env` 读写逻辑：

**读取**：
- 用 `dotenv` 解析 `.env`（如果文件不存在，返回 `.env.example` 的默认值）
- `apiKey` 掩码处理：只显示前3位和后4位，中间用 `***` 替代

**写入**：
- 逐行读取现有 `.env`（或 `.env.example`），替换匹配的 key 行
- 保留注释行和空行不变
- 新增的 key 追加到文件末尾
- 写入后调用 `dotenv.config()` 重新加载，使配置立即生效（无需重启服务器）

**配置项映射**：

| 前端字段 | .env 变量 |
|---------|-----------|
| provider | （仅前端使用，不写入 .env） |
| apiBase | OPENAI_API_BASE |
| apiKey | OPENAI_API_KEY |
| defaultModel | MATH_AGENT_DEFAULT_MODEL |
| strongModel | MATH_AGENT_STRONG_MODEL |
| llmTimeout | MATH_AGENT_LLM_TIMEOUT |
| maxModelIterations | MATH_AGENT_MAX_MODEL_ITERATIONS |
| ragEnabled | MATH_AGENT_RAG_ENABLED（"1" 或 "0"） |
| ragEmbeddingModel | MATH_AGENT_RAG_EMBED |
| ragTopK | MATH_AGENT_RAG_TOPK |
| port | PORT |

## 9. 错误处理

| 场景 | 处理方式 |
|------|---------|
| winget 不可用（旧版 Windows） | 返回 `status: "failed", message: "winget 不可用"`，前端显示手动下载链接 |
| winget 安装失败 | 返回 winget 的错误输出，前端显示 + 手动下载链接降级 |
| winget 安装超时（>5分钟） | 返回 `status: "failed", message: "安装超时"`，前端显示手动下载链接降级 |
| LLM 测试连接超时 | 30 秒超时，返回 `success: false, error: "连接超时"` |
| LLM 测试连接返回 401 | 返回 `success: false, error: "API 密钥无效"` |
| .env 写入失败（权限问题） | 返回 500 + 错误信息 |
| 用户在引导中途刷新页面 | 重新从步骤1开始（状态不持久化，保持简单） |

## 10. 不做的事情（YAGNI）

- **不做** PyInstaller 打包成 exe — 依赖太复杂（sqlite-vec C 扩展、litellm 动态导入），不稳定
- **不做** Docker 容器化 — 目标用户是学生，Docker 本身是大依赖
- **不做** macOS/Linux 自动安装 — 仅 Windows
- **不做** API 密钥加密存储 — 继续 .env 明文，与现有架构一致
- **不做** 引导向导进度持久化 — 刷新重新开始，保持简单
- **不做** Python 后端改动 — 所有逻辑在 Node 侧完成
- **不做** 多用户/多配置文件 — 单机单用户场景

## 11. 测试策略

- **后端 API 测试**：扩展 `frontend/server.test.mjs`，新增 env/config/onboarding 路由的单元测试
- **.env 读写测试**：测试保留注释、替换值、追加新 key
- **LLM 测试连接**：mock fetch，测试成功/失败/超时场景
- **前端交互测试**：手动验证引导向导5步流程、设置页面读写
