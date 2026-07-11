# Beacon 打包与使用门槛降低 - 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 Beacon 架构上新增引导向导（5步）、设置页面、启动脚本增强，让用户无需手动编辑 .env 或安装命令行工具即可使用。

**Architecture:** Node 服务器全权负责环境检测/winget 安装/.env 读写/LLM 连接测试。将 server.mjs 的共享工具提取到 `frontend/lib/shared.mjs`，新增路由模块挂载到 `/api/env/*`、`/api/config/*`、`/api/onboarding/*`。前端新增引导向导覆盖层和设置页面。Python 后端完全不改。

**Tech Stack:** Node.js (原生 http)、vanilla HTML/CSS/JS、winget (Windows 包管理)、dotenv

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `frontend/lib/shared.mjs` | 新建 | 共享工具：sendJson、readJsonBody、safeProjectPath、HttpError、projectRoot、env 解析 |
| `frontend/providers.json` | 新建 | 6 个 API 提供商预设数据 |
| `frontend/routes/env.mjs` | 新建 | 环境检测 + winget 安装 |
| `frontend/routes/config.mjs` | 新建 | .env 读写 + LLM 连接测试 |
| `frontend/routes/onboarding.mjs` | 新建 | 引导状态判断 |
| `frontend/server.mjs` | 修改 | 路由分发 + health 增强 |
| `frontend/server.test.mjs` | 修改 | 新增 API 测试 |
| `frontend/index.html` | 修改 | 引导向导覆盖层 + 设置页入口 |
| `frontend/styles.css` | 修改 | 引导/设置样式 |
| `frontend/app.js` | 修改 | 引导/设置交互逻辑 |
| `scripts/start.bat` | 修改 | 不阻断启动 + 友好提示 |

---

### Task 1: 创建共享工具模块

**Files:**
- Create: `frontend/lib/shared.mjs`

- [ ] **Step 1: 创建 `frontend/lib/shared.mjs`**

从 `server.mjs` 提取共享工具，供路由模块复用。

```javascript
// frontend/lib/shared.mjs
import { resolve } from "node:path";
import { isAbsolute, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

export const root = resolve(fileURLToPath(new URL(".", import.meta.url)));
export const projectRoot = resolve(root, "..", "..");
export const env = globalThis.process?.env || {};

export class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export function sendJson(response, status, payload) {
  response.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(payload));
}

export function safeProjectPath(inputPath) {
  const target = resolve(projectRoot, inputPath || ".");
  const rel = relative(projectRoot, target);
  if (isAbsolute(rel) || rel === ".." || rel.startsWith(`..${sep}`)) {
    throw new HttpError(403, "Path is outside the project workspace.");
  }
  return target;
}

export async function readJsonBody(request) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > 1024 * 1024) throw new HttpError(413, "Request body exceeds 1 MiB.");
    chunks.push(chunk);
  }
  const text = Buffer.concat(chunks).toString("utf8");
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new HttpError(400, `Invalid JSON body: ${error.message}`);
  }
}
```

- [ ] **Step 2: 验证模块可被导入**

Run: `node -e "import('./frontend/lib/shared.mjs').then(m => console.log('OK', m.projectRoot))"`
Expected: 输出 `OK` + 项目根路径

- [ ] **Step 3: Commit**

```bash
git add frontend/lib/shared.mjs
git commit -m "feat: extract shared utilities to lib/shared.mjs"
```

---

### Task 2: 创建 API 提供商预设数据

**Files:**
- Create: `frontend/providers.json`

- [ ] **Step 1: 创建 `frontend/providers.json`**

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

- [ ] **Step 2: Commit**

```bash
git add frontend/providers.json
git commit -m "feat: add API provider presets (providers.json)"
```

---

### Task 3: 创建环境检测路由模块

**Files:**
- Create: `frontend/routes/env.mjs`

- [ ] **Step 1: 创建 `frontend/routes/env.mjs`**

```javascript
// frontend/routes/env.mjs
import { execSync } from "node:child_process";
import { sendJson, readJsonBody, HttpError } from "../lib/shared.mjs";

/**
 * 检测单个工具的版本信息。
 * @param {string} command - 命令名 (如 "python", "node", "uv")
 * @returns {{ installed: boolean, version: string }}
 */
function checkTool(command) {
  try {
    const output = execSync(`${command} --version`, {
      encoding: "utf8",
      timeout: 10_000,
      windowsHide: true,
      stdio: ["pipe", "pipe", "pipe"],
    }).trim();
    // 提取版本号：取输出中第一个 x.y.z 格式的字符串
    const match = output.match(/(\d+\.\d+(?:\.\d+)?)/);
    return { installed: true, version: match ? match[1] : output };
  } catch {
    return { installed: false, version: "" };
  }
}

/**
 * 比较版本号是否达标。
 * @param {string} version - 实际版本 "3.12.3"
 * @param {string} minVersion - 最低要求 "3.11"
 * @returns {boolean}
 */
function versionOk(version, minVersion) {
  if (!version) return false;
  const actual = version.split(".").map(Number);
  const required = minVersion.split(".").map(Number);
  for (let i = 0; i < required.length; i++) {
    const a = actual[i] || 0;
    const r = required[i] || 0;
    if (a > r) return true;
    if (a < r) return false;
  }
  return true;
}

const WINGET_PACKAGES = {
  python: "Python.Python.3.12",
  node: "OpenJS.NodeJS.LTS",
  uv: "astral-sh.uv",
};

/**
 * 处理 /api/env/* 路由。
 */
export async function handleEnvRoutes(request, response, url) {
  // GET /api/env/check - 检测环境
  if (request.method === "GET" && url.pathname === "/api/env/check") {
    const python = checkTool("python");
    const python3 = !python.installed ? checkTool("python3") : null;
    const node = checkTool("node");
    const uv = checkTool("uv");

    const pythonResult = python.installed
      ? python
      : python3 && python3.installed
        ? python3
        : { installed: false, version: "" };

    sendJson(response, 200, {
      python: {
        installed: pythonResult.installed,
        version: pythonResult.version,
        ok: pythonResult.installed && versionOk(pythonResult.version, "3.11"),
      },
      node: {
        installed: node.installed,
        version: node.version,
        ok: node.installed && versionOk(node.version, "18"),
      },
      uv: {
        installed: uv.installed,
        version: uv.version,
        ok: uv.installed,
      },
      allOk:
        pythonResult.installed &&
        versionOk(pythonResult.version, "3.11") &&
        node.installed &&
        versionOk(node.version, "18") &&
        uv.installed,
    });
    return;
  }

  // POST /api/env/install - winget 安装
  if (request.method === "POST" && url.pathname === "/api/env/install") {
    const body = await readJsonBody(request);
    const tool = body.tool;
    if (!tool || !WINGET_PACKAGES[tool]) {
      throw new HttpError(400, `Invalid tool. Must be one of: ${Object.keys(WINGET_PACKAGES).join(", ")}`);
    }

    // 检查 winget 是否可用
    try {
      execSync("winget --version", { encoding: "utf8", timeout: 10_000, windowsHide: true, stdio: ["pipe", "pipe", "pipe"] });
    } catch {
      sendJson(response, 200, {
        tool,
        status: "failed",
        message: "winget 不可用，请手动安装",
        downloadUrl: getDownloadUrl(tool),
      });
      return;
    }

    // 阻塞式安装
    const pkg = WINGET_PACKAGES[tool];
    try {
      const output = execSync(
        `winget install ${pkg} --silent --accept-package-agreements --accept-source-agreements`,
        {
          encoding: "utf8",
          timeout: 300_000, // 5 分钟
          windowsHide: true,
          stdio: ["pipe", "pipe", "pipe"],
        },
      );
      sendJson(response, 200, {
        tool,
        status: "done",
        message: "安装完成",
        output: output.slice(-500),
      });
    } catch (error) {
      sendJson(response, 200, {
        tool,
        status: "failed",
        message: error.killed ? "安装超时（超过5分钟）" : "安装失败",
        output: String(error.stderr || error.stdout || error.message).slice(-500),
        downloadUrl: getDownloadUrl(tool),
      });
    }
    return;
  }

  // GET /api/providers - 返回提供商预设
  if (request.method === "GET" && url.pathname === "/api/providers") {
    const { readFile } = await import("node:fs/promises");
    const { resolve } = await import("node:path");
    const { projectRoot } = await import("../lib/shared.mjs");
    try {
      const raw = await readFile(resolve(projectRoot, "frontend/providers.json"), "utf8");
      sendJson(response, 200, JSON.parse(raw));
    } catch {
      sendJson(response, 200, []);
    }
    return;
  }

  sendJson(response, 404, { error: "Unknown env route." });
}

function getDownloadUrl(tool) {
  const urls = {
    python: "https://www.python.org/downloads/",
    node: "https://nodejs.org/",
    uv: "https://docs.astral.sh/uv/getting-started/installation/",
  };
  return urls[tool] || "";
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/routes/env.mjs
git commit -m "feat: add env check + winget install route module"
```

---

### Task 4: 创建配置读写路由模块

**Files:**
- Create: `frontend/routes/config.mjs`

- [ ] **Step 1: 创建 `frontend/routes/config.mjs`**

```javascript
// frontend/routes/config.mjs
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { sendJson, readJsonBody, HttpError, projectRoot } from "../lib/shared.mjs";

const ENV_PATH = resolve(projectRoot, ".env");
const ENV_EXAMPLE_PATH = resolve(projectRoot, ".env.example");

/** 前端字段 -> .env 变量名映射 */
const FIELD_MAP = {
  apiBase: "OPENAI_API_BASE",
  apiKey: "OPENAI_API_KEY",
  defaultModel: "MATH_AGENT_DEFAULT_MODEL",
  strongModel: "MATH_AGENT_STRONG_MODEL",
  llmTimeout: "MATH_AGENT_LLM_TIMEOUT",
  maxModelIterations: "MATH_AGENT_MAX_MODEL_ITERATIONS",
  ragEnabled: "MATH_AGENT_RAG_ENABLED",
  ragEmbeddingModel: "MATH_AGENT_RAG_EMBED",
  ragTopK: "MATH_AGENT_RAG_TOPK",
  port: "PORT",
};

/** .env 变量名 -> 前端字段 反向映射 */
const ENV_TO_FIELD = Object.fromEntries(
  Object.entries(FIELD_MAP).map(([field, envKey]) => [envKey, field]),
);

/**
 * 解析 .env 文件为 key-value 对象。
 */
function parseEnvFile(filePath) {
  if (!existsSync(filePath)) return {};
  const content = readFileSync(filePath, "utf8");
  const result = {};
  for (const line of content.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eqIndex = trimmed.indexOf("=");
    if (eqIndex === -1) continue;
    const key = trimmed.slice(0, eqIndex).trim();
    const value = trimmed.slice(eqIndex + 1).trim();
    result[key] = value;
  }
  return result;
}

/**
 * 掩码 API 密钥：显示前3位 + *** + 后4位。
 */
function maskApiKey(key) {
  if (!key || key.length < 8) return key ? "***" : "";
  return `${key.slice(0, 3)}***${key.slice(-4)}`;
}

/**
 * 更新 .env 文件，保留注释和格式，仅替换对应行的值。
 */
function updateEnvFile(updates) {
  const templatePath = existsSync(ENV_PATH) ? ENV_PATH : ENV_EXAMPLE_PATH;
  const content = readFileSync(templatePath, "utf8");
  const lines = content.split("\n");
  const remaining = new Set(Object.keys(updates));

  const result = lines.map((line) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) return line;
    const eqIndex = trimmed.indexOf("=");
    if (eqIndex === -1) return line;
    const key = trimmed.slice(0, eqIndex).trim();
    if (key in updates) {
      remaining.delete(key);
      return `${key}=${updates[key]}`;
    }
    return line;
  });

  // 追加 .env 中不存在的新 key
  for (const key of remaining) {
    result.push(`${key}=${updates[key]}`);
  }

  writeFileSync(ENV_PATH, result.join("\n"), "utf8");
}

/**
 * 处理 /api/config* 路由。
 */
export async function handleConfigRoutes(request, response, url) {
  // GET /api/config - 读取当前配置
  if (request.method === "GET" && url.pathname === "/api/config") {
    const envVars = parseEnvFile(ENV_PATH);
    const config = {
      apiBase: envVars.OPENAI_API_BASE || "",
      apiKey: maskApiKey(envVars.OPENAI_API_KEY || ""),
      hasApiKey: !!(envVars.OPENAI_API_KEY && envVars.OPENAI_API_KEY !== "123456"),
      defaultModel: envVars.MATH_AGENT_DEFAULT_MODEL || "",
      strongModel: envVars.MATH_AGENT_STRONG_MODEL || "",
      llmTimeout: Number(envVars.MATH_AGENT_LLM_TIMEOUT || 300),
      maxModelIterations: Number(envVars.MATH_AGENT_MAX_MODEL_ITERATIONS || 3),
      ragEnabled: envVars.MATH_AGENT_RAG_ENABLED === "1",
      ragEmbeddingModel: envVars.MATH_AGENT_RAG_EMBED || "text-embedding-3-small",
      ragTopK: Number(envVars.MATH_AGENT_RAG_TOPK || 4),
      port: Number(envVars.PORT || 5173),
    };
    sendJson(response, 200, config);
    return;
  }

  // POST /api/config - 保存配置到 .env
  if (request.method === "POST" && url.pathname === "/api/config") {
    const body = await readJsonBody(request);

    const updates = {};
    // 只更新提供的字段
    if (body.apiBase !== undefined) updates[FIELD_MAP.apiBase] = body.apiBase;
    // apiKey: 如果值包含 *** 表示是掩码值，跳过不更新
    if (body.apiKey !== undefined && !body.apiKey.includes("***")) {
      updates[FIELD_MAP.apiKey] = body.apiKey;
    }
    if (body.defaultModel !== undefined) updates[FIELD_MAP.defaultModel] = body.defaultModel;
    if (body.strongModel !== undefined) updates[FIELD_MAP.strongModel] = body.strongModel;
    if (body.llmTimeout !== undefined) updates[FIELD_MAP.llmTimeout] = String(body.llmTimeout);
    if (body.maxModelIterations !== undefined) updates[FIELD_MAP.maxModelIterations] = String(body.maxModelIterations);
    if (body.ragEnabled !== undefined) updates[FIELD_MAP.ragEnabled] = body.ragEnabled ? "1" : "0";
    if (body.ragEmbeddingModel !== undefined) updates[FIELD_MAP.ragEmbeddingModel] = body.ragEmbeddingModel;
    if (body.ragTopK !== undefined) updates[FIELD_MAP.ragTopK] = String(body.ragTopK);
    if (body.port !== undefined) updates[FIELD_MAP.port] = String(body.port);

    if (Object.keys(updates).length === 0) {
      throw new HttpError(400, "No config fields to update.");
    }

    updateEnvFile(updates);
    sendJson(response, 200, { ok: true });
    return;
  }

  // POST /api/config/test-llm - 测试 LLM 连接
  if (request.method === "POST" && url.pathname === "/api/config/test-llm") {
    const body = await readJsonBody(request);
    const { apiBase, apiKey, model } = body;

    if (!apiBase || !apiKey || !model) {
      throw new HttpError(400, "apiBase, apiKey, and model are required.");
    }

    const startTime = Date.now();
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 30_000);

      const res = await fetch(`${apiBase.replace(/\/+$/, "")}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: "hi" }],
          max_tokens: 1,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeout);
      const latency = Date.now() - startTime;

      if (res.ok) {
        sendJson(response, 200, { success: true, latency_ms: latency });
      } else {
        const errorText = await res.text().catch(() => "");
        let errorMsg = `HTTP ${res.status}`;
        if (res.status === 401) errorMsg = "API 密钥无效";
        else if (res.status === 404) errorMsg = "模型不存在或端点错误";
        else if (errorText) errorMsg = `${errorMsg}: ${errorText.slice(0, 200)}`;
        sendJson(response, 200, { success: false, latency_ms: latency, error: errorMsg });
      }
    } catch (error) {
      const latency = Date.now() - startTime;
      const errorMsg = error.name === "AbortError"
        ? "连接超时（30秒）"
        : `连接失败: ${error.message}`;
      sendJson(response, 200, { success: false, latency_ms: latency, error: errorMsg });
    }
    return;
  }

  sendJson(response, 404, { error: "Unknown config route." });
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/routes/config.mjs
git commit -m "feat: add config read/write + LLM test route module"
```

---

### Task 5: 创建引导状态路由模块

**Files:**
- Create: `frontend/routes/onboarding.mjs`

- [ ] **Step 1: 创建 `frontend/routes/onboarding.mjs`**

```javascript
// frontend/routes/onboarding.mjs
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { sendJson, projectRoot } from "../lib/shared.mjs";

const ENV_PATH = resolve(projectRoot, ".env");

/**
 * 判断是否需要引导。
 * @returns {{ needed: boolean, reason: "no_env" | "no_api_key" | "done" }}
 */
export function getOnboardingStatus() {
  if (!existsSync(ENV_PATH)) {
    return { needed: true, reason: "no_env" };
  }

  const content = readFileSync(ENV_PATH, "utf8");
  const apiKeyMatch = content.match(/^OPENAI_API_KEY\s*=\s*(.+)$/m);
  const apiKey = apiKeyMatch ? apiKeyMatch[1].trim() : "";

  if (!apiKey || apiKey === "123456") {
    return { needed: true, reason: "no_api_key" };
  }

  return { needed: false, reason: "done" };
}

/**
 * 处理 /api/onboarding/* 路由。
 */
export async function handleOnboardingRoutes(request, response, url) {
  // GET /api/onboarding/status
  if (request.method === "GET" && url.pathname === "/api/onboarding/status") {
    sendJson(response, 200, getOnboardingStatus());
    return;
  }

  sendJson(response, 404, { error: "Unknown onboarding route." });
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/routes/onboarding.mjs
git commit -m "feat: add onboarding status route module"
```

---

### Task 6: 重构 server.mjs 接入新路由

**Files:**
- Modify: `frontend/server.mjs`

- [ ] **Step 1: 在 server.mjs 顶部添加路由导入**

在 `frontend/server.mjs` 第 9 行（`import { fileURLToPath }` 之后）添加：

```javascript
import { handleEnvRoutes } from "./routes/env.mjs";
import { handleConfigRoutes } from "./routes/config.mjs";
import { handleOnboardingRoutes, getOnboardingStatus } from "./routes/onboarding.mjs";
```

- [ ] **Step 2: 在 handleApi 函数中添加新路由分发**

在 `frontend/server.mjs` 的 `handleApi` 函数开头（第 249 行 `if (request.method === "POST" && url.pathname === "/api/upload")` 之前）插入：

```javascript
  // --- 新增路由：环境检测、配置、引导 ---
  if (url.pathname.startsWith("/api/env/") || url.pathname === "/api/providers") {
    return handleEnvRoutes(request, response, url);
  }
  if (url.pathname.startsWith("/api/config")) {
    return handleConfigRoutes(request, response, url);
  }
  if (url.pathname.startsWith("/api/onboarding/")) {
    return handleOnboardingRoutes(request, response, url);
  }
```

- [ ] **Step 3: 增强 health 端点，合并 onboarding 状态**

在 `frontend/server.mjs` 中找到 health 端点（约第 313 行），替换为：

```javascript
  if (request.method === "GET" && url.pathname === "/api/health") {
    const fixtures = await listFixtures().catch(() => []);
    const onboarding = getOnboardingStatus();
    sendJson(response, 200, {
      ok: true,
      projectRoot,
      mathAgentCommand: env.MATH_AGENT_COMMAND || "uv run math-agent",
      fixtures: fixtures.length,
      onboarding,
    });
    return;
  }
```

- [ ] **Step 4: 验证服务器能启动**

Run: `node -e "import('./frontend/lib/shared.mjs').then(() => console.log('shared OK'))" && node -e "import('./frontend/routes/env.mjs').then(() => console.log('env OK'))" && node -e "import('./frontend/routes/config.mjs').then(() => console.log('config OK'))" && node -e "import('./frontend/routes/onboarding.mjs').then(() => console.log('onboarding OK'))"`
Expected: 四行 `OK`

- [ ] **Step 5: Commit**

```bash
git add frontend/server.mjs
git commit -m "feat: wire env/config/onboarding routes into server.mjs + enhance health"
```

---

### Task 7: 编写后端 API 集成测试

**Files:**
- Modify: `frontend/server.test.mjs`

- [ ] **Step 1: 在 `frontend/server.test.mjs` 末尾追加新测试**

在文件末尾（第 172 行之后）追加：

```javascript
test("GET /api/env/check 返回环境检测结果", async () => {
  const response = await fetch(`${base}/api/env/check`);
  assert.equal(response.status, 200);
  const data = await response.json();
  assert.ok(data.python);
  assert.ok(data.node);
  assert.ok(data.uv);
  assert.equal(typeof data.allOk, "boolean");
});

test("GET /api/onboarding/status 返回引导状态", async () => {
  const response = await fetch(`${base}/api/onboarding/status`);
  assert.equal(response.status, 200);
  const data = await response.json();
  assert.ok(["no_env", "no_api_key", "done"].includes(data.reason));
  assert.equal(typeof data.needed, "boolean");
});

test("GET /api/config 返回配置（密钥掩码）", async () => {
  const response = await fetch(`${base}/api/config`);
  assert.equal(response.status, 200);
  const data = await response.json();
  assert.ok(data.apiBase !== undefined);
  assert.ok(data.defaultModel !== undefined);
  assert.ok(data.strongModel !== undefined);
  // apiKey 不应包含明文（如果存在的话应被掩码）
  if (data.apiKey) {
    assert.ok(data.apiKey.includes("***") || data.apiKey.length === 0);
  }
});

test("GET /api/health 包含 onboarding 字段", async () => {
  const response = await fetch(`${base}/api/health`);
  const data = await response.json();
  assert.ok(data.onboarding);
  assert.ok(["no_env", "no_api_key", "done"].includes(data.onboarding.reason));
});

test("POST /api/config/test-llm 拒绝缺少参数的请求", async () => {
  const response = await fetch(`${base}/api/config/test-llm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ apiBase: "https://example.com/v1" }),
  });
  assert.equal(response.status, 400);
});

test("GET /api/providers 返回提供商列表", async () => {
  const response = await fetch(`${base}/api/providers`);
  assert.equal(response.status, 200);
  const data = await response.json();
  assert.ok(Array.isArray(data));
  assert.ok(data.length >= 5);
  assert.ok(data.some((p) => p.id === "deepseek"));
});
```

- [ ] **Step 2: 运行测试**

Run: `node --test frontend/server.test.mjs`
Expected: 所有测试 PASS（包括原有测试）

- [ ] **Step 3: Commit**

```bash
git add frontend/server.test.mjs
git commit -m "test: add integration tests for env/config/onboarding APIs"
```

---

### Task 8: 添加引导向导和设置页面的 HTML

**Files:**
- Modify: `frontend/index.html`

- [ ] **Step 1: 在侧边栏导航添加"设置"入口**

在 `frontend/index.html` 第 23 行（`<a href="#knowledge">知识库</a>` 之后）添加：

```html
          <a href="#settings">设置</a>
```

- [ ] **Step 2: 在 `</main>` 之前添加设置页面容器**

在 `frontend/index.html` 找到 `</main>`（第 230 行），在其之前添加：

```html
        <section class="panel settings-panel" id="settings" hidden aria-labelledby="settings-title">
          <div class="panel-heading">
            <div>
              <p class="eyebrow">Configuration</p>
              <h2 id="settings-title">设置</h2>
            </div>
          </div>
          <div class="settings-layout">
            <nav class="settings-nav" aria-label="设置分类">
              <button class="active" type="button" data-settings-tab="api">API 配置</button>
              <button type="button" data-settings-tab="models">模型选择</button>
              <button type="button" data-settings-tab="rag">RAG 知识库</button>
              <button type="button" data-settings-tab="advanced">高级设置</button>
            </nav>
            <div class="settings-content" id="settingsContent"></div>
          </div>
          <div class="settings-footer">
            <button class="ghost-button" type="button" id="settingsReset">重置默认</button>
            <button class="primary-button" type="button" id="settingsSave">💾 保存配置</button>
          </div>
        </section>
```

- [ ] **Step 3: 在 `<div class="toast"` 之前添加引导向导覆盖层**

在 `frontend/index.html` 找到 `<div class="toast"` （第 233 行），在其之前添加：

```html
    <div class="onboarding-overlay" id="onboardingOverlay" hidden>
      <div class="onboarding-card">
        <div class="onboarding-header">
          <h2 id="onboardingTitle">欢迎使用 Beacon</h2>
          <p id="onboardingSubtitle">让我们花几分钟完成初始配置</p>
          <div class="onboarding-steps">
            <span class="step-dot active" data-step="1"></span>
            <span class="step-dot" data-step="2"></span>
            <span class="step-dot" data-step="3"></span>
            <span class="step-dot" data-step="4"></span>
            <span class="step-dot" data-step="5"></span>
          </div>
        </div>
        <div class="onboarding-body" id="onboardingBody"></div>
        <div class="onboarding-footer">
          <button class="ghost-button" type="button" id="onboardingPrev" disabled>上一步</button>
          <button class="primary-button" type="button" id="onboardingNext">下一步</button>
        </div>
      </div>
    </div>
```

- [ ] **Step 4: Commit**

```bash
git add frontend/index.html
git commit -m "feat: add onboarding overlay + settings panel to index.html"
```

---

### Task 9: 添加引导向导和设置页面的 CSS

**Files:**
- Modify: `frontend/styles.css`

- [ ] **Step 1: 在 `frontend/styles.css` 末尾追加引导向导和设置页面样式**

```css
/* ---- 引导向导覆盖层 ---- */

.onboarding-overlay {
  position: fixed;
  inset: 0;
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(7, 27, 58, 0.6);
  backdrop-filter: blur(6px);
}

.onboarding-overlay[hidden] {
  display: none;
}

.onboarding-card {
  width: min(620px, calc(100vw - 48px));
  max-height: calc(100vh - 48px);
  display: flex;
  flex-direction: column;
  padding: 32px;
  border-radius: 16px;
  background: var(--white);
  box-shadow: 0 40px 80px rgba(7, 27, 58, 0.25);
  overflow-y: auto;
}

.onboarding-header h2 {
  margin: 0;
  font-size: 1.5rem;
}

.onboarding-header p {
  margin: 6px 0 0;
  color: var(--muted);
  font-size: 0.92rem;
}

.onboarding-steps {
  display: flex;
  gap: 8px;
  margin-top: 16px;
}

.step-dot {
  flex: 1;
  height: 4px;
  border-radius: 999px;
  background: var(--line);
  transition: background 200ms ease;
}

.step-dot.active {
  background: var(--gold);
}

.step-dot.done {
  background: var(--green);
}

.onboarding-body {
  flex: 1;
  min-height: 200px;
  margin: 24px 0;
}

.onboarding-body h3 {
  margin: 0 0 12px;
  font-size: 1.1rem;
}

.onboarding-body p {
  color: var(--ink-soft);
  line-height: 1.6;
}

.onboarding-footer {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}

/* 引导步骤内容 */

.env-check-list {
  display: grid;
  gap: 10px;
}

.env-check-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfe;
}

.env-check-item.ok {
  border-color: var(--green);
  background: #f0faf6;
}

.env-check-item.missing {
  border-color: var(--rose);
  background: #fff0f3;
}

.env-check-item .env-name {
  font-weight: 700;
  color: var(--ink);
}

.env-check-item .env-version {
  display: block;
  margin-top: 2px;
  font-size: 0.84rem;
  color: var(--muted);
}

.env-check-item .env-status-ok {
  color: var(--green);
  font-weight: 700;
}

.env-check-item .env-status-missing {
  color: var(--rose);
  font-weight: 700;
}

.env-install-btn {
  min-height: 34px;
  padding: 0 14px;
  border: none;
  border-radius: 6px;
  background: var(--gold);
  color: var(--ink);
  font-weight: 700;
  cursor: pointer;
}

.env-install-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.env-install-btn.installing {
  background: var(--muted);
}

.provider-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
}

.provider-card {
  padding: 14px;
  border: 2px solid var(--line);
  border-radius: 8px;
  background: #fbfcfe;
  cursor: pointer;
  transition: border-color 150ms ease;
}

.provider-card:hover {
  border-color: var(--blue);
}

.provider-card.selected {
  border-color: var(--gold);
  background: var(--gold-soft);
}

.provider-card strong {
  display: block;
  font-size: 1rem;
}

.provider-card span {
  display: block;
  margin-top: 4px;
  color: var(--muted);
  font-size: 0.82rem;
}

/* ---- 设置页面 ---- */

.settings-panel[hidden] {
  display: none;
}

.settings-layout {
  display: flex;
  gap: 20px;
  min-height: 320px;
}

.settings-nav {
  display: flex;
  flex-direction: column;
  gap: 4px;
  width: 140px;
  flex-shrink: 0;
}

.settings-nav button {
  padding: 10px 14px;
  text-align: left;
  border: none;
  border-left: 3px solid transparent;
  border-radius: 0;
  background: transparent;
  color: var(--muted);
  font-weight: 600;
  cursor: pointer;
}

.settings-nav button.active {
  color: var(--blue);
  font-weight: 800;
  border-left-color: var(--blue);
  background: #f0f6ff;
}

.settings-content {
  flex: 1;
  display: grid;
  gap: 14px;
  align-content: start;
}

.settings-content .field {
  margin-bottom: 0;
}

.settings-content .field span {
  font-size: 0.86rem;
}

.settings-content .field input,
.settings-content .field select {
  width: 100%;
  height: 42px;
  padding: 0 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  color: var(--ink);
  background: var(--white);
  font-size: 14px;
  outline: 0;
}

.settings-content .field input:focus,
.settings-content .field select:focus {
  border-color: var(--blue);
  box-shadow: 0 0 0 3px rgba(21, 94, 150, 0.13);
}

.settings-content .field-hint {
  color: var(--muted);
  font-size: 0.8rem;
}

.settings-content .test-result {
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 0.84rem;
}

.settings-content .test-result.ok {
  background: #dff6eb;
  color: var(--green);
}

.settings-content .test-result.fail {
  background: #fff0f3;
  color: var(--rose);
}

.settings-content .toggle-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfe;
}

.toggle-switch {
  position: relative;
  width: 44px;
  height: 24px;
  border-radius: 12px;
  background: var(--line);
  cursor: pointer;
  transition: background 200ms ease;
}

.toggle-switch.on {
  background: var(--green);
}

.toggle-switch::after {
  content: "";
  position: absolute;
  top: 2px;
  left: 2px;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--white);
  transition: transform 200ms ease;
}

.toggle-switch.on::after {
  transform: translateX(20px);
}

.settings-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 16px;
  padding-top: 14px;
  border-top: 1px solid var(--line);
}

@media (max-width: 860px) {
  .settings-layout {
    flex-direction: column;
  }
  .settings-nav {
    flex-direction: row;
    width: 100%;
    overflow-x: auto;
  }
  .settings-nav button {
    border-left: none;
    border-bottom: 3px solid transparent;
    white-space: nowrap;
  }
  .settings-nav button.active {
    border-left: none;
    border-bottom-color: var(--blue);
  }
  .provider-grid {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/styles.css
git commit -m "feat: add onboarding overlay + settings page styles"
```

---

### Task 10: 添加引导向导和设置页面的 JS 逻辑

**Files:**
- Modify: `frontend/app.js`

- [ ] **Step 1: 在 `frontend/app.js` 开头添加 DOM 引用**

在第 35 行（`const pipelineItems = ...` 之后）添加：

```javascript
const onboardingOverlay = document.querySelector("#onboardingOverlay");
const onboardingBody = document.querySelector("#onboardingBody");
const onboardingTitle = document.querySelector("#onboardingTitle");
const onboardingSubtitle = document.querySelector("#onboardingSubtitle");
const onboardingPrev = document.querySelector("#onboardingPrev");
const onboardingNext = document.querySelector("#onboardingNext");
const stepDots = document.querySelectorAll(".step-dot");
const settingsPanel = document.querySelector("#settings");
const settingsContent = document.querySelector("#settingsContent");
const settingsSave = document.querySelector("#settingsSave");
const settingsReset = document.querySelector("#settingsReset");
const settingsNavButtons = document.querySelectorAll("[data-settings-tab]");
```

- [ ] **Step 2: 在 `frontend/app.js` 末尾添加引导向导逻辑**

在文件末尾追加：

```javascript
// ============ 引导向导 ============

let onboardingStep = 1;
let onboardingData = {
  provider: null,
  apiBase: "",
  apiKey: "",
  defaultModel: "",
  strongModel: "",
};

const onboardingSteps = [
  { title: "欢迎使用 Beacon", subtitle: "让我们花几分钟完成初始配置" },
  { title: "环境检测", subtitle: "检查运行所需依赖是否已安装" },
  { title: "选择 API 提供商", subtitle: "选择你的 LLM 服务提供商" },
  { title: "API 密钥 & 端点配置", subtitle: "填写 API 密钥和访问端点" },
  { title: "模型选择 & 连接测试", subtitle: "选择模型并验证连接" },
  { title: "配置完成", subtitle: "一切就绪，可以开始使用了" },
];

function showOnboarding() {
  onboardingOverlay.hidden = false;
  onboardingStep = 1;
  renderOnboardingStep();
}

function hideOnboarding() {
  onboardingOverlay.hidden = true;
}

function updateStepDots() {
  stepDots.forEach((dot, i) => {
    dot.classList.toggle("active", i + 1 === onboardingStep);
    dot.classList.toggle("done", i + 1 < onboardingStep);
  });
}

function renderOnboardingStep() {
  const step = onboardingSteps[onboardingStep - 1];
  onboardingTitle.textContent = step.title;
  onboardingSubtitle.textContent = step.subtitle;
  updateStepDots();
  onboardingPrev.disabled = onboardingStep === 1;
  onboardingNext.textContent = onboardingStep === 5 ? "进入工作台" : "下一步";

  if (onboardingStep === 1) renderEnvCheckStep();
  else if (onboardingStep === 2) renderProviderStep();
  else if (onboardingStep === 3) renderApiKeyStep();
  else if (onboardingStep === 4) renderModelTestStep();
  else if (onboardingStep === 5) renderCompleteStep();
}

async function renderEnvCheckStep() {
  onboardingBody.innerHTML = `
    <h3>环境检测</h3>
    <p>正在检测运行所需依赖...</p>
    <div class="env-check-list" id="envCheckList"></div>
  `;
  try {
    const data = await api("/api/env/check");
    const list = document.querySelector("#envCheckList");
    list.innerHTML = renderEnvItem("Python", "≥ 3.11", data.python) +
      renderEnvItem("Node.js", "≥ 18", data.node) +
      renderEnvItem("uv", "Python 包管理器", data.uv);
    onboardingNext.disabled = !data.allOk;
    if (!data.allOk) {
      list.querySelectorAll(".env-install-btn").forEach((btn) => {
        btn.addEventListener("click", () => installTool(btn.dataset.tool, btn));
      });
    }
  } catch (error) {
    onboardingBody.innerHTML = `<p>检测失败: ${escapeHtml(error.message)}</p>`;
  }
}

function renderEnvItem(name, requirement, info) {
  const ok = info.ok;
  const status = ok
    ? `<span class="env-status-ok">✓ ${info.version}</span>`
    : `<span class="env-status-missing">✗ ${info.installed ? info.version + " 版本过低" : "未检测到"}</span>`;
  const action = ok ? "" : `<button class="env-install-btn" data-tool="${name.toLowerCase().replace(".", "")}">一键安装</button>`;
  // 修正 tool 名称映射
  const toolName = name === "Python" ? "python" : name === "Node.js" ? "node" : name === "uv" ? "uv" : "";
  const actionBtn = ok ? "" : `<button class="env-install-btn" data-tool="${toolName}">一键安装</button>`;
  return `
    <div class="env-check-item ${ok ? "ok" : "missing"}">
      <div>
        <span class="env-name">${escapeHtml(name)}</span>
        <span class="env-version">${escapeHtml(requirement)}</span>
      </div>
      <div style="display:flex;align-items:center;gap:10px;">
        ${status}
        ${actionBtn}
      </div>
    </div>
  `;
}

async function installTool(tool, btn) {
  btn.disabled = true;
  btn.classList.add("installing");
  btn.textContent = "安装中...";
  try {
    const result = await api("/api/env/install", {
      method: "POST",
      body: JSON.stringify({ tool }),
    });
    if (result.status === "done") {
      showToast(`${tool} 安装完成`);
      renderEnvCheckStep();
    } else {
      btn.disabled = false;
      btn.classList.remove("installing");
      btn.textContent = "重试";
      showToast(`${tool} 安装失败: ${result.message}`);
      if (result.downloadUrl) {
        onboardingBody.insertAdjacentHTML("beforeend",
          `<p style="margin-top:12px;color:var(--rose);">手动安装: <a href="${result.downloadUrl}" target="_blank">${result.downloadUrl}</a></p>`);
      }
    }
  } catch (error) {
    btn.disabled = false;
    btn.classList.remove("installing");
    btn.textContent = "重试";
    showToast(`安装失败: ${error.message}`);
  }
}

async function renderProviderStep() {
  onboardingNext.disabled = true;
  onboardingBody.innerHTML = `
    <h3>选择 API 提供商</h3>
    <p>选择你的 LLM 服务提供商，选择后会自动填入端点和推荐模型。</p>
    <div class="provider-grid" id="providerGrid"></div>
  `;
  try {
    const providers = await api("/api/providers");
    const grid = document.querySelector("#providerGrid");
    grid.innerHTML = providers.map((p) => `
      <div class="provider-card" data-provider="${escapeHtml(p.id)}">
        <strong>${escapeHtml(p.name)}</strong>
        <span>${escapeHtml(p.description)}</span>
      </div>
    `).join("");
    grid.querySelectorAll(".provider-card").forEach((card) => {
      card.addEventListener("click", () => {
        grid.querySelectorAll(".provider-card").forEach((c) => c.classList.remove("selected"));
        card.classList.add("selected");
        const provider = providers.find((p) => p.id === card.dataset.provider);
        onboardingData.provider = provider.id;
        onboardingData.apiBase = provider.apiBase;
        onboardingData.defaultModel = provider.defaultModel;
        onboardingData.strongModel = provider.strongModel;
        onboardingNext.disabled = false;
      });
    });
  } catch (error) {
    onboardingBody.innerHTML = `<p>加载提供商列表失败: ${escapeHtml(error.message)}</p>`;
  }
}

function renderApiKeyStep() {
  onboardingBody.innerHTML = `
    <h3>API 密钥 & 端点配置</h3>
    <p>填写你的 API 密钥。端点已根据所选提供商自动填入，可修改。</p>
    <div class="field">
      <span>API 端点（Base URL）</span>
      <input id="obApiBase" value="${escapeHtml(onboardingData.apiBase)}" placeholder="https://api.deepseek.com/v1" style="width:100%;height:42px;padding:0 12px;border:1px solid var(--line);border-radius:8px;font-size:14px;color:var(--ink);background:var(--white);outline:0;" />
    </div>
    <div class="field">
      <span>API 密钥</span>
      <div style="display:flex;gap:8px;">
        <input id="obApiKey" type="password" value="${escapeHtml(onboardingData.apiKey)}" placeholder="sk-..." style="flex:1;height:42px;padding:0 12px;border:1px solid var(--line);border-radius:8px;font-size:14px;color:var(--ink);background:var(--white);outline:0;" />
        <button class="ghost-button" type="button" id="obToggleKey">👁 显示</button>
      </div>
    </div>
  `;
  const apiBaseInput = document.querySelector("#obApiBase");
  const apiKeyInput = document.querySelector("#obApiKey");
  const toggleBtn = document.querySelector("#obToggleKey");

  apiBaseInput.addEventListener("input", () => { onboardingData.apiBase = apiBaseInput.value; });
  apiKeyInput.addEventListener("input", () => { onboardingData.apiKey = apiKeyInput.value; });
  toggleBtn.addEventListener("click", () => {
    if (apiKeyInput.type === "password") {
      apiKeyInput.type = "text";
      toggleBtn.textContent = "👁 隐藏";
    } else {
      apiKeyInput.type = "password";
      toggleBtn.textContent = "👁 显示";
    }
  });
  onboardingNext.disabled = false;
}

function renderModelTestStep() {
  onboardingBody.innerHTML = `
    <h3>模型选择 & 连接测试</h3>
    <p>选择主力模型和强力模型，然后测试连接。</p>
    <div class="field">
      <span>主力模型（编码、灵敏度分析）</span>
      <input id="obDefaultModel" value="${escapeHtml(onboardingData.defaultModel)}" style="width:100%;height:42px;padding:0 12px;border:1px solid var(--line);border-radius:8px;font-size:14px;color:var(--ink);background:var(--white);outline:0;" />
    </div>
    <div class="field">
      <span>强力模型（分析、建模、写作、评审）</span>
      <input id="obStrongModel" value="${escapeHtml(onboardingData.strongModel)}" style="width:100%;height:42px;padding:0 12px;border:1px solid var(--line);border-radius:8px;font-size:14px;color:var(--ink);background:var(--white);outline:0;" />
    </div>
    <button class="primary-button" type="button" id="obTestBtn">🔍 测试连接</button>
    <div id="obTestResult"></div>
  `;
  const dmInput = document.querySelector("#obDefaultModel");
  const smInput = document.querySelector("#obStrongModel");
  dmInput.addEventListener("input", () => { onboardingData.defaultModel = dmInput.value; });
  smInput.addEventListener("input", () => { onboardingData.strongModel = smInput.value; });

  let testOk = false;
  document.querySelector("#obTestBtn").addEventListener("click", async () => {
    const result = document.querySelector("#obTestResult");
    result.innerHTML = '<p style="color:var(--muted);">测试中...</p>';
    try {
      const [r1, r2] = await Promise.all([
        testLlm(onboardingData.apiBase, onboardingData.apiKey, onboardingData.defaultModel),
        testLlm(onboardingData.apiBase, onboardingData.apiKey, onboardingData.strongModel),
      ]);
      const html = renderTestResult("主力模型", r1) + renderTestResult("强力模型", r2);
      result.innerHTML = html;
      testOk = r1.success && r2.success;
      onboardingNext.disabled = !testOk;
    } catch (error) {
      result.innerHTML = `<div class="test-result fail">测试失败: ${escapeHtml(error.message)}</div>`;
    }
  });
  onboardingNext.disabled = true;
}

async function testLlm(apiBase, apiKey, model) {
  const res = await api("/api/config/test-llm", {
    method: "POST",
    body: JSON.stringify({ apiBase, apiKey, model }),
  });
  return res;
}

function renderTestResult(label, result) {
  const cls = result.success ? "ok" : "fail";
  const text = result.success
    ? `✓ ${label} 连接成功 · ${result.latency_ms}ms`
    : `✗ ${label} 失败: ${escapeHtml(result.error || "未知错误")}`;
  return `<div class="test-result ${cls}">${text}</div>`;
}

async function renderCompleteStep() {
  // 保存配置
  try {
    await api("/api/config", {
      method: "POST",
      body: JSON.stringify({
        apiBase: onboardingData.apiBase,
        apiKey: onboardingData.apiKey,
        defaultModel: onboardingData.defaultModel,
        strongModel: onboardingData.strongModel,
      }),
    });
  } catch (error) {
    showToast(`保存配置失败: ${error.message}`);
  }

  const providerName = onboardingData.provider === "custom" ? "自定义" : onboardingData.provider;
  onboardingBody.innerHTML = `
    <div style="text-align:center;padding:20px 0;">
      <div style="font-size:48px;margin-bottom:12px;">🎉</div>
      <h3 style="color:var(--green);">配置完成！</h3>
      <p>配置已自动保存到 .env 文件</p>
      <div style="margin:16px auto;padding:14px;border:1px solid var(--line);border-radius:8px;background:#fbfcfe;text-align:left;font-size:0.86rem;max-width:400px;">
        <p><strong>端点:</strong> ${escapeHtml(onboardingData.apiBase)}</p>
        <p><strong>主力模型:</strong> ${escapeHtml(onboardingData.defaultModel)}</p>
        <p><strong>强力模型:</strong> ${escapeHtml(onboardingData.strongModel)}</p>
      </div>
    </div>
  `;
}

onboardingPrev?.addEventListener("click", () => {
  if (onboardingStep > 1) {
    onboardingStep--;
    renderOnboardingStep();
  }
});

onboardingNext?.addEventListener("click", () => {
  if (onboardingStep < 5) {
    onboardingStep++;
    renderOnboardingStep();
  } else {
    hideOnboarding();
    showToast("配置完成，欢迎使用 Beacon！");
  }
});
```

- [ ] **Step 3: 在 `frontend/app.js` 末尾添加设置页面逻辑**

```javascript
// ============ 设置页面 ============

let settingsTab = "api";
let settingsConfig = null;

async function loadSettings() {
  try {
    settingsConfig = await api("/api/config");
    renderSettings();
  } catch (error) {
    settingsContent.innerHTML = `<p>加载配置失败: ${escapeHtml(error.message)}</p>`;
  }
}

function renderSettings() {
  if (!settingsConfig) return;
  if (settingsTab === "api") renderSettingsApi();
  else if (settingsTab === "models") renderSettingsModels();
  else if (settingsTab === "rag") renderSettingsRag();
  else if (settingsTab === "advanced") renderSettingsAdvanced();
}

function renderSettingsApi() {
  settingsContent.innerHTML = `
    <div class="field">
      <span>API 端点</span>
      <input id="setApiBase" value="${escapeHtml(settingsConfig.apiBase)}" />
      <span class="field-hint">OpenAI 兼容端点地址</span>
    </div>
    <div class="field">
      <span>API 密钥</span>
      <div style="display:flex;gap:8px;">
        <input id="setApiKey" type="password" value="${escapeHtml(settingsConfig.apiKey)}" style="flex:1;" />
        <button class="ghost-button" type="button" id="setToggleKey">👁 显示</button>
        <button class="primary-button" type="button" id="setTestBtn">🔍 测试</button>
      </div>
      <div id="setTestResult"></div>
    </div>
  `;
  document.querySelector("#setToggleKey")?.addEventListener("click", (e) => {
    const input = document.querySelector("#setApiKey");
    if (input.type === "password") { input.type = "text"; e.target.textContent = "👁 隐藏"; }
    else { input.type = "password"; e.target.textContent = "👁 显示"; }
  });
  document.querySelector("#setTestBtn")?.addEventListener("click", async () => {
    const apiBase = document.querySelector("#setApiBase").value;
    const apiKey = document.querySelector("#setApiKey").value;
    const model = settingsConfig.defaultModel || "test";
    const result = document.querySelector("#setTestResult");
    result.innerHTML = '<p class="field-hint">测试中...</p>';
    try {
      const r = await testLlm(apiBase, apiKey.includes("***") ? "" : apiKey, model);
      result.innerHTML = r.success
        ? `<div class="test-result ok">✓ 连接成功 · ${r.latency_ms}ms</div>`
        : `<div class="test-result fail">✗ ${escapeHtml(r.error)}</div>`;
    } catch (error) {
      result.innerHTML = `<div class="test-result fail">✗ ${escapeHtml(error.message)}</div>`;
    }
  });
}

function renderSettingsModels() {
  settingsContent.innerHTML = `
    <div class="field">
      <span>主力模型（编码、灵敏度分析）</span>
      <input id="setDefaultModel" value="${escapeHtml(settingsConfig.defaultModel)}" />
      <span class="field-hint">用于 routine 节点，格式：provider/model</span>
    </div>
    <div class="field">
      <span>强力模型（分析、建模、写作、评审）</span>
      <input id="setStrongModel" value="${escapeHtml(settingsConfig.strongModel)}" />
      <span class="field-hint">用于核心节点</span>
    </div>
  `;
}

function renderSettingsRag() {
  settingsContent.innerHTML = `
    <div class="toggle-row">
      <div>
        <strong>启用 RAG 知识库</strong>
        <p class="field-hint">启用后注入经典模型模式到提示词</p>
      </div>
      <div class="toggle-switch ${settingsConfig.ragEnabled ? "on" : ""}" id="setRagToggle"></div>
    </div>
    <div class="field">
      <span>Embedding 模型</span>
      <input id="setRagEmbed" value="${escapeHtml(settingsConfig.ragEmbeddingModel)}" />
    </div>
    <div class="field">
      <span>检索 Top-K</span>
      <input id="setRagTopK" type="number" value="${settingsConfig.ragTopK}" />
    </div>
  `;
  document.querySelector("#setRagToggle")?.addEventListener("click", (e) => {
    e.target.classList.toggle("on");
  });
}

function renderSettingsAdvanced() {
  settingsContent.innerHTML = `
    <div class="field">
      <span>模型迭代轮次上限</span>
      <div style="display:flex;align-items:center;gap:12px;">
        <input id="setIterations" type="range" min="1" max="5" value="${settingsConfig.maxModelIterations}" style="flex:1;" />
        <span id="setIterationsVal" style="font-weight:700;color:var(--blue);">${settingsConfig.maxModelIterations}</span>
      </div>
    </div>
    <div class="field">
      <span>LLM 超时（秒）</span>
      <input id="setLlmTimeout" type="number" value="${settingsConfig.llmTimeout}" />
    </div>
    <div class="field">
      <span>前端端口</span>
      <input id="setPort" type="number" value="${settingsConfig.port}" />
      <span class="field-hint">修改后需重启服务生效</span>
    </div>
  `;
  document.querySelector("#setIterations")?.addEventListener("input", (e) => {
    document.querySelector("#setIterationsVal").textContent = e.target.value;
  });
}

function collectSettings() {
  const config = {};
  const apiBase = document.querySelector("#setApiBase");
  const apiKey = document.querySelector("#setApiKey");
  const defaultModel = document.querySelector("#setDefaultModel");
  const strongModel = document.querySelector("#setStrongModel");
  const ragToggle = document.querySelector("#setRagToggle");
  const ragEmbed = document.querySelector("#setRagEmbed");
  const ragTopK = document.querySelector("#setRagTopK");
  const iterations = document.querySelector("#setIterations");
  const llmTimeout = document.querySelector("#setLlmTimeout");
  const port = document.querySelector("#setPort");

  if (apiBase) config.apiBase = apiBase.value;
  if (apiKey && !apiKey.value.includes("***")) config.apiKey = apiKey.value;
  if (defaultModel) config.defaultModel = defaultModel.value;
  if (strongModel) config.strongModel = strongModel.value;
  if (ragToggle) config.ragEnabled = ragToggle.classList.contains("on");
  if (ragEmbed) config.ragEmbeddingModel = ragEmbed.value;
  if (ragTopK) config.ragTopK = Number(ragTopK.value);
  if (iterations) config.maxModelIterations = Number(iterations.value);
  if (llmTimeout) config.llmTimeout = Number(llmTimeout.value);
  if (port) config.port = Number(port.value);
  return config;
}

settingsNavButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    settingsNavButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    settingsTab = btn.dataset.settingsTab;
    renderSettings();
  });
});

settingsSave?.addEventListener("click", async () => {
  try {
    const config = collectSettings();
    await api("/api/config", { method: "POST", body: JSON.stringify(config) });
    showToast("配置已保存");
    await loadSettings();
  } catch (error) {
    showToast(`保存失败: ${error.message}`);
  }
});

settingsReset?.addEventListener("click", async () => {
  if (!confirm("确定要重置为 .env.example 的默认值吗？")) return;
  try {
    // 读取 .env.example 的默认值
    const { readFile } = await import("node:fs/promises");
    // 通过 API 保存默认值
    await api("/api/config", {
      method: "POST",
      body: JSON.stringify({
        apiBase: "http://localhost:20128/v1",
        apiKey: "123456",
        defaultModel: "deepseek-v4-flash-free",
        strongModel: "deepseek-v4-flash-free",
        llmTimeout: 300,
        maxModelIterations: 3,
        ragEnabled: false,
        ragEmbeddingModel: "text-embedding-3-small",
        ragTopK: 4,
        port: 5173,
      }),
    });
    showToast("已重置为默认值");
    await loadSettings();
  } catch (error) {
    showToast(`重置失败: ${error.message}`);
  }
});
```

- [ ] **Step 4: 修改启动逻辑，检查 onboarding 状态**

在 `frontend/app.js` 末尾找到现有的 health 检查（约第 656 行）：

```javascript
api("/api/health")
  .then((health) => showToast(`已连接项目：${health.fixtures} 个示例题`))
  .catch((error) => showToast(`项目 API 未连接：${error.message}`));
```

替换为：

```javascript
api("/api/health")
  .then((health) => {
    if (health.onboarding && health.onboarding.needed) {
      showOnboarding();
    } else {
      showToast(`已连接项目：${health.fixtures} 个示例题`);
    }
  })
  .catch((error) => showToast(`项目 API 未连接：${error.message}`));
```

- [ ] **Step 5: 修改导航逻辑，支持设置页面切换**

在 `frontend/app.js` 中找到 `navLinks.forEach` 行（约第 509 行），在其之前添加设置页面切换逻辑：

```javascript
// 设置页面导航
document.querySelectorAll(".nav-list a").forEach((link) => {
  link.addEventListener("click", (e) => {
    if (link.getAttribute("href") === "#settings") {
      e.preventDefault();
      document.querySelectorAll(".nav-list a").forEach((l) => l.classList.remove("active"));
      link.classList.add("active");
      // 隐藏工作台，显示设置
      document.querySelector("#workspace").style.display = "none";
      settingsPanel.hidden = false;
      loadSettings();
    } else {
      // 显示工作台，隐藏设置
      document.querySelector("#workspace").style.display = "";
      settingsPanel.hidden = true;
    }
  });
});
```

- [ ] **Step 6: Commit**

```bash
git add frontend/app.js
git commit -m "feat: add onboarding wizard + settings page JS logic"
```

---

### Task 11: 增强 start.bat

**Files:**
- Modify: `scripts/start.bat`

- [ ] **Step 1: 替换 `scripts/start.bat` 全部内容**

```batch
@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ============================================
echo   Beacon Math Agent - 一键启动 (Windows)
echo ============================================
echo.

cd /d "%~dp0.."

REM --- 检查 Node.js（唯一硬依赖，没有它无法启动服务器）---
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 Node.js，请先安装：https://nodejs.org
    echo        安装后重新运行此脚本。
    pause
    exit /b 1
)
for /f "delims=" %%v in ('node --version') do set "node_version=%%v"
echo [OK] Node.js %node_version% 已就绪

REM --- 检查 .env ---
if not exist ".env" (
    if exist ".env.example" (
        echo [初始化] 从 .env.example 创建 .env
        copy ".env.example" ".env" >nul
        echo           启动后可通过引导向导配置 API 密钥
    )
)

REM --- 安装 Node.js 依赖 ---
if not exist "node_modules" (
    echo.
    echo [安装] Node.js 依赖...
    call npm install
    if !errorlevel! neq 0 (
        echo [错误] npm install 失败
        pause
        exit /b 1
    )
)

REM --- 检查 uv（不阻断启动，引导向导会处理安装）---
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [提示] 未找到 uv，Python 后端暂不可用
    echo        启动后可在引导向导中一键安装 uv
) else (
    echo [OK] uv 已就绪
    REM --- 安装 Python 依赖 ---
    echo [检查] Python 依赖...
    uv sync 2>nul
    if !errorlevel! neq 0 (
        echo [警告] uv sync 失败，Python 后端可能无法启动
    )
)

REM --- 启动 ---
echo.
echo ============================================
echo   正在启动服务...
echo   浏览器访问 http://localhost:5173
echo   首次使用会自动进入配置引导
echo   按 Ctrl+C 停止
echo ============================================
echo.

REM 等待 2 秒后自动打开浏览器
start "" /b cmd /c "timeout /t 2 >nul && start http://localhost:5173"

call npm start
pause
```

- [ ] **Step 2: Commit**

```bash
git add scripts/start.bat
git commit -m "feat: enhance start.bat - non-blocking uv check + onboarding hint"
```

---

### Task 12: 端到端验证

- [ ] **Step 1: 运行全部测试**

Run: `node --test frontend/server.test.mjs`
Expected: 所有测试 PASS

- [ ] **Step 2: 启动服务器手动验证**

Run: `npm start`

验证项：
1. 打开 `http://localhost:5173`，如果 .env 没有 API 密钥，应自动弹出引导向导
2. 引导向导步骤1应显示 Python/Node/uv 检测结果
3. 点击侧边栏"设置"应进入设置页面
4. 设置页面四个分类切换正常
5. 保存配置后 `.env` 文件应被更新

- [ ] **Step 3: 最终 Commit**

```bash
git add -A
git commit -m "feat: complete onboarding wizard + settings page + start.bat enhancement"
```

---

## 自查

**1. Spec 覆盖：**
- ✓ 引导向导5步：Task 8 (HTML) + Task 9 (CSS) + Task 10 (JS)
- ✓ 设置页面4个分类：Task 8 + Task 10
- ✓ 启动脚本增强：Task 11
- ✓ 环境检测 + winget 安装：Task 3
- ✓ .env 读写：Task 4
- ✓ LLM 连接测试：Task 4
- ✓ 引导状态判断：Task 5
- ✓ providers.json 预设：Task 2
- ✓ server.mjs 路由分发：Task 6
- ✓ health 端点增强：Task 6
- ✓ 错误处理（winget不可用/超时/失败）：Task 3
- ✓ 测试：Task 7

**2. 占位符扫描：** 无 TBD/TODO，所有步骤都有完整代码。✓

**3. 类型一致性：**
- `handleEnvRoutes(request, response, url)` 签名在 Task 3 和 Task 6 一致 ✓
- `handleConfigRoutes(request, response, url)` 签名在 Task 4 和 Task 6 一致 ✓
- `handleOnboardingRoutes(request, response, url)` 签名在 Task 5 和 Task 6 一致 ✓
- `getOnboardingStatus()` 在 Task 5 定义，Task 6 导入使用 ✓
- 前端 API 路径 `/api/env/check`、`/api/config`、`/api/config/test-llm`、`/api/onboarding/status`、`/api/providers` 在前后端一致 ✓
- `testLlm(apiBase, apiKey, model)` 在 Task 10 定义并在多处使用 ✓
