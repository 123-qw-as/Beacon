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
