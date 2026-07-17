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
  figureModel: "MATH_AGENT_FIGURE_MODEL",
  fallbackModels: "MATH_AGENT_LLM_FALLBACK_MODELS",
  llmTimeout: "MATH_AGENT_LLM_TIMEOUT",
  // §8.3 新增：普通/长文本调用最长等待
  llmAttemptTimeout: "MATH_AGENT_LLM_ATTEMPT_TIMEOUT",
  llmLongAttemptTimeout: "MATH_AGENT_LLM_LONG_ATTEMPT_TIMEOUT",
  maxModelIterations: "MATH_AGENT_MAX_MODEL_ITERATIONS",
  ragEnabled: "MATH_AGENT_RAG_ENABLED",
  ragEmbeddingModel: "MATH_AGENT_RAG_EMBED",
  ragTopK: "MATH_AGENT_RAG_TOPK",
  port: "PORT",
};

/**
 * LiteLLM 原生 provider 前缀：这些前缀会让 LiteLLM 走各自的传输协议
 * （如 ollama 走 OLLAMA_API_BASE），不经过 OPENAI_API_BASE。
 * 不在此列表的前缀（如 ocg）或裸模型名，一律补 openai/ 前缀，
 * 让 LiteLLM 走 OpenAI 兼容协议打 OPENAI_API_BASE（本地 router）。
 */
const LITELLM_NATIVE_PREFIXES = new Set([
  "openai", "azure", "anthropic", "bedrock", "vertex_ai", "gemini",
  "ollama", "cohere", "mistral", "groq", "together_ai", "huggingface",
  "fireworks_ai", "ai21", "nlp_cloud", "anyscale",
]);

/**
 * 规范化模型名，确保 LiteLLM 能识别 provider。
 * - 已带原生 provider 前缀（openai/、ollama/ 等）-> 原样返回
 * - 裸模型名或未知前缀（如 ocg/）-> 补 openai/ 前缀走 OpenAI 兼容协议
 * - 空值 -> 原样返回
 */
function normalizeModelForLitellm(model) {
  if (!model || typeof model !== "string") return model;
  const trimmed = model.trim();
  if (!trimmed) return trimmed;
  const slashIndex = trimmed.indexOf("/");
  if (slashIndex !== -1) {
    const prefix = trimmed.slice(0, slashIndex).toLowerCase();
    if (LITELLM_NATIVE_PREFIXES.has(prefix)) return trimmed;
  }
  return `openai/${trimmed}`;
}

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
      hasApiKey: !!envVars.OPENAI_API_KEY,
      defaultModel: envVars.MATH_AGENT_DEFAULT_MODEL || "",
      strongModel: envVars.MATH_AGENT_STRONG_MODEL || "",
      figureModel: envVars.MATH_AGENT_FIGURE_MODEL || "",
      fallbackModels: envVars.MATH_AGENT_LLM_FALLBACK_MODELS || "",
      llmTimeout: Number(envVars.MATH_AGENT_LLM_TIMEOUT || 300),
      // §8.3：新超时变量（兼容旧 llmTimeout）
      llmAttemptTimeout: Number(envVars.MATH_AGENT_LLM_ATTEMPT_TIMEOUT || envVars.MATH_AGENT_LLM_TIMEOUT || 180),
      llmLongAttemptTimeout: Number(envVars.MATH_AGENT_LLM_LONG_ATTEMPT_TIMEOUT || 300),
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
    if (body.defaultModel !== undefined) updates[FIELD_MAP.defaultModel] = normalizeModelForLitellm(body.defaultModel);
    if (body.strongModel !== undefined) updates[FIELD_MAP.strongModel] = normalizeModelForLitellm(body.strongModel);
    if (body.figureModel !== undefined) updates[FIELD_MAP.figureModel] = normalizeModelForLitellm(body.figureModel);
    if (body.fallbackModels !== undefined) {
      updates[FIELD_MAP.fallbackModels] = String(body.fallbackModels)
        .split(",")
        .map((model) => normalizeModelForLitellm(model))
        .filter(Boolean)
        .join(",");
    }
    if (body.llmTimeout !== undefined) updates[FIELD_MAP.llmTimeout] = String(body.llmTimeout);
    if (body.llmAttemptTimeout !== undefined) {
      updates[FIELD_MAP.llmAttemptTimeout] = String(body.llmAttemptTimeout);
      // 同步旧变量保持兼容
      updates[FIELD_MAP.llmTimeout] = String(body.llmAttemptTimeout);
    }
    if (body.llmLongAttemptTimeout !== undefined) updates[FIELD_MAP.llmLongAttemptTimeout] = String(body.llmLongAttemptTimeout);
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
