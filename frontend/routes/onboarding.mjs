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
