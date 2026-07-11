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
