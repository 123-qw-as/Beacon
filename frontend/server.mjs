import "dotenv/config";
import { createServer } from "node:http";
import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import { mkdir, readFile, readdir, stat, writeFile, open } from "node:fs/promises";
import { watch } from "node:fs";
import { extname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { handleEnvRoutes } from "./routes/env.mjs";
import { handleConfigRoutes } from "./routes/config.mjs";
import { handleOnboardingRoutes, getOnboardingStatus } from "./routes/onboarding.mjs";

const root = resolve(fileURLToPath(new URL(".", import.meta.url)));
const projectRoot = resolve(root, "..");
const env = globalThis.process?.env || {};
const requestedPort = Number.parseInt(env.PORT || "5173", 10);
const port = Number.isInteger(requestedPort) && requestedPort >= 1 && requestedPort <= 65_535
  ? requestedPort
  : 5173;
const runs = new Map();

const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
const MAX_UPLOAD_BODY = 200 * 1024 * 1024; // 200MB
const ACCEPTED_SUFFIXES = new Set([".json", ".md", ".txt", ".pdf", ".docx", ".xlsx", ".xls", ".csv"]);

function parseMultipart(buffer, boundary) {
  const parts = [];
  const boundaryBytes = Buffer.from(`--${boundary}`);
  let start = 0;
  while (true) {
    const bStart = buffer.indexOf(boundaryBytes, start);
    if (bStart === -1) break;
    const nextStart = bStart + boundaryBytes.length;
    const nextBoundary = buffer.indexOf(boundaryBytes, nextStart);
    if (nextBoundary === -1) break;
    const partData = buffer.slice(nextStart + 2, nextBoundary - 2); // skip \r\n, trim trailing \r\n
    const headerEnd = partData.indexOf("\r\n\r\n");
    if (headerEnd === -1) break;
    const headerStr = partData.slice(0, headerEnd).toString("utf8");
    const content = partData.slice(headerEnd + 4);
    const nameMatch = headerStr.match(/name="([^"]*)"/);
    const filenameMatch = headerStr.match(/filename="([^"]*)"/);
    parts.push({
      name: nameMatch ? nameMatch[1] : "",
      filename: filenameMatch ? filenameMatch[1] : null,
      content: filenameMatch ? content : content.toString("utf8").trim(),
    });
    start = nextBoundary;
  }
  return parts;
}

async function generateFileMeta(filePath) {
  const python = env.PYTHON || "python";
  const py = spawn(python, ["scripts/extract_file_meta.py", filePath], {
    cwd: projectRoot,
    windowsHide: true,
  });
  let stdout = "", stderr = "";
  py.stdout.on("data", (c) => { stdout += c; });
  py.stderr.on("data", (c) => { stderr += c; });
  return new Promise((resolveP, reject) => {
    py.on("close", (code) => {
      if (code === 0) {
        try { resolveP(JSON.parse(stdout.trim())); }
        catch { reject(new Error("meta script returned invalid JSON")); }
      } else {
        reject(new Error(stderr || "meta script failed"));
      }
    });
    py.on("error", reject);
  });
}

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".png": "image/png",
};

function sendJson(response, status, payload) {
  response.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(payload));
}

function splitCommandLine(commandLine) {
  const parts = [];
  const pattern = /"([^"]*)"|'([^']*)'|([^\s]+)/g;
  for (const match of commandLine.matchAll(pattern)) {
    parts.push(match[1] ?? match[2] ?? match[3]);
  }
  if (parts.length === 0) throw new Error("MATH_AGENT_COMMAND is empty.");
  return parts;
}

function safeProjectPath(inputPath) {
  const target = resolve(projectRoot, inputPath || ".");
  const rel = relative(projectRoot, target);
  if (isAbsolute(rel) || rel === ".." || rel.startsWith(`..${sep}`)) {
    throw new HttpError(403, "Path is outside the project workspace.");
  }
  return target;
}

async function readJsonBody(request) {
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

async function listFixtures() {
  const fixtureDir = safeProjectPath("tests/fixtures");
  const entries = await readdir(fixtureDir, { withFileTypes: true });
  const fixtures = [];
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith(".json")) continue;
    const relPath = `tests/fixtures/${entry.name}`;
    const raw = await readFile(resolve(fixtureDir, entry.name), "utf8");
    const data = JSON.parse(raw);
    fixtures.push({
      path: relPath,
      name: entry.name,
      title: data.title || entry.name,
      background: data.background || "",
      questions: data.questions || [],
    });
  }
  return fixtures;
}

async function listDirectoryFiles(dirPath) {
  try {
    const entries = await readdir(dirPath, { withFileTypes: true });
    return entries.map((entry) => ({ name: entry.name, type: entry.isDirectory() ? "dir" : "file" }));
  } catch {
    return [];
  }
}

function terminateRunProcess(run) {
  if (!run.child || !run.pid) return;
  if (process.platform === "win32") {
    // uv/npm 等包装进程退出后可能遗留 Python 子进程，必须终止整棵进程树。
    const killer = spawn("taskkill", ["/pid", String(run.pid), "/T", "/F"], {
      windowsHide: true,
    });
    killer.on("error", () => { try { run.child.kill(); } catch {} });
  } else {
    try { run.child.kill("SIGTERM"); } catch {}
  }
}

function _notifySseClients(run) {
  if (!run.sseClients || run.sseClients.size === 0) return;
  const msg = `event: status\ndata: ${JSON.stringify({ status: run.status, exitCode: run.exitCode })}\n\n`;
  for (const res of run.sseClients) {
    try { res.write(msg); res.end(); } catch {}
  }
  run.sseClients.clear();
}

async function _spawnResume(run, approve, notes) {
  const commandParts = splitCommandLine(env.MATH_AGENT_COMMAND || "uv run math-agent");
  const command = commandParts[0];
  const args = [...commandParts.slice(1), "resume",
    "--out", run.out, "--thread", run.threadId];
  if (approve) { args.push("--approve"); } else { args.push("--no-approve"); }
  if (notes) { args.push("--notes", notes); }

  const runDir = safeProjectPath(`runs/ui-server/${run.id}`);
  const logPath = resolve(runDir, "resume.log");
  const logStream = createWriteStream(logPath, { flags: "a" });
  logStream.write(`$ ${command} ${args.join(" ")}\n\n`);

  // 重置 run 状态为 running
  run.status = "running";
  run.endedAt = null;
  run.exitCode = null;
  run.stdoutBuffer = "";
  run.sseClients = run.sseClients || new Set();

  const child = spawn(command, args, {
    cwd: projectRoot,
    env: {
      ...env,
      MATH_AGENT_RAG_ENABLED: run.ragEnabled === false ? "0" : "1",
      MATH_AGENT_MAX_MODEL_ITERATIONS: String(run.iterationDepth || 3),
      UV_CACHE_DIR: env.UV_CACHE_DIR || resolve(projectRoot, ".uv-cache"),
    },
    windowsHide: true,
  });

  run.child = child;
  run.pid = child.pid;
  run.logPath = logPath;
  let childSettled = false;
  child.stdout.on("data", (chunk) => {
    run.stdoutBuffer = (run.stdoutBuffer + chunk.toString()).slice(-8192);
  });
  child.stdout.pipe(logStream);
  child.stderr.pipe(logStream);
  child.on("error", (error) => {
    if (childSettled) return;
    childSettled = true;
    run.status = "failed";
    run.endedAt = new Date().toISOString();
    logStream.write(`\n[spawn error] ${error.message}\n`);
    logStream.end();
    _notifySseClients(run);
  });
  child.on("close", (code) => {
    if (childSettled) return;
    childSettled = true;
    if (run.status === "stopped") {
      // 保留用户主动停止状态，不让 close 事件覆盖。
    } else if (code === 0 && run.stdoutBuffer.includes("pipeline rejected at human_review")) {
      run.status = "rejected";
    } else if (code === 0 && run.stdoutBuffer.includes("pipeline paused before human_review")) {
      run.status = "paused";
    } else {
      run.status = code === 0 ? "completed" : "failed";
    }
    run.exitCode = code;
    run.endedAt = new Date().toISOString();
    logStream.write(`\n[exit ${code}] [status ${run.status}]\n`);
    logStream.end();
    _notifySseClients(run);
  });
}

async function handleApi(request, response, url) {
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

  if (request.method === "POST" && url.pathname === "/api/upload") {
    const contentType = request.headers["content-type"] || "";
    const boundaryMatch = contentType.match(/boundary=(.+)/);
    if (!boundaryMatch) {
      sendJson(response, 400, { error: "Missing multipart boundary." });
      return;
    }
    const chunks = [];
    let size = 0;
    for await (const chunk of request) {
      size += chunk.length;
      if (size > MAX_UPLOAD_BODY) {
        sendJson(response, 413, { error: "Upload body exceeds 200MB." });
        return;
      }
      chunks.push(chunk);
    }
    const buffer = Buffer.concat(chunks);
    const parts = parseMultipart(buffer, boundaryMatch[1]);
    const filePart = parts.find((p) => p.filename);
    const purposePart = parts.find((p) => p.name === "purpose");
    if (!filePart) {
      sendJson(response, 400, { error: "No file in upload." });
      return;
    }
    const ext = extname(filePart.filename).toLowerCase();
    if (!ACCEPTED_SUFFIXES.has(ext)) {
      sendJson(response, 415, { error: `Unsupported file type: ${ext}` });
      return;
    }
    if (filePart.content.length > MAX_FILE_SIZE) {
      sendJson(response, 413, { error: "File exceeds 50MB." });
      return;
    }
    const purpose = purposePart ? purposePart.content : "attachment";
    const uploadId = `att-${Date.now()}-${randomUUID().slice(0, 8)}`;
    const uploadDir = safeProjectPath(`runs/ui-server/uploads/${uploadId}`);
    await mkdir(uploadDir, { recursive: true });
    const filePath = resolve(uploadDir, filePart.filename);
    await writeFile(filePath, filePart.content);

    let meta;
    try {
      meta = await generateFileMeta(filePath);
    } catch (e) {
      sendJson(response, 500, { error: `File meta extraction failed: ${e.message}` });
      return;
    }
    const result = {
      id: uploadId,
      filename: filePart.filename,
      fileType: meta.file_type,
      size: filePart.content.length,
      storedPath: relative(projectRoot, filePath).replace(/\\/g, "/"),
      summary: meta.summary,
      text: "",
    };
    if (purpose === "problem" && meta.summary && meta.summary.text_excerpt) {
      result.text = meta.summary.text_excerpt;
    }
    sendJson(response, 200, result);
    return;
  }

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

  if (request.method === "GET" && url.pathname === "/api/fixtures") {
    sendJson(response, 200, { fixtures: await listFixtures() });
    return;
  }

  if (request.method === "GET" && url.pathname === "/api/artifacts") {
    const out = url.searchParams.get("out") || "runs/latest";
    const outDir = safeProjectPath(out);
    const files = await listDirectoryFiles(outDir);
    let paper = "";
    let trace = null;
    let stateSummary = null;
    try {
      paper = await readFile(resolve(outDir, "paper.md"), "utf8");
    } catch {}
    try {
      trace = JSON.parse(await readFile(resolve(outDir, "trace.json"), "utf8"));
    } catch {}
    try {
      stateSummary = JSON.parse(await readFile(resolve(outDir, "state_summary.json"), "utf8"));
    } catch {}
    sendJson(response, 200, {
      out,
      exists: files.length > 0,
      files,
      paperExcerpt: paper.slice(0, 1800),
      traceSummary: trace
        ? {
            threadId: trace.thread_id,
            llmCalls: trace.llm_calls,
            tokens: trace.tokens,
            nodes: Array.isArray(trace.nodes) ? trace.nodes.length : 0,
          }
        : null,
      stateSummary,
    });
    return;
  }

    if (request.method === "POST" && url.pathname.startsWith("/api/runs/") && url.pathname.endsWith("/stop")) {
    const id = decodeURIComponent(url.pathname.split("/").at(-2) || "");
    const run = runs.get(id);
    if (!run) {
      sendJson(response, 404, { error: "Run not found." });
      return;
    }
    if (run.child && run.status === "running") {
      terminateRunProcess(run);
      run.status = "stopped";
      run.endedAt = new Date().toISOString();
    }
    const { child, sseClients, stdoutBuffer, ...safeRun } = run;
    sendJson(response, 200, { run: safeRun });
    return;
  }

  // GET /api/runs/:id/log -- SSE log stream (must be before generic GET /api/runs/:id)
  if (request.method === "GET" && /^\/api\/runs\/[^/]+\/log$/.test(url.pathname)) {
    const id = decodeURIComponent(url.pathname.split("/").at(-2) || "");
    const run = runs.get(id);
    if (!run) {
      sendJson(response, 404, { error: "Run not found." });
      return;
    }
    response.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    });
    run.sseClients = run.sseClients || new Set();
    run.sseClients.add(response);

    // 如果 run 已经结束，立即推送 status 并关闭
    if (run.status !== "running" && run.status !== "paused") {
      const msg = `event: status\ndata: ${JSON.stringify({ status: run.status, exitCode: run.exitCode })}\n\n`;
      response.write(msg);
      response.end();
      run.sseClients.delete(response);
      return;
    }
    if (run.status === "paused") {
      const msg = `event: status\ndata: ${JSON.stringify({ status: "paused" })}\n\n`;
      response.write(msg);
    }

    let offset = 0;
    let reading = false;
    const sendChunk = async () => {
      if (reading) return;
      reading = true;
      try {
        const s = await stat(run.logPath);
        if (s.size > offset) {
          const fh = await open(run.logPath, "r");
          try {
            const buf = Buffer.alloc(s.size - offset);
            await fh.read(buf, 0, buf.length, offset);
            offset = s.size;
            response.write(`data: ${JSON.stringify({ log: buf.toString("utf8") })}\n\n`);
          } finally {
            await fh.close();
          }
        }
      } catch {
      } finally {
        reading = false;
      }
    };
    await sendChunk();
    let watcher;
    try { watcher = watch(run.logPath, () => sendChunk()); } catch {}
    const poll = setInterval(sendChunk, 1000);
    request.on("close", () => {
      if (watcher) watcher.close();
      clearInterval(poll);
      run.sseClients?.delete(response);
    });
    return;
  }

  if (request.method === "GET" && url.pathname.startsWith("/api/runs/")) {
    const id = decodeURIComponent(url.pathname.split("/").pop() || "");
    const run = runs.get(id);
    if (!run) {
      sendJson(response, 404, { error: "Run not found." });
      return;
    }
    let log = "";
    try {
      log = await readFile(run.logPath, "utf8");
    } catch {}
    const { child, sseClients, stdoutBuffer, ...safeRun } = run;
    sendJson(response, 200, { ...safeRun, log: log.slice(-6000) });
    return;
  }

  if (request.method === "POST" && url.pathname === "/api/run") {
    const body = await readJsonBody(request);
    if (body === null || Array.isArray(body) || typeof body !== "object") {
      throw new HttpError(400, "Request body must be a JSON object.");
    }
    for (const key of ["title", "background", "outputDir", "threadId", "fixturePath", "template"]) {
      if (body[key] !== undefined && typeof body[key] !== "string") {
        throw new HttpError(400, `${key} must be a string.`);
      }
    }
    if (body.template && !["default", "gmcm"].includes(body.template)) {
      throw new HttpError(400, "template must be default or gmcm.");
    }
    const iterationDepth = Math.min(5, Math.max(1, Number.parseInt(body.iterationDepth || "3", 10) || 3));
    const runId = `ui-${Date.now()}-${randomUUID().slice(0, 8)}`;
    const out = body.outputDir || "runs/ui-latest";
    const outDir = safeProjectPath(out);
    const conflictingRun = [...runs.values()].find(
      (item) => item.out === out && ["running", "paused"].includes(item.status),
    );
    if (conflictingRun) {
      throw new HttpError(409, `Output directory is already used by run ${conflictingRun.id}.`);
    }
    const runDir = safeProjectPath(`runs/ui-server/${runId}`);
    await mkdir(runDir, { recursive: true });
    await mkdir(outDir, { recursive: true });

    // 附件复制到 run 专属 data 目录
    let dataDir = "";
    let dataFiles = [];
    if (Array.isArray(body.attachments) && body.attachments.length) {
      const dataDirPath = resolve(runDir, "data");
      await mkdir(dataDirPath, { recursive: true });
      for (const att of body.attachments) {
        if (!att.storedPath) continue;
        const src = safeProjectPath(att.storedPath);
        const dst = resolve(dataDirPath, att.filename);
        await writeFile(dst, await readFile(src));
        dataFiles.push({
          filename: att.filename,
          file_type: att.fileType,
          path: att.filename,
          summary: att.summary || {},
        });
      }
      dataDir = dataDirPath;
    }

    const problemPath = resolve(runDir, "problem.json");
    const problem = {
      title: body.title || "Beacon UI Problem",
      background: body.background || "",
      questions: String(body.background || "")
        .split(/\n+/)
        .map((line) => line.trim())
        .filter(Boolean)
        .slice(0, 6),
    };
    if (problem.questions.length === 0) problem.questions = [body.title || "请完成数学建模分析。"];
    if (dataDir) {
      problem.data_dir = dataDir;
      problem.data_files = dataFiles;
    }
    await writeFile(problemPath, JSON.stringify(problem, null, 2), "utf8");

    const commandParts = splitCommandLine(env.MATH_AGENT_COMMAND || "uv run math-agent");
    const command = commandParts[0];
    const problemArg = body.fixturePath ? safeProjectPath(body.fixturePath) : problemPath;
    const args = [...commandParts.slice(1), "run", "--problem", problemArg, "--out", out, "--thread", body.threadId || "default"];
    if (body.noInterrupt) args.push("--no-interrupt");
    if (body.template && body.template !== "default") args.push("--template", body.template);
    if (body.force) args.push("--force");

    const logPath = resolve(runDir, "run.log");
    const logStream = createWriteStream(logPath, { flags: "a" });
    logStream.write(`$ ${command} ${args.join(" ")}\n\n`);

    const run = {
      id: runId,
      status: "running",
      command: `${command} ${args.join(" ")}`,
      out,
      threadId: body.threadId || "default",
      ragEnabled: body.ragEnabled !== false,
      iterationDepth,
      logPath,
      startedAt: new Date().toISOString(),
      endedAt: null,
      exitCode: null,
      stdoutBuffer: "",
      sseClients: new Set(),
    };
    runs.set(runId, run);

    const child = spawn(command, args, {
      cwd: projectRoot,
      env: {
        ...env,
        MATH_AGENT_RAG_ENABLED: body.ragEnabled === false ? "0" : "1",
        MATH_AGENT_MAX_MODEL_ITERATIONS: String(iterationDepth),
        UV_CACHE_DIR: env.UV_CACHE_DIR || resolve(projectRoot, ".uv-cache"),
      },
      windowsHide: true,
    });

    run.child = child;
    run.pid = child.pid;
    let childSettled = false;
    // tap stdout to detect HITL pause marker (CLI exits 0 for both pause and complete)
    child.stdout.on("data", (chunk) => {
      run.stdoutBuffer = (run.stdoutBuffer + chunk.toString()).slice(-8192);
    });
    child.stdout.pipe(logStream);
    child.stderr.pipe(logStream);
    child.on("error", (error) => {
      if (childSettled) return;
      childSettled = true;
      run.status = "failed";
      run.endedAt = new Date().toISOString();
      logStream.write(`\n[spawn error] ${error.message}\n`);
      logStream.end();
      _notifySseClients(run);
    });
    child.on("close", (code) => {
      if (childSettled) return;
      childSettled = true;
      if (run.status !== "stopped") {
        if (code === 0 && run.stdoutBuffer.includes("pipeline paused before human_review")) {
          run.status = "paused";
        } else {
          run.status = code === 0 ? "completed" : "failed";
        }
      }
      run.exitCode = code;
      run.endedAt = new Date().toISOString();
      logStream.write(`\n[exit ${code}] [status ${run.status}]\n`);
      logStream.end();
      _notifySseClients(run);
    });

    const { child: _child, sseClients: _s, stdoutBuffer: _b, ...safeRun } = run;
    sendJson(response, 202, { run: safeRun });
    return;
  }

  // POST /api/runs/:id/resume -- resume a paused HITL run
  if (request.method === "POST" && url.pathname.startsWith("/api/runs/") && url.pathname.endsWith("/resume")) {
    const id = decodeURIComponent(url.pathname.split("/").at(-2) || "");
    const run = runs.get(id);
    if (!run) {
      sendJson(response, 404, { error: "Run not found." });
      return;
    }
    if (run.status !== "paused") {
      sendJson(response, 409, { error: `Run is ${run.status}, not paused.` });
      return;
    }
    const body = await readJsonBody(request);
    await _spawnResume(run, body.approve !== false, body.notes || "");
    const { child, sseClients, stdoutBuffer, ...safeRun } = run;
    sendJson(response, 200, { run: safeRun });
    return;
  }

  sendJson(response, 404, { error: "Unknown API route." });
}

createServer(async (request, response) => {
  const url = new URL(request.url || "/", `http://${request.headers.host || "localhost"}`);

  try {
    if (url.pathname.startsWith("/api/")) {
      await handleApi(request, response, url);
      return;
    }

    const pathname = url.pathname === "/" ? "index.html" : url.pathname.slice(1);
    const filePath = resolve(join(root, pathname));
    const frontendRelative = relative(root, filePath);

    if (isAbsolute(frontendRelative) || frontendRelative === ".." || frontendRelative.startsWith(`..${sep}`)) {
      response.writeHead(403);
      response.end("Forbidden");
      return;
    }

    const body = await readFile(filePath);
    response.writeHead(200, {
      "Content-Type": contentTypes[extname(filePath)] || "application/octet-stream",
    });
    response.end(body);
  } catch (error) {
    if (url.pathname.startsWith("/api/")) {
      sendJson(response, error.status || 500, { error: error.message || "Server error." });
      return;
    }
    response.writeHead(404);
    response.end("Not found");
  }
}).listen(port, "127.0.0.1", () => {
  const apiBase = env.OPENAI_API_BASE || "(not set)";
  const defaultModel = env.MATH_AGENT_DEFAULT_MODEL || "(not set)";
  const ragEnabled = env.MATH_AGENT_RAG_ENABLED === "1" ? "on" : "off";
  console.log("");
  console.log("  ╔══════════════════════════════════════════════╗");
  console.log("  ║   Beacon Math Agent — Web UI ready          ║");
  console.log("  ╠══════════════════════════════════════════════╣");
  console.log(`  ║   Frontend  → http://127.0.0.1:${String(port).padEnd(19)}║`);
  console.log(`  ║   LLM API   → ${apiBase.padEnd(27)}║`);
  console.log(`  ║   Model     → ${defaultModel.padEnd(27)}║`);
  console.log(`  ║   RAG       → ${ragEnabled.padEnd(27)}║`);
  console.log("  ╚══════════════════════════════════════════════╝");
  console.log("");
});




