import "dotenv/config";
import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import { mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import { extname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL(".", import.meta.url)));
const projectRoot = resolve(root, "..");
const env = globalThis.process?.env || {};
const port = Number.parseInt(env.PORT || "5173", 10);
const runs = new Map();

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

function safeProjectPath(inputPath) {
  const target = resolve(projectRoot, inputPath || ".");
  const rel = relative(projectRoot, target);
  if (rel.startsWith("..") || rel === ".." || rel.includes(`..${join("a", "b").slice(1, 2)}`)) {
    throw new Error("Path is outside the project workspace.");
  }
  return target;
}

async function readJsonBody(request) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  const text = Buffer.concat(chunks).toString("utf8");
  return text ? JSON.parse(text) : {};
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

async function handleApi(request, response, url) {
  if (request.method === "GET" && url.pathname === "/api/health") {
    const fixtures = await listFixtures().catch(() => []);
    sendJson(response, 200, {
      ok: true,
      projectRoot,
      mathAgentCommand: env.MATH_AGENT_COMMAND || "uv run math-agent",
      fixtures: fixtures.length,
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
    try {
      paper = await readFile(resolve(outDir, "paper.md"), "utf8");
    } catch {}
    try {
      trace = JSON.parse(await readFile(resolve(outDir, "trace.json"), "utf8"));
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
      run.child.kill();
      run.status = "stopped";
      run.endedAt = new Date().toISOString();
    }
    const { child, ...safeRun } = run;
    sendJson(response, 200, { run: safeRun });
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
    const { child, ...safeRun } = run;
    sendJson(response, 200, { ...safeRun, log: log.slice(-6000) });
    return;
  }

  if (request.method === "POST" && url.pathname === "/api/run") {
    const body = await readJsonBody(request);
    const runId = `ui-${Date.now()}`;
    const out = body.outputDir || "runs/ui-latest";
    const outDir = safeProjectPath(out);
    const runDir = safeProjectPath(`runs/ui-server/${runId}`);
    await mkdir(runDir, { recursive: true });
    await mkdir(outDir, { recursive: true });

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
    await writeFile(problemPath, JSON.stringify(problem, null, 2), "utf8");

        const commandParts = (env.MATH_AGENT_COMMAND || "uv run math-agent").split(/\s+/).filter(Boolean);
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
      logPath,
      startedAt: new Date().toISOString(),
      endedAt: null,
      exitCode: null,
    };
    runs.set(runId, run);

    const child = spawn(command, args, {
      cwd: projectRoot,
      env: {
        ...env,
        MATH_AGENT_RAG_ENABLED: body.ragEnabled === false ? "0" : "1",
        UV_CACHE_DIR: env.UV_CACHE_DIR || resolve(projectRoot, ".uv-cache"),
      },
      windowsHide: true,
    });

    run.child = child;
    run.pid = child.pid;
    child.stdout.pipe(logStream);
    child.stderr.pipe(logStream);
    child.on("error", (error) => {
      run.status = "failed";
      run.endedAt = new Date().toISOString();
      logStream.write(`\n[spawn error] ${error.message}\n`);
      logStream.end();
    });
    child.on("close", (code) => {
      if (run.status !== "stopped") run.status = code === 0 ? "completed" : "failed";
      run.exitCode = code;
      run.endedAt = new Date().toISOString();
      logStream.write(`\n[exit ${code}]\n`);
      logStream.end();
    });

    const { child: _child, ...safeRun } = run;
    sendJson(response, 202, { run: safeRun });
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

    if (!filePath.startsWith(root)) {
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
      sendJson(response, 500, { error: error.message || "Server error." });
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




