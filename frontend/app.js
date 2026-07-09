const runButton = document.querySelector("#runPipeline");
const importDemoButton = document.querySelector("#importDemo");
const resetPipelineButton = document.querySelector("#resetPipeline");
const openOutputsButton = document.querySelector("#openOutputs");
const currentStage = document.querySelector("#currentStage");
const nodeProgress = document.querySelector("#nodeProgress");
const qualityScore = document.querySelector("#qualityScore");
const outputSummary = document.querySelector("#outputSummary");
const problemTitle = document.querySelector("#problemTitle");
const problemBrief = document.querySelector("#problemBrief");
const problemBadge = document.querySelector("#problemBadge");
const artifactPreview = document.querySelector("#artifactPreview");
const commandPreview = document.querySelector("#commandPreview");
const outputDir = document.querySelector("#outputDir");
const threadId = document.querySelector("#threadId");
const iterationDepth = document.querySelector("#iterationDepth");
const iterationValue = document.querySelector("#iterationValue");
const ragToggle = document.querySelector("#ragToggle");
const hitlToggle = document.querySelector("#hitlToggle");
const knowledgeBadge = document.querySelector("#knowledgeBadge");
const runState = document.querySelector("#runState");
const uploadZone = document.querySelector("#uploadZone");
const problemFile = document.querySelector("#problemFile");
const uploadTitle = document.querySelector("#uploadTitle");
const uploadMeta = document.querySelector("#uploadMeta");
const toast = document.querySelector("#toast");
const tabButtons = document.querySelectorAll("[data-tab]");
const navLinks = document.querySelectorAll(".nav-list a");
const templateButtons = document.querySelectorAll("[data-template]");
const templateHint = document.querySelector("#templateHint");
const pipelineItems = [...document.querySelectorAll("#pipelineList li")];

const stages = ["Analyst", "Blueprint Critic", "Modeler", "Model Critic", "Coder", "Code Consistency", "Sensitivity", "Figure Pipeline", "Writer", "Paper Critic", "Evaluation", "LaTeX"];
const templateHints = {
  default: "适配建模论文、代码、敏感性分析与 LaTeX 编译流程。",
  gmcm: "启用国赛论文封面字段，命令会追加 --template gmcm。",
};

let stageIndex = 6;
let activeTemplate = "default";
let toastTimer;
let currentRunId = null;
let pollTimer = null;
let lastArtifacts = null;
let currentFixturePath = null;

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("show"), 2400);
}

function setPreview(title, body, extra = "") {
  artifactPreview.innerHTML = `<h3>${title}</h3><p>${body}</p>${extra}`;
}

function setRunLogPreview(run) {
  const log = run.log ? run.log.replace(/[<>&]/g, (char) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" })[char]) : "暂无日志。";
  artifactPreview.innerHTML = `
    <h3>运行日志</h3>
    <p>${run.command || "math-agent run"}</p>
    <pre class="log-preview">${log}</pre>
  `;
}

function updatePipeline(mode = "local") {
  pipelineItems.forEach((item, index) => {
    item.classList.toggle("done", index < stageIndex);
    item.classList.toggle("active", index === stageIndex && stageIndex < stages.length);
  });
  const isComplete = stageIndex >= stages.length;
  currentStage.textContent = isComplete ? "Completed" : stages[Math.max(0, Math.min(stageIndex, stages.length - 1))];
  nodeProgress.textContent = `${Math.min(stageIndex + 1, stages.length)} / ${stages.length}`;
  qualityScore.textContent = isComplete ? "92.1" : (82 + Math.max(0, stageIndex) * 1.4).toFixed(1);
  runButton.textContent = mode === "running" ? "运行中" : isComplete ? "重新运行" : "启动流水线";
  runButton.disabled = mode === "running";
  runState.textContent = mode === "running" ? "Running" : isComplete ? "Done" : "Ready";
}

function updateCommand() {
  const parts = [
    "math-agent run",
    "--problem problem.json",
    `--out ${outputDir.value || "runs/ui-latest"}`,
    `--thread ${threadId.value || "default"}`,
  ];
  if (!hitlToggle.checked) parts.push("--no-interrupt");
  if (activeTemplate !== "default") parts.push(`--template ${activeTemplate}`);
  commandPreview.textContent = parts.join(" ");
  outputSummary.textContent = outputDir.value || "runs/ui-latest";
  iterationValue.textContent = iterationDepth.value;
  knowledgeBadge.textContent = ragToggle.checked ? "RAG On" : "RAG Off";
}

function activateNav(hash) {
  navLinks.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === hash));
}

async function loadFixturesIntoProblem() {
  const { fixtures } = await api("/api/fixtures");
  const fixture = fixtures[0];
  if (!fixture) throw new Error("tests/fixtures 中没有可导入的题目。");
  problemTitle.value = fixture.title;
  problemBrief.value = [fixture.background, ...fixture.questions].filter(Boolean).join("\n");
  problemBadge.textContent = fixture.name.split(".").pop().toUpperCase();
  currentFixturePath = fixture.path;
  stageIndex = 0;
  updatePipeline();
  showToast(`已导入 ${fixture.name}`);
}

function renderBlueprint(summary) {
  if (!summary) {
    setPreview("Problem Blueprint", "还没有 Blueprint 数据。先运行流水线，完成后在此查看结构化题目理解。");
    return;
  }
  const bp = summary.problem_blueprint;
  const esc = (s) => String(s || "").replace(/[<>&]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" })[c]);
  let html = '<div class="blueprint-view">';

  // scores row
  const bc = summary.blueprint_critic;
  const mc = summary.model_code_consistency;
  html += '<div class="bp-scores">';
  if (bc) {
    html += `<div class="bp-score-card ${bc.approved ? "pass" : "fail"}"><span>Blueprint</span><strong>${bc.score ?? "?"}/10</strong><small>${bc.approved ? "通过" : "未通过"}</small></div>`;
  }
  if (summary.model_critic) {
    html += `<div class="bp-score-card ${summary.model_critic.approved ? "pass" : "fail"}"><span>Model</span><strong>${summary.model_critic.score ?? "?"}/10</strong><small>${summary.model_critic.approved ? "通过" : "未通过"}</small></div>`;
  }
  if (mc) {
    html += `<div class="bp-score-card ${mc.approved ? "pass" : "fail"}"><span>Code</span><strong>${mc.score ?? "?"}/10</strong><small>${mc.approved ? "通过" : "未通过"}</small></div>`;
  }
  if (summary.paper_critic) {
    html += `<div class="bp-score-card ${summary.paper_critic.approved ? "pass" : "fail"}"><span>Paper</span><strong>${summary.paper_critic.score ?? "?"}/10</strong><small>${summary.paper_critic.approved ? "通过" : "未通过"}</small></div>`;
  }
  html += "</div>";

  // coverage + unresolved
  html += '<div class="bp-meta">';
  html += `<span class="bp-tag">Question Coverage: ${esc(summary.question_coverage)}</span>`;
  html += `<span class="bp-tag ${summary.unresolved_issues > 0 ? "warn" : "ok"}">Unresolved Issues: ${summary.unresolved_issues}</span>`;
  html += "</div>";

  // blueprint details
  if (bp) {
    html += `<h4>核心任务</h4><p>${esc(bp.core_task)}</p>`;
    if (bp.subquestions && bp.subquestions.length) {
      html += "<h4>小问</h4><ul class=\"bp-list\">";
      bp.subquestions.forEach((sq) => {
        html += `<li><strong>[${esc(sq.id)}]</strong> ${esc(sq.original_text)} <em>(${esc(sq.task_type)})</em></li>`;
      });
      html += "</ul>";
    }
    if (bp.decision_variables && bp.decision_variables.length) {
      html += "<h4>决策变量</h4><ul class=\"bp-list\">";
      bp.decision_variables.forEach((v) => {
        html += `<li><code>${esc(v.name)}</code> ${esc(v.meaning)}</li>`;
      });
      html += "</ul>";
    }
    if (bp.objectives && bp.objectives.length) {
      html += "<h4>目标</h4><ul class=\"bp-list\">";
      bp.objectives.forEach((o) => {
        html += `<li>[${esc(o.direction)}] ${esc(o.description)}</li>`;
      });
      html += "</ul>";
    }
    if (bp.constraints && bp.constraints.length) {
      html += "<h4>约束</h4><ul class=\"bp-list\">";
      bp.constraints.forEach((c) => {
        html += `<li>${esc(c.description)} <em>(${esc(c.source)})</em></li>`;
      });
      html += "</ul>";
    }
    if (bp.metrics && bp.metrics.length) {
      html += "<h4>指标</h4><ul class=\"bp-list\">";
      bp.metrics.forEach((m) => {
        html += `<li><code>${esc(m.name)}</code> ${esc(m.meaning)}</li>`;
      });
      html += "</ul>";
    }
  }

  // consistency details
  if (mc && (mc.missing_variables?.length || mc.missing_objectives?.length || mc.missing_constraints?.length || mc.issues?.length)) {
    html += "<h4>模型-代码一致性</h4>";
    if (mc.missing_variables?.length) html += `<p class="bp-warn">缺失变量: ${esc(mc.missing_variables.join(", "))}</p>`;
    if (mc.missing_objectives?.length) html += `<p class="bp-warn">缺失目标: ${esc(mc.missing_objectives.join(", "))}</p>`;
    if (mc.missing_constraints?.length) html += `<p class="bp-warn">缺失约束: ${esc(mc.missing_constraints.join(", "))}</p>`;
    if (mc.issues?.length) {
      html += "<ul class=\"bp-list\">";
      mc.issues.forEach((i) => { html += `<li>${esc(i)}</li>`; });
      html += "</ul>";
    }
  }

  // blueprint critic issues
  if (bc && bc.issues?.length) {
    html += "<h4>Blueprint Critic 意见</h4><ul class=\"bp-list\">";
    bc.issues.forEach((i) => { html += `<li>${esc(i)}</li>`; });
    html += "</ul>";
  }

  html += "</div>";
  artifactPreview.innerHTML = html;
}

async function loadArtifacts(tab = "paper") {
  const payload = await api(`/api/artifacts?out=${encodeURIComponent(outputDir.value || "runs/ui-latest")}`);
  lastArtifacts = payload;
  if (tab === "paper") {
    setPreview(
      payload.paperExcerpt ? "论文预览" : "还没有论文产物",
      payload.paperExcerpt ? payload.paperExcerpt.slice(0, 700) : `未在 ${payload.out} 找到 paper.md。先启动流水线，或检查输出目录。`,
    );
  } else if (tab === "blueprint") {
    renderBlueprint(payload.stateSummary);
  } else if (tab === "trace") {
    setPreview(
      payload.traceSummary ? "Trace 摘要" : "还没有 Trace",
      payload.traceSummary
        ? `thread=${payload.traceSummary.threadId || "-"}，LLM 调用 ${payload.traceSummary.llmCalls || 0} 次，节点 ${payload.traceSummary.nodes || 0} 个。`
        : `未在 ${payload.out} 找到 trace.json。`,
      payload.traceSummary ? `<pre class="log-preview">${JSON.stringify(payload.traceSummary, null, 2)}</pre>` : "",
    );
  } else {
    const fileTags = payload.files.length
      ? `<div class="artifact-list">${payload.files.map((file) => `<span>${file.type === "dir" ? "dir" : "file"} · ${file.name}</span>`).join("")}</div>`
      : `<div class="artifact-list"><span>暂无文件</span></div>`;
    setPreview("输出目录", `当前目录：${payload.out}`, fileTags);
  }
}

async function startProjectRun() {
  updateCommand();
  stageIndex = 0;
  updatePipeline("running");
  setPreview("正在启动 math-agent", "本地服务正在写入 problem.json 并调用项目 CLI，日志会自动刷新。", "<div class=\"paper-lines\"><span></span><span></span><span></span></div>");
  const { run } = await api("/api/run", {
    method: "POST",
    body: JSON.stringify({
      title: problemTitle.value,
      background: problemBrief.value,
      fixturePath: currentFixturePath,
      outputDir: outputDir.value || "runs/ui-latest",
      threadId: threadId.value || "default",
      template: activeTemplate,
      noInterrupt: !hitlToggle.checked,
      ragEnabled: ragToggle.checked,
      force: false,
    }),
  });
  currentRunId = run.id;
  showToast("已启动项目流水线");
  pollCurrentRun();
}

async function pollCurrentRun() {
  if (!currentRunId) return;
  window.clearTimeout(pollTimer);
  try {
    const run = await api(`/api/runs/${encodeURIComponent(currentRunId)}`);
    if (run.status === "running") {
      stageIndex = Math.min(stageIndex + 1, stages.length - 1);
      updatePipeline("running");
      setRunLogPreview(run);
      pollTimer = window.setTimeout(pollCurrentRun, 1800);
      return;
    }
    stageIndex = run.status === "completed" ? stages.length : stageIndex;
    updatePipeline();
    setRunLogPreview(run);
    showToast(run.status === "completed" ? "流水线完成" : "流水线失败，已显示日志");
    if (run.status === "completed") await loadArtifacts("paper");
  } catch (error) {
    updatePipeline();
    showToast(error.message);
  }
}

runButton?.addEventListener("click", () => {
  startProjectRun().catch((error) => {
    updatePipeline();
    setPreview("启动失败", error.message);
    showToast(error.message);
  });
});

resetPipelineButton?.addEventListener("click", () => {
  currentRunId = null;
  window.clearTimeout(pollTimer);
  stageIndex = 0;
  updatePipeline();
  setPreview("流水线状态已刷新", "已回到 Analyst 起点。再次点击启动会调用项目 CLI。");
  showToast("流水线状态已刷新");
});

importDemoButton?.addEventListener("click", () => {
  loadFixturesIntoProblem().catch((error) => showToast(error.message));
});

openOutputsButton?.addEventListener("click", () => {
  document.querySelector("#outputs")?.scrollIntoView({ behavior: "smooth", block: "start" });
  loadArtifacts("files").catch((error) => {
    setPreview("读取输出目录失败", error.message);
    showToast(error.message);
  });
});

templateButtons.forEach((button) => {
  button.addEventListener("click", () => {
    templateButtons.forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    activeTemplate = button.dataset.template;
    templateHint.textContent = templateHints[activeTemplate];
    updateCommand();
    showToast(`已切换到 ${button.textContent} 模板`);
  });
});

tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    tabButtons.forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    loadArtifacts(button.dataset.tab).catch((error) => {
      setPreview("读取产物失败", error.message);
      showToast(error.message);
    });
  });
});

navLinks.forEach((link) => link.addEventListener("click", () => activateNav(link.getAttribute("href"))));
[outputDir, threadId, iterationDepth, ragToggle, hitlToggle].forEach((control) => {
  control?.addEventListener("input", updateCommand);
  control?.addEventListener("change", updateCommand);
});

function loadProblemFile(file) {
  if (!file) return;
  uploadTitle.textContent = file.name;
  uploadMeta.textContent = `${Math.ceil(file.size / 1024)} KB · ${file.type || "本地文件"}`;
  problemBadge.textContent = file.name.split(".").pop()?.toUpperCase() || "FILE";
  currentFixturePath = null;
  if (file.type.includes("json") || file.name.endsWith(".json") || file.name.endsWith(".md") || file.name.endsWith(".txt")) {
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      const text = String(reader.result || "");
      try {
        const data = JSON.parse(text);
        problemTitle.value = data.title || problemTitle.value;
        problemBrief.value = [data.background, ...(data.questions || [])].filter(Boolean).join("\n") || text.slice(0, 720);
      } catch {
        problemBrief.value = text.slice(0, 720) || problemBrief.value;
      }
      showToast("题目文件已读取");
    });
    reader.readAsText(file, "utf-8");
  } else {
    showToast("文件已选择，PDF 内容解析会在后端流水线中处理");
  }
}

uploadZone?.addEventListener("click", () => problemFile.click());
uploadZone?.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    problemFile.click();
  }
});
problemFile?.addEventListener("change", () => loadProblemFile(problemFile.files[0]));
uploadZone?.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadZone.classList.add("dragging");
});
uploadZone?.addEventListener("dragleave", () => uploadZone.classList.remove("dragging"));
uploadZone?.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadZone.classList.remove("dragging");
  loadProblemFile(event.dataTransfer.files[0]);
});

window.addEventListener("hashchange", () => activateNav(window.location.hash || "#workspace"));
activateNav(window.location.hash || "#workspace");
updatePipeline();
updateCommand();
api("/api/health")
  .then((health) => showToast(`已连接项目：${health.fixtures} 个示例题`))
  .catch((error) => showToast(`项目 API 未连接：${error.message}`));


