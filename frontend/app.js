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
const forceToggle = document.querySelector("#forceToggle");
const knowledgeBadge = document.querySelector("#knowledgeBadge");
const runState = document.querySelector("#runState");
const uploadZone = document.querySelector("#uploadZone");
const problemFile = document.querySelector("#problemFile");
const uploadTitle = document.querySelector("#uploadTitle");
const uploadMeta = document.querySelector("#uploadMeta");
const attachmentZone = document.querySelector("#attachmentZone");
const attachmentFile = document.querySelector("#attachmentFile");
const attachmentList = document.querySelector("#attachmentList");
const toast = document.querySelector("#toast");
const tabButtons = document.querySelectorAll("[data-tab]");
const navLinks = document.querySelectorAll(".nav-list a");
const templateButtons = document.querySelectorAll("[data-template]");
const templateHint = document.querySelector("#templateHint");
const pipelineItems = [...document.querySelectorAll("#pipelineList li")];
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
const dashboardSections = document.querySelectorAll(
  "#workspace > .topbar, #workspace > .status-strip, #workspace > .layout-grid",
);

const stages = ["Analyst", "Blueprint Critic", "Modeler", "Model Critic", "Coder", "Code Consistency", "Sensitivity", "Figure Pipeline", "Writer", "Paper Critic", "Table Assembler", "Evaluation", "Human Review", "LaTeX"];
const stageLogNames = ["analyst", "blueprint_critic", "modeler", "model_critic", "coder", "model_code_consistency", "sensitivity", "figure_pipeline", "writer", "paper_critic", "table_assembler", "evaluation", "human_review", "latex"];
const MAX_PROBLEM_TEXT_CHARS = 400_000;
const templateHints = {
  default: "适配建模论文、代码、敏感性分析与 LaTeX 编译流程。",
  gmcm: "启用国赛论文封面字段，命令会追加 --template gmcm。",
};

let stageIndex = -1;
let activeTemplate = "default";
let toastTimer;
let currentRunId = null;
let pollTimer = null;
let logStream = null;
let lastArtifacts = null;
let currentFixturePath = null;
let uploadedAttachments = [];

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

function escapeHtml(value) {
  return String(value ?? "").replace(/[<>&"']/g, (char) => ({
    "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function setPreview(title, body, extra = "") {
  artifactPreview.innerHTML = `<h3>${escapeHtml(title)}</h3><p>${escapeHtml(body)}</p>${extra}`;
}

function setRunLogPreview(run) {
  const log = run.log ? run.log.replace(/[<>&]/g, (char) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" })[char]) : "暂无日志。";
  artifactPreview.innerHTML = `
    <h3>运行日志</h3>
    <p>${escapeHtml(run.command || "math-agent run")}</p>
    <pre class="log-preview">${log}</pre>
  `;
}

function updatePipeline(mode = "local") {
  pipelineItems.forEach((item, index) => {
    item.classList.toggle("done", index < stageIndex);
    item.classList.toggle("active", index === stageIndex && stageIndex < stages.length);
    item.classList.toggle("paused", mode === "paused" && index === stageIndex);
  });
  const isComplete = stageIndex >= stages.length;
  currentStage.textContent = stageIndex < 0
    ? "Ready"
    : isComplete
      ? "Completed"
      : (mode === "paused" ? "Paused" : stages[Math.min(stageIndex, stages.length - 1)]);
  nodeProgress.textContent = `${Math.max(0, Math.min(stageIndex + 1, stages.length))} / ${stages.length}`;
  const actualScore = lastArtifacts?.stateSummary?.evaluation_overall;
  qualityScore.textContent = isComplete && Number.isFinite(Number(actualScore))
    ? Number(actualScore).toFixed(2)
    : "--";
  runButton.textContent = mode === "running" ? "运行中" : isComplete ? "重新运行" : "启动流水线";
  runButton.disabled = mode === "running";
  runState.textContent = mode === "running" ? "Running" : mode === "paused" ? "Paused" : isComplete ? "Done" : "Ready";
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
  if (forceToggle.checked) parts.push("--force");
  commandPreview.textContent = parts.join(" ");
  outputSummary.textContent = outputDir.value || "runs/ui-latest";
  iterationValue.textContent = iterationDepth.value;
  knowledgeBadge.textContent = ragToggle.checked ? "RAG On" : "RAG Off";
}

function activateNav(hash) {
  navLinks.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === hash));
  setSettingsView(hash === "#settings");
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
  const esc = escapeHtml;
  let html = '<div class="blueprint-view">';

  // scores row
  const bc = summary.blueprint_critic;
  const mc = summary.model_code_consistency;
  html += '<div class="bp-scores">';
  if (bc) {
    html += `<div class="bp-score-card ${bc.approved ? "pass" : "fail"}"><span>Blueprint</span><strong>${esc(bc.score ?? "?")}/10</strong><small>${bc.approved ? "通过" : "未通过"}</small></div>`;
  }
  if (summary.model_critic) {
    html += `<div class="bp-score-card ${summary.model_critic.approved ? "pass" : "fail"}"><span>Model</span><strong>${esc(summary.model_critic.score ?? "?")}/10</strong><small>${summary.model_critic.approved ? "通过" : "未通过"}</small></div>`;
  }
  if (mc) {
    html += `<div class="bp-score-card ${mc.approved ? "pass" : "fail"}"><span>Code</span><strong>${esc(mc.score ?? "?")}/10</strong><small>${mc.approved ? "通过" : "未通过"}</small></div>`;
  }
  if (summary.paper_critic) {
    html += `<div class="bp-score-card ${summary.paper_critic.approved ? "pass" : "fail"}"><span>Paper</span><strong>${esc(summary.paper_critic.score ?? "?")}/10</strong><small>${summary.paper_critic.approved ? "通过" : "未通过"}</small></div>`;
  }
  html += "</div>";

  // coverage + unresolved
  html += '<div class="bp-meta">';
  html += `<span class="bp-tag">Question Coverage: ${esc(summary.question_coverage)}</span>`;
  html += `<span class="bp-tag ${summary.unresolved_issues > 0 ? "warn" : "ok"}">Unresolved Issues: ${esc(summary.unresolved_issues)}</span>`;
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
  const actualScore = payload.stateSummary?.evaluation_overall;
  qualityScore.textContent = Number.isFinite(Number(actualScore))
    ? Number(actualScore).toFixed(2)
    : "--";
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
      payload.traceSummary ? `<pre class="log-preview">${escapeHtml(JSON.stringify(payload.traceSummary, null, 2))}</pre>` : "",
    );
  } else {
    const fileTags = payload.files.length
      ? `<div class="artifact-list">${payload.files.map((file) => `<span>${file.type === "dir" ? "dir" : "file"} · ${escapeHtml(file.name)}</span>`).join("")}</div>`
      : `<div class="artifact-list"><span>暂无文件</span></div>`;
    setPreview("输出目录", `当前目录：${payload.out}`, fileTags);
  }
}

async function startProjectRun() {
  lastArtifacts = null;
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
      iterationDepth: Number.parseInt(iterationDepth.value, 10),
      force: forceToggle.checked,
      attachments: uploadedAttachments,
    }),
  });
  currentRunId = run.id;
  showToast("已启动项目流水线");
  startLogStream(currentRunId);
  // 慢速状态轮询：SSE 活跃时仅检测 paused/终态（不写日志），SSE 断开时作为日志 fallback
  pollTimer = window.setTimeout(pollCurrentRun, 2000);
}

async function pollCurrentRun() {
  if (!currentRunId) return;
  window.clearTimeout(pollTimer);
  try {
    const run = await api(`/api/runs/${encodeURIComponent(currentRunId)}`);
    if (run.status === "running") {
      if (!logStream) {
        // SSE 未连接 -> 用 API 轮询作为日志 fallback
        setRunLogPreview(run);
        pollTimer = window.setTimeout(pollCurrentRun, 3000);
      } else {
        // SSE 活跃 -> 不写日志，只做慢速状态探测（SSE 断开时的 safety net）
        pollTimer = window.setTimeout(pollCurrentRun, 5000);
      }
      return;
    }
    if (run.status === "paused") {
      handlePaused(run);
      return;
    }
    handleRunEnd(run);
  } catch (error) {
    updatePipeline();
    showToast(error.message);
  }
}

function startLogStream(runId) {
  // 关闭旧的 SSE 连接
  if (logStream) { try { logStream.close(); } catch {} }
  try {
    logStream = new EventSource(`/api/runs/${encodeURIComponent(runId)}/log`);
  } catch {
    // EventSource 不可用 -> 回退到纯轮询
    return;
  }
  let logBuffer = "";
  logStream.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.log) {
        logBuffer = (logBuffer + data.log).slice(-6000);
        updateLogPreview(logBuffer);
        advanceStageFromLog(data.log);
      }
    } catch {}
  };
  logStream.addEventListener("status", (event) => {
    try {
      const data = JSON.parse(event.data);
      // SSE 推送了终态 -> 拉取最终 run 信息
      if (data.status && data.status !== "running") {
        logStream.close();
        logStream = null;
        pollCurrentRun();
      }
    } catch {}
  });
  logStream.onerror = () => {
    // SSE 断开 -> 回退到轮询
    try { logStream.close(); } catch {}
    logStream = null;
    if (currentRunId) pollTimer = window.setTimeout(pollCurrentRun, 2000);
  };
}

function updateLogPreview(logText) {
  const esc = logText.replace(/[<>&]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" })[c]);
  artifactPreview.innerHTML = `
    <h3>运行日志</h3>
    <pre class="log-preview">${esc}</pre>
  `;
}

function advanceStageFromLog(logChunk) {
  const lower = logChunk.toLowerCase();
  for (let i = stageIndex; i < stages.length; i++) {
    if (lower.includes(`node: ${stageLogNames[i]}`)) {
      stageIndex = i;
      updatePipeline("running");
      break;
    }
  }
}

function handlePaused(run) {
  // Evaluation 完成后暂停在 Human Review。
  stageIndex = stages.indexOf("Human Review");
  if (stageIndex < 0) stageIndex = stages.length - 3;
  updatePipeline("paused");
  setRunLogPreview(run);
  showToast("流水线暂停，等待人工审核");
  showResumeBar(run);
}

function handleRunEnd(run) {
  if (logStream) { try { logStream.close(); } catch {} logStream = null; }
  stageIndex = run.status === "completed" ? stages.length : stageIndex;
  updatePipeline();
  setRunLogPreview(run);
  if (["completed", "failed", "rejected", "stopped"].includes(run.status)) {
    // 终态后再次点击运行即是明确重跑；勾选状态可见，用户仍可手动取消。
    forceToggle.checked = true;
    updateCommand();
  }
  if (run.status === "completed") {
    showToast("流水线完成");
    loadArtifacts("paper").catch(() => {});
  } else if (run.status === "paused") {
    handlePaused(run);
  } else if (run.status === "rejected") {
    showToast("人工审核已拒绝，未生成最终稿");
  } else if (run.status === "stopped") {
    showToast("流水线已停止");
  } else {
    showToast("流水线失败，已显示日志");
  }
}

function showResumeBar(run) {
  const esc = escapeHtml;
  artifactPreview.innerHTML = `
    <h3>等待人工审核</h3>
    <p>流水线已暂停在 human_review 节点。请审核当前结果后选择继续或拒绝。</p>
    <div class="resume-bar">
      <button class="primary-button" type="button" id="resumeApprove">批准并继续</button>
      <button class="ghost-button" type="button" id="resumeReject">拒绝</button>
    </div>
    <pre class="log-preview">${esc(run.log || "")}</pre>
  `;
  document.querySelector("#resumeApprove")?.addEventListener("click", () => resumeRun(true));
  document.querySelector("#resumeReject")?.addEventListener("click", () => resumeRun(false));
}

async function resumeRun(approve) {
  if (!currentRunId) return;
  try {
    const { run } = await api(`/api/runs/${encodeURIComponent(currentRunId)}/resume`, {
      method: "POST",
      body: JSON.stringify({ approve }),
    });
    currentRunId = run.id;
    showToast(approve ? "已批准，继续运行" : "已拒绝，正在安全结束");
    // 关闭旧 SSE，重新开始
    if (logStream) { try { logStream.close(); } catch {} logStream = null; }
    stageIndex = stages.indexOf("Human Review");
    updatePipeline("running");
    setPreview("正在恢复运行", "正在执行 resume 命令，日志会自动刷新。");
    startLogStream(currentRunId);
    pollTimer = window.setTimeout(pollCurrentRun, 2000);
  } catch (error) {
    showToast(`恢复失败: ${error.message}`);
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
  if (logStream) { try { logStream.close(); } catch {} logStream = null; }
  stageIndex = -1;
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

navLinks.forEach((link) => link.addEventListener("click", () => {
  const hash = link.getAttribute("href");
  activateNav(hash);
  if (hash === "#settings") loadSettings();
}));
[problemTitle, problemBrief].forEach((control) => {
  control?.addEventListener("input", () => { currentFixturePath = null; });
});
[outputDir, threadId, iterationDepth, ragToggle, hitlToggle, forceToggle].forEach((control) => {
  control?.addEventListener("input", updateCommand);
  control?.addEventListener("change", updateCommand);
});

async function uploadFile(file, purpose) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("purpose", purpose);
  const response = await fetch("/api/upload", { method: "POST", body: formData });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function renderAttachmentList() {
  attachmentList.innerHTML = uploadedAttachments.map((att, i) => {
    let meta = "";
    if (att.summary?.sheets?.length) {
      const s = att.summary.sheets[0];
      meta = `${s.rows}行×${s.cols}列`;
      if (s.columns?.length) meta += ` · ${s.columns.slice(0, 4).join(", ")}`;
    } else if (att.summary?.text_excerpt) {
      meta = `${Math.ceil(att.summary.text_excerpt.length / 1024)} KB 文本`;
    }
    return `<div class="attachment-item">
      <span class="att-name">${escapeHtml(att.filename)}</span>
      <span class="att-meta">${escapeHtml(att.fileType)} · ${escapeHtml(meta)}</span>
      <button class="att-remove" type="button" data-idx="${i}">×</button>
    </div>`;
  }).join("");
  attachmentList.querySelectorAll(".att-remove").forEach((btn) => {
    btn.addEventListener("click", () => {
      uploadedAttachments.splice(Number(btn.dataset.idx), 1);
      renderAttachmentList();
    });
  });
}

async function loadAttachmentFile(file) {
  if (!file) return;
  showToast(`正在上传 ${file.name}...`);
  try {
    const result = await uploadFile(file, "attachment");
    uploadedAttachments.push(result);
    renderAttachmentList();
    showToast(`${file.name} 已上传`);
  } catch (error) {
    showToast(`上传失败：${error.message}`);
  }
}

function loadProblemFile(file) {
  if (!file) return;
  uploadTitle.textContent = file.name;
  uploadMeta.textContent = `${Math.ceil(file.size / 1024)} KB · ${file.type || "本地文件"}`;
  problemBadge.textContent = file.name.split(".").pop()?.toUpperCase() || "FILE";
  currentFixturePath = null;
  const ext = file.name.split(".").pop()?.toLowerCase();
  if (ext === "json" || ext === "md" || ext === "txt") {
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      const text = String(reader.result || "");
      try {
        const data = JSON.parse(text);
        const questions = Array.isArray(data.questions)
          ? data.questions.filter((item) => typeof item === "string")
          : [];
        if (typeof data.title === "string" && data.title) problemTitle.value = data.title;
        const background = typeof data.background === "string" ? data.background : "";
        problemBrief.value = [background, ...questions].filter(Boolean).join("\n") || text.slice(0, MAX_PROBLEM_TEXT_CHARS);
      } catch {
        problemBrief.value = text.slice(0, MAX_PROBLEM_TEXT_CHARS) || problemBrief.value;
      }
      showToast("题目文件已读取");
    });
    reader.readAsText(file, "utf-8");
  } else if (ext === "pdf" || ext === "docx") {
    showToast(`正在提取 ${file.name} 文本...`);
    uploadFile(file, "problem").then((result) => {
      if (result.text) {
        problemBrief.value = result.text.slice(0, MAX_PROBLEM_TEXT_CHARS);
        showToast(`${file.name} 文本已提取`);
      } else {
        showToast("未能从文件中提取文本");
      }
    }).catch((error) => {
      showToast(`提取失败：${error.message}`);
    });
  } else {
    showToast("当前仅支持 JSON、Markdown、TXT、PDF、Word 文件");
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

attachmentZone?.addEventListener("click", () => attachmentFile.click());
attachmentZone?.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    attachmentFile.click();
  }
});
attachmentFile?.addEventListener("change", () => {
  for (const file of attachmentFile.files) {
    loadAttachmentFile(file);
  }
  attachmentFile.value = "";
});
attachmentZone?.addEventListener("dragover", (event) => {
  event.preventDefault();
  attachmentZone.classList.add("dragging");
});
attachmentZone?.addEventListener("dragleave", () => attachmentZone.classList.remove("dragging"));
attachmentZone?.addEventListener("drop", (event) => {
  event.preventDefault();
  attachmentZone.classList.remove("dragging");
  for (const file of event.dataTransfer.files) {
    loadAttachmentFile(file);
  }
});

window.addEventListener("hashchange", () => {
  const hash = window.location.hash || "#workspace";
  activateNav(hash);
  if (hash === "#settings") loadSettings();
});
activateNav(window.location.hash || "#workspace");
updatePipeline();
updateCommand();
api("/api/health")
  .then((health) => {
    if (health.onboarding && health.onboarding.needed) {
      showOnboarding();
    } else {
      showToast(`已连接项目：${health.fixtures} 个示例题`);
    }
  })
  .catch((error) => showToast(`项目 API 未连接：${error.message}`));

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
  const toolName = name === "Python" ? "python" : name === "Node.js" ? "node" : name === "uv" ? "uv" : "";
  const status = ok
    ? `<span class="env-status-ok">✓ ${escapeHtml(info.version)}</span>`
    : `<span class="env-status-missing">✗ ${info.installed ? escapeHtml(info.version) + " 版本过低" : "未检测到"}</span>`;
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
          `<p style="margin-top:12px;color:var(--rose);">手动安装: <a href="${escapeHtml(result.downloadUrl)}" target="_blank">${escapeHtml(result.downloadUrl)}</a></p>`);
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
      onboardingNext.disabled = !(r1.success && r2.success);
    } catch (error) {
      result.innerHTML = `<div class="test-result fail">测试失败: ${escapeHtml(error.message)}</div>`;
    }
  });
  onboardingNext.disabled = true;
}

async function testLlm(apiBase, apiKey, model) {
  return await api("/api/config/test-llm", {
    method: "POST",
    body: JSON.stringify({ apiBase, apiKey, model }),
  });
}

function renderTestResult(label, result) {
  const cls = result.success ? "ok" : "fail";
  const text = result.success
    ? `✓ ${escapeHtml(label)} 连接成功 · ${result.latency_ms}ms`
    : `✗ ${escapeHtml(label)} 失败: ${escapeHtml(result.error || "未知错误")}`;
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
    e.currentTarget.classList.toggle("on");
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
  if (!confirm("确定要重置为默认值吗？")) return;
  try {
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

// 设置页面导航
function setSettingsView(isSettingsView) {
  dashboardSections.forEach((section) => {
    section.hidden = isSettingsView;
  });
  settingsPanel.hidden = !isSettingsView;
}

if (window.location.hash === "#settings") loadSettings();

