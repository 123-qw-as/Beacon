# PaperCritic 注入 stdout — Implementation Plan

> **历史方案记录（非现行事实源）**：路径、状态、命令和待办保留当时上下文；当前实现与限制见 [`../../README.md`](../../README.md)。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 paper_critic 看到 coder 的真实 stdout，**用语义判断**正文中的关键数字是否与代码运行结果相符；不做正则字面对照（避免合理四舍五入被误判为编造）。

**Architecture:** 在 paper_critic_node 取最后一个 `success=True` 的 code_artifact 的 stdout，传给 build_prompt 注入"事实源"区块；SYSTEM 加一句"对比正文数字与 stdout，明显不符的列入 issues + approved=False"。**LLM 用语义判断，不要它做字面匹配**。

**Tech Stack:** 不引入新依赖。改 1 个 node + 1 个 prompt + 1 个测试文件。

---

## eval_v5 实测背景（已验证）

```
attempt_1 stdout 真实数字：
  优化成功，目标总成本 = 52.7174
  扰动 +20% → 目标成本 53.7718

paper.md 摘要：
  最优目标成本为52.6（放缩因子22.12，对应实际成本718）

paper_critic 报告（runs/eval_v5/checkpoints.sqlite, target='paper'）：
  score=7 approved=False
  issues 里没有"22.12 是编造"的指控——critic 没看 stdout，只能怀疑
```

52.6 vs 52.7174 是合理四舍五入，**不算编造**。22.12 / 718 是 writer 第二轮"为回应 critic 而编"——stdout 里没有，**这才是编造**。

设计目标：让 critic 自己用语义把 22.12 这种**明显**编造拎出来。不强求 critic 替我做"小数容差匹配"。

---

## 文件结构

```
src/math_agent/
├── nodes/paper_critic.py            # MODIFY: 取 last success=True stdout
└── prompts/paper_critic.py          # MODIFY: build_prompt +code_stdout 形参
                                     #         SYSTEM +一句"对比数字"
tests/nodes/test_paper_critic.py     # MODIFY: 既有测试加 code_artifacts；+1 条新测试
```

---

## Phase 1：prompt 接 stdout

### Task 1.1：写新测试

**Files:**
- Modify: `tests/nodes/test_paper_critic.py`

- [ ] **Step 1: 在 `tests/nodes/test_paper_critic.py` 末尾追加**

```python
from math_agent.state import CodeArtifact
from math_agent.prompts.paper_critic import build_prompt as _build_paper_critic_prompt


def _paper_with_numbers():
    return PaperSections(
        abstract="目标成本52.6（放缩因子22.12，对应实际成本718）。",
        problem_restatement="x"*200, assumptions="x"*200, notation="x"*200,
        model_section="x"*200, solution="x"*200, sensitivity="x"*200,
        conclusion="x"*200, references="-",
    )


def test_prompt_includes_code_stdout_block():
    """build_prompt 第 4 个形参 code_stdout：注入 stdout 文本块。"""
    p = build_prompt_with_stdout()  # helper below
    real_stdout = "优化成功，目标总成本 = 52.7174\n扰动 +20% → 目标成本 53.7718"
    prompt = _build_paper_critic_prompt(_paper_with_numbers(), 0, 0, real_stdout)
    assert "52.7174" in prompt
    assert "53.7718" in prompt
    assert "代码运行" in prompt or "stdout" in prompt.lower()


def test_prompt_omits_stdout_block_when_empty():
    """没有 success=True code_artifact 时不渲染 stdout 区块（避免噪声）。"""
    prompt = _build_paper_critic_prompt(_paper_with_numbers(), 0, 0, "")
    # 区块标题不出现，省去 LLM 处理"空 stdout"的负担
    assert "代码运行真实输出" not in prompt


def test_paper_critic_node_passes_last_successful_stdout(mocker):
    """节点：从 state.code_artifacts 取最后一个 success=True 的 stdout 传给 build_prompt。"""
    captured = {}

    def _capture(prompt, **kw):
        captured["prompt"] = prompt
        return CriticReport(target="paper", score=7, issues=[], suggestions=[], approved=False)

    mocker.patch("math_agent.nodes.paper_critic.complete", side_effect=_capture)
    s = MathModelingState(problem="p")
    s.paper = _paper_with_numbers()
    # 两个 artifact：旧的失败 + 新的成功；critic 只该看新的
    s.code_artifacts.append(CodeArtifact(purpose="x", code="...", success=False,
                                          stdout="OLD_FAILED", stderr="error"))
    s.code_artifacts.append(CodeArtifact(purpose="y", code="...", success=True,
                                          stdout="NEW_TRUE_VALUE=52.7174", stderr=""))
    delta = paper_critic_node(s)
    assert delta["critic_reports"][0].target == "paper"
    assert "NEW_TRUE_VALUE=52.7174" in captured["prompt"]
    assert "OLD_FAILED" not in captured["prompt"]


def build_prompt_with_stdout():
    """文档化：本测试期望 build_prompt 接受 code_stdout 形参。"""
    return None
```

(Step 1 末尾的 `build_prompt_with_stdout` 是占位文档函数，不会被执行——只是说明本测试假设新签名)

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/nodes/test_paper_critic.py -v -k "stdout"`
Expected: 3 条 FAIL（`build_prompt` 不接受第 4 个形参；节点不传 stdout）

---

### Task 1.2：改 prompt build_prompt

**Files:**
- Modify: `src/math_agent/prompts/paper_critic.py`

- [ ] **Step 1: 改写 `src/math_agent/prompts/paper_critic.py`**

把整个文件替换为：

```python
"""PaperCritic：对组装好的论文初稿做整体评审，输出 CriticReport(target='paper')。"""

SYSTEM = (
    "你是国赛资深评委。请审阅一份建模论文初稿。要点："
    "（1）摘要是否凸显方法和结论；（2）假设是否被正文承接；"
    "（3）模型与求解是否一致、可复现；（4）是否有敏感性分析；"
    "（5）图表是否被正文引用并解读；（6）整体行文是否专业。"
    "总评 0-10，>=8 approved。"
    "\n\n"
    "**关键事实核查**：若下文给出『代码运行真实输出』区块，请把它当作唯一可靠的数字事实源。"
    "用语义判断正文中的关键定量结论（成本、占比、敏感度幅度、性能指标等）是否与 stdout 相符。"
    "明显与 stdout 不符的数字（如 stdout 显示 52.7174 但正文写 718）视为编造，"
    "把它逐条列入 issues 并把 approved 设为 False。"
    "合理四舍五入（如 52.7174→52.6、53.7718→53.8）不算编造，不要因此扣分。"
)


def build_prompt(paper, n_figures, n_sensitivity, code_stdout: str = ""):
    sections = {
        "abstract": paper.abstract, "problem_restatement": paper.problem_restatement,
        "assumptions": paper.assumptions, "notation": paper.notation,
        "model_section": paper.model_section, "solution": paper.solution,
        "sensitivity": paper.sensitivity, "conclusion": paper.conclusion,
    }
    body = "\n\n".join(f"## {k}\n{v[:1000]}" for k, v in sections.items())
    stdout_block = ""
    if code_stdout.strip():
        stdout_block = (
            f"\n# 代码运行真实输出（事实源；用于核对正文数字）\n"
            f"```\n{code_stdout[:4000]}\n```\n"
        )
    return (
        f"# 章节素材\n{body}\n\n"
        f"# 客观信号\n- 图表数：{n_figures}\n- 敏感性 run 数：{n_sensitivity}\n"
        f"{stdout_block}\n"
        f"请输出 JSON：{{\"target\":\"paper\",\"score\":int,\"issues\":[str],"
        f"\"suggestions\":[str],\"approved\":bool}}。"
    )
```

注意几点：
- `code_stdout` 默认空字符串 → 既有调用方（节点未改前）继续可用，单测里 `build_prompt(p, 0, 0)` 不会因此挂掉。
- 截断 4000 字符上限：避免大 stdout 把 prompt 撑爆。eval_v5 attempt_1 stdout 是 914 字符，4000 留足空间。
- "合理四舍五入不算编造" 是直接告诉 LLM 的容差规则——比让节点做正则容差靠谱。

- [ ] **Step 2: 跑 prompt 相关测试**

Run: `pytest tests/nodes/test_paper_critic.py -v -k "stdout"`
Expected: 前两条 PASS（prompt 改完）；第三条仍 FAIL（节点没改）。

---

### Task 1.3：改 paper_critic_node 传 stdout

**Files:**
- Modify: `src/math_agent/nodes/paper_critic.py`

- [ ] **Step 1: 改写整个文件**

```python
from math_agent.config import MODEL_ROUTING
from math_agent.llm import complete
from math_agent.prompts.paper_critic import SYSTEM, build_prompt
from math_agent.state import CriticReport, MathModelingState


def _last_successful_stdout(state: MathModelingState) -> str:
    """最后一个 success=True 的 code_artifact.stdout。没有则空串。"""
    for art in reversed(state.code_artifacts):
        if art.success:
            return art.stdout
    return ""


def paper_critic_node(state: MathModelingState) -> dict:
    p = state.paper
    if not any([p.abstract, p.model_section, p.solution]):
        return {"errors": ["paper_critic: 论文初稿为空，跳过整体评审"]}

    out: CriticReport = complete(
        build_prompt(p, len(state.figures), len(state.sensitivity_runs),
                     _last_successful_stdout(state)),
        schema=CriticReport, system=SYSTEM,
        model=MODEL_ROUTING["paper_critic"],
    )
    out.target = "paper"
    return {"critic_reports": [out]}
```

- [ ] **Step 2: 跑全部 paper_critic 测试**

Run: `pytest tests/nodes/test_paper_critic.py -v`
Expected: 5 passed（原有 2 + 新增 3）

- [ ] **Step 3: 全量回归**

Run: `pytest -q`
Expected: 78 passed, 1 skipped（baseline 75 passed + 1 skipped + 新增 3）

- [ ] **Step 4: Commit**

```bash
git add src/math_agent/nodes/paper_critic.py src/math_agent/prompts/paper_critic.py tests/nodes/test_paper_critic.py
git commit -m "feat(paper_critic): inject latest success=True stdout for number fact-check

- prompts/paper_critic: build_prompt +code_stdout (default empty); SYSTEM
  adds a rule: numbers in body must match stdout semantically; rounding OK
- nodes/paper_critic: pick stdout from latest success=True code_artifact
- 4000-char cap on stdout block to bound prompt size
- tests cover: prompt contains stdout, prompt omits block when empty,
  node picks the right artifact (last success=True), legacy tests still pass"
```

---

## Phase 2：v6.1 端到端验收（手动，非门禁）

⚠️ 这一段**不是 pytest**，是手动跑一次 LLM 实测。

### Task 2.1：清理 + 跑

- [ ] **Step 1: 在新 Git Bash 窗口里跑**

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
rm -rf runs/eval_v6_1 runs/eval_v6_1.log
python -m math_agent.cli run \
  --problem tests/fixtures/sample_problem.json \
  --out runs/eval_v6_1 \
  --thread v6_1 \
  --no-interrupt 2>&1 | tee runs/eval_v6_1.log
```

⚠️ 必须**新开 Git Bash**，不能从 ZCode harness 起后台任务——ZCode 子进程不继承新 PATH，且 600s 超时跑不完一次 17 分钟的端到端。

预计 15-25 分钟。

### Task 2.2：验收点

- [ ] **Step 1: paper_critic 报告里至少有一条"数字编造"指控**

```bash
python -c "
from langgraph.checkpoint.sqlite import SqliteSaver
from math_agent.state import CriticReport
with SqliteSaver.from_conn_string('runs/eval_v6_1/checkpoints.sqlite') as s:
    snap = s.get({'configurable': {'thread_id': 'v6_1'}})
ch = snap['channel_values']
pcs = [r for r in ch['critic_reports'] if isinstance(r, CriticReport) and r.target == 'paper']
for r in pcs:
    print('--- paper_critic ---')
    print(' score=', r.score, ' approved=', r.approved)
    for i in r.issues: print('  issue:', i[:150])
"
```

期望：`issues` 中至少有一条**字面**或**语义**点名某个数字与 stdout 不符。例如：
- ✅ "摘要中『放缩因子 22.12』在代码 stdout 中未出现"
- ✅ "正文写『实际成本 718』，但 stdout 总成本是 52.7174，数值不一致"
- ❌ （只说"放缩因子推导缺失"——这种含糊指控不算）

- [ ] **Step 2: 合理四舍五入没被误判**

肉眼对照 `runs/eval_v6_1/paper.md` 与 `runs/eval_v6_1/attempt_*/` 真实 stdout。
- paper 写 52.6 而 stdout 是 52.7174：critic **不应**把它列入 issues。
- 如果误判了，Fix 失败——回到 prompt 调"合理四舍五入"那句话不够强。

- [ ] **Step 3: 测试套件回归没破**

```bash
pytest -q
```

Expected: 78 passed, 1 skipped。

---

## 已知非本计划范围

- writer 输出 LaTeX 兼容（数学符号 `$...$`）：见 `2026-06-29-writer-latex-compat.md`
- paper_critic 拒绝多模态：critic 看不到图，只能从 caption/analysis 推；本计划不解决
- paper_critic 拒绝时不二次重写：writer↔critic 闭环已存在但只跑 2 轮 cap；不调

---

## 自我审查

**1. Spec 覆盖：**
- 注入 stdout → Task 1.2/1.3 ✓
- 语义判断不做字面匹配 → SYSTEM 文案 ✓
- 容差由 LLM 而非节点判断 → 「合理四舍五入不算编造」一句 ✓
- v6.1 验收点 → Phase 2 ✓

**2. Placeholder 扫描：** 无 TBD/TODO；`build_prompt_with_stdout()` 是文档化占位（Task 1.1 Step 1 末尾说明了用途），不会被执行。

**3. 类型一致性：**
- `build_prompt(paper, n_figures, n_sensitivity, code_stdout: str = "")` ↔ 节点调用 `build_prompt(p, len(state.figures), len(state.sensitivity_runs), _last_successful_stdout(state))`：4 个位置参数对齐 ✓
- `code_stdout: str = ""` 默认值兜底既有调用 ↔ Task 1.1 第二条测试 `build_prompt(_paper, 0, 0, "")` 与 omit 区块一致 ✓
- `_last_successful_stdout` 返回类型 `str` ↔ build_prompt 形参 `str` ✓
- 现有 `test_paper_critic_appends_report` 没 code_artifacts，依赖默认空 stdout：兼容 ✓
