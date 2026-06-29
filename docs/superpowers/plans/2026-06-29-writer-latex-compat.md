# Writer LaTeX 兼容 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 writer 产出的 `paper.md` 在被 latex_node 渲成 `paper.tex` 后，**真正能用 xelatex 编出 PDF**——目前 writer 输出大量裸 unicode 数学符号（`α_i`、`σ_d`）和裸 LaTeX 下标（`s_i^0`、`\hat{d}_i`），text mode 下 `_` 不合法、`α` 不在默认 roman font 里。

**Architecture:** 双保险根治：
1. **prompt 治源**：`writer_prompt.md.j2` IRON RULES 加一条「数学符号必须 `$...$` 包裹」。
2. **latex_node 兜底**：`paper.tex.j2` 模板用 `\usepackage[utf-8]{inputenc}` 不够（xelatex 不靠它），而是在渲染时**自动把孤立的 unicode 数学字符**（α/β/σ/μ/Φ/π/≥/≤/±/∑ 等）替换为 `$\alpha$`/`$\beta$` 等 inline math。这是**确定性**的，不依赖 LLM 自律。

**Tech Stack:** 不引入新依赖。改 1 个 writer prompt + 1 个 latex_node helper + 模板加 1 个 fontspec 包（兜底字体）。

---

## eval_v5 实测证据（已验证）

```
runs/eval_v5/paper.tex line 17 真实内容：
  ...在置信水平1-α_i下满足借还需求...

xelatex 报错：
  Missing character: There is no α in font [lmroman10-regular]
  ! Missing $ inserted.  l.17 ...1-α_i...
```

两类问题：
1. **unicode 数学符号**（α/σ/μ）：lmroman 字体不含 → 字符消失 + LaTeX 把它当变量名时下标 `_` 报错
2. **裸 LaTeX 下标**（`s_i^0`、`\hat{d}_i`）：在 text mode 下 `_` 是错误

**两类问题用两种修法**：
- unicode：latex_node 在 render 时自动包 `$...$`，**完全确定性**
- 裸 LaTeX：prompt 教 writer 自己包 `$...$`，**靠 LLM 自律**

如果 LLM 漏了一两处裸 LaTeX，prompt 没治住 → tex 编译仍会挂在 `s_i^0` 这种 token 上。**这是已知的 prompt-only 风险**，但 latex_node 没法用正则区分"`s_i^0` 是数学"和"`abc_def` 是普通变量名"——所以这层只能依赖 prompt + LLM。

---

## 文件结构

```
src/math_agent/
├── nodes/latex.py                       # MODIFY: +_wrap_unicode_math() 确定性兜底
├── templates/paper.tex.j2               # MODIFY: +fontspec 包，避免 α 这类符号字体缺失
├── templates/writer_prompt.md.j2        # MODIFY: IRON RULES +rule 4（LaTeX 兼容）
└── prompts/writer.py                    # 不动（仅传参，不增形参）

tests/nodes/test_latex.py                # MODIFY: +2 条 unicode math 包裹测试
tests/nodes/test_writer.py               # MODIFY: +1 条 prompt 含 LaTeX 兼容规则测试
```

---

## Phase 1：latex_node 确定性兜底（unicode → inline math）

### Task 1.1：写测试

**Files:**
- Modify: `tests/nodes/test_latex.py`

- [ ] **Step 1: 在 `tests/nodes/test_latex.py` 末尾追加**

```python
from math_agent.nodes.latex import _wrap_unicode_math


def test_wrap_unicode_math_handles_greek_letters():
    """孤立 unicode 希腊字母被包成 inline math。"""
    s = "在置信水平 α 与 σ 之间，参数 μ 是常数。"
    out = _wrap_unicode_math(s)
    assert "$\\alpha$" in out
    assert "$\\sigma$" in out
    assert "$\\mu$" in out
    # 中文/普通字母不动
    assert "在置信水平" in out
    assert "之间" in out


def test_wrap_unicode_math_handles_relational_operators():
    """≥ ≤ ± 等关系符号被包。"""
    out = _wrap_unicode_math("x ≥ 0 且 y ≤ 1，误差 ±5%")
    assert "$\\geq$" in out
    assert "$\\leq$" in out
    assert "$\\pm$" in out


def test_wrap_unicode_math_idempotent_inside_existing_math():
    """已经在 $...$ 内的符号不再二次包裹。"""
    s = "已 wrapped: $\\sigma_d$，未 wrapped: σ_r"
    out = _wrap_unicode_math(s)
    # 已 wrapped 段不变
    assert "$\\sigma_d$" in out
    # 未 wrapped 段被处理（注意 _r 这种 LaTeX 下标是 prompt 责任，本函数只管孤立 unicode 字符）
    assert "$\\sigma$" in out
    # 不应出现双重 $$...$$
    assert "$$" not in out


def test_latex_node_writes_tex_with_wrapped_math(mocker, workdir):
    """端到端：含 unicode 数学的 paper section 写入 tex 后已包 $...$。"""
    mocker.patch(
        "math_agent.nodes.latex.compile_latex",
        return_value=type("R", (), {"success": True, "pdf_path": "", "log": ""})(),
    )
    from math_agent.state import MathModelingState, PaperSections
    s = MathModelingState(problem="测试 α 模型", output_dir=str(workdir))
    s.paper = PaperSections(
        abstract="参数 α 影响 σ。", problem_restatement="x"*100, assumptions="x"*100,
        notation="x"*100, model_section="x"*100, solution="x"*100,
        sensitivity="敏感度 ±20%", conclusion="x"*100, references="-",
    )
    latex_node(s)
    tex = (workdir / "paper.tex").read_text(encoding="utf-8")
    assert "$\\alpha$" in tex
    assert "$\\sigma$" in tex
    assert "$\\pm$" in tex
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/nodes/test_latex.py -v -k "wrap_unicode or wrapped_math"`
Expected: ImportError (`_wrap_unicode_math` 不存在)

---

### Task 1.2：实现 `_wrap_unicode_math`

**Files:**
- Modify: `src/math_agent/nodes/latex.py`

- [ ] **Step 1: 在 `_latex_path` 函数之后插入**

```python
# 映射表：孤立 unicode 数学字符 → LaTeX 命令。覆盖国赛论文最常出现的 ~40 个符号。
# 不在表中的 unicode 字符不动（如中文、英文字母）。
_UNICODE_MATH_MAP = {
    # 希腊字母（小写）
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    # 希腊字母（大写常用）
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
    # 关系/算术
    "≥": r"\geq", "≤": r"\leq", "≠": r"\neq", "≈": r"\approx",
    "±": r"\pm", "∓": r"\mp", "×": r"\times", "÷": r"\div",
    "·": r"\cdot",
    # 集合/逻辑
    "∈": r"\in", "∉": r"\notin", "⊂": r"\subset", "⊆": r"\subseteq",
    "∪": r"\cup", "∩": r"\cap", "∅": r"\emptyset",
    "∀": r"\forall", "∃": r"\exists",
    # 求和/极限
    "∑": r"\sum", "∏": r"\prod", "∫": r"\int",
    "∞": r"\infty", "∂": r"\partial", "∇": r"\nabla",
    "→": r"\to", "←": r"\leftarrow", "↔": r"\leftrightarrow",
}


def _wrap_unicode_math(s: str) -> str:
    """把字符串中孤立的 unicode 数学字符替换为 `$\\cmd$`。

    跳过已在 `$...$` 内的部分，避免双重包裹。**只处理 _UNICODE_MATH_MAP 中的字符；
    不识别 `s_i^0` 这种裸 LaTeX 下标**（那是 writer prompt 的责任，本函数无法仅
    凭正则区分"`s_i` 是数学"和"`abc_def` 是普通名字"）。
    """
    if not s:
        return s
    # 用 $ 切分：奇数下标位于 $...$ 内（如果数量正确）；偶数下标在外面
    parts = s.split("$")
    for i in range(0, len(parts), 2):  # 只处理 $ 之外的段
        for ch, cmd in _UNICODE_MATH_MAP.items():
            if ch in parts[i]:
                parts[i] = parts[i].replace(ch, f"${cmd}$")
    return "$".join(parts)
```

- [ ] **Step 2: 改 `latex_node` 渲染前把 paper 各段都过一遍 `_wrap_unicode_math`**

找到 `latex_node` 里渲染段：

```python
    tmpl = _env.get_template("paper.tex.j2")
    tex = tmpl.render(
        problem=_latex_escape(state.problem), paper=state.paper,
        figures=safe_figures, sensitivity_runs=state.sensitivity_runs,
    )
```

替换为：

```python
    # ponytail: 对 paper 各段做确定性 unicode 数学符号包裹，避免 xelatex 字体缺失
    from math_agent.state import PaperSections
    safe_paper = PaperSections(**{
        k: _wrap_unicode_math(v) if isinstance(v, str) else v
        for k, v in state.paper.model_dump().items()
    })
    # sensitivity_runs 的 interpretation 字段同样过一遍
    from math_agent.state import SensitivityRun
    safe_sens = [
        SensitivityRun(
            parameter=r.parameter, values=r.values, metric=r.metric,
            results=r.results, interpretation=_wrap_unicode_math(r.interpretation),
            figure_path=r.figure_path,
        )
        for r in state.sensitivity_runs
    ]

    tmpl = _env.get_template("paper.tex.j2")
    tex = tmpl.render(
        problem=_latex_escape(_wrap_unicode_math(state.problem)),
        paper=safe_paper, figures=safe_figures, sensitivity_runs=safe_sens,
    )
```

注意 `problem` 顺序：**先** wrap unicode（让 `α` 变成 `$\alpha$`），**再** escape 高危字符——`$` 在 escape 表里也是高危字符，escape 在前会把 `$\alpha$` 的 `$` 转成 `\$`，wrap 就废了。

对 paper 段**不调** `_latex_escape`——因为 writer 故意写了 markdown 表格 `|...|` 和裸 LaTeX `\sigma`，这些字符是 writer 故意的，不能 escape。

- [ ] **Step 3: 跑全部 latex 测试**

Run: `pytest tests/nodes/test_latex.py -v`
Expected: 6 passed（原 2 + 新 4）

- [ ] **Step 4: 全量回归**

Run: `pytest -q`
Expected: 78 passed, 1 skipped（与 stdout-injection plan 跑完后的基线一致；若两个 plan 顺序前后则按相应基线）

- [ ] **Step 5: Commit**

```bash
git add src/math_agent/nodes/latex.py tests/nodes/test_latex.py
git commit -m "feat(latex): deterministic unicode math wrapping (alpha→\$\\alpha\$ etc.)

- latex.py: +_wrap_unicode_math() with ~40-char map covering Greek, relations,
  set/logic, sums; respects existing \$...\$ segments (split-by-\$)
- latex_node: wrap paper.* / problem / sensitivity_runs.interpretation
  BEFORE _latex_escape (else \$ gets escaped first and wrap is lost)
- tests cover greek, relations, idempotency, end-to-end tex output

Does NOT handle bare LaTeX subscripts (s_i^0, \\hat{d}_i) in text mode —
that requires writer to wrap them in \$...\$, addressed in the prompt rule
in the next commit."
```

---

## Phase 2：模板加 fontspec 兜底字体

### Task 2.1：模板加 fontspec

**Files:**
- Modify: `src/math_agent/templates/paper.tex.j2`

> 即使 Phase 1 把 unicode 都包成了 inline math，仍可能有遗漏（map 表覆盖不全的字符）。fontspec 让 xelatex 用系统字体兜底（如 STIXTwoMath、Arial Unicode）。

- [ ] **Step 1: 改 `src/math_agent/templates/paper.tex.j2` 前 7 行**

把：

```
\documentclass[12pt,a4paper]{article}
\usepackage{xeCJK}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{margin=2.5cm}
\setCJKmainfont{SimSun}  % 系统无 SimSun 时由编译失败回退到 Markdown
```

改为：

```
\documentclass[12pt,a4paper]{article}
\usepackage{xeCJK}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{fontspec}              % xelatex 字体后备
\geometry{margin=2.5cm}
\setCJKmainfont{SimSun}            % 系统无 SimSun 时编译失败回退到 Markdown
\setmainfont{Times New Roman}      % 西文主字体，覆盖默认 lmroman 不含的字符
```

> `Times New Roman` Windows 自带，国赛评委机器也一定有。

- [ ] **Step 2: 已有 latex 测试是否需要更新**

Run: `pytest tests/nodes/test_latex.py -v`
Expected: 既有测试不依赖具体字体名，应该全 PASS。

- [ ] **Step 3: Commit**

```bash
git add src/math_agent/templates/paper.tex.j2
git commit -m "feat(latex): add fontspec + Times New Roman main font

Provides a fallback for unicode characters not covered by the math-wrap map.
Both Times New Roman and SimSun ship with Windows."
```

---

## Phase 3：writer prompt 加 LaTeX 兼容规则

### Task 3.1：写测试

**Files:**
- Modify: `tests/nodes/test_writer.py`

- [ ] **Step 1: 看现有测试结构**

```bash
head -40 tests/nodes/test_writer.py
```

观察 `_rich_state()` helper 与既有 prompt 测试（如 `test_prompt_includes_prior_paper_critic_feedback`）的风格。

- [ ] **Step 2: 在 `tests/nodes/test_writer.py` 末尾追加**

```python
def test_writer_prompt_includes_latex_compat_rule():
    """IRON RULE 4：数学符号必须 $...$ 包裹。"""
    s = _rich_state()
    prompt = build_prompt(s)
    # 字面包含规则文本（用关键字面量 anchor，不要 over-specify）
    assert "$" in prompt and "LaTeX" in prompt
    # 至少两个关键关键词
    keywords = ["数学符号", "$...$", "希腊字母"]
    hit = sum(1 for k in keywords if k in prompt)
    assert hit >= 2, f"prompt 应含 LaTeX 兼容规则关键词，命中 {hit}/3"
```

- [ ] **Step 3: 跑测试确认失败**

Run: `pytest tests/nodes/test_writer.py::test_writer_prompt_includes_latex_compat_rule -v`
Expected: FAIL（关键词命中不足）

---

### Task 3.2：改 writer prompt 模板

**Files:**
- Modify: `src/math_agent/templates/writer_prompt.md.j2`

- [ ] **Step 1: 找到现有 IRON RULES 段（开头部分）**

```bash
head -10 src/math_agent/templates/writer_prompt.md.j2
```

应该看到 3 条 rule：

```
1. **禁编造数据**：...
2. **禁占位**：...
3. **每段必有论点**：...
```

- [ ] **Step 2: 在第 3 条之后追加第 4 条**

把现有的：

```
3. **每段必有论点**：禁止"罗列定义但不论证"，禁止"复述题目"。

---
```

改为：

```
3. **每段必有论点**：禁止"罗列定义但不论证"，禁止"复述题目"。
4. **LaTeX 兼容**：本论文最终会被 xelatex 渲染为 PDF。所有数学符号——包括希腊字母（`σ`, `α`）、下标变量（`s_i`, `\hat{d}_i`）、关系符（`≥`, `≤`）、求和（`∑`）——**必须**用 `$...$` 包裹为 inline math。例如：
   - ❌ `参数 σ_d 控制 ...`
   - ✅ `参数 $\sigma_d$ 控制 ...`
   - ❌ `约束 s_i^0 ≥ 0`
   - ✅ `约束 $s_i^0 \geq 0$`
   纯文本里写希腊字母**不**包 `$...$` 会让 PDF 渲染失败。markdown 表格里同样适用（`| $\sigma_d$ | 标准差 |`）。

---
```

- [ ] **Step 3: 跑测试确认通过**

Run: `pytest tests/nodes/test_writer.py -v`
Expected: 13 passed（baseline 12 + 新 1）

- [ ] **Step 4: 全量回归**

Run: `pytest -q`
Expected: 78 passed, 1 skipped（前两 Phase 之后）

- [ ] **Step 5: Commit**

```bash
git add src/math_agent/templates/writer_prompt.md.j2 tests/nodes/test_writer.py
git commit -m "feat(writer): IRON RULE 4 — math symbols must be wrapped in \$...\$

eval_v5 paper.tex failed xelatex because writer wrote bare 'α_i' / 's_i^0'
in text mode. Greek letters need a font that has them (covered by fontspec
+ math-wrap), but bare LaTeX subscripts need writer cooperation.

Combined with the deterministic unicode wrap in latex_node, this should
unblock xelatex for the next end-to-end run."
```

---

## Phase 4：v6.2 端到端验收（手动，非门禁）

⚠️ 这一段**不是 pytest**，是手动跑一次 LLM 实测。

### Task 4.1：清理 + 跑

- [ ] **Step 1: 在新 Git Bash 窗口里跑**

```bash
cd C:/Users/lwh86/Desktop/progame/build-agent
rm -rf runs/eval_v6_2 runs/eval_v6_2.log
python -m math_agent.cli run \
  --problem tests/fixtures/sample_problem.json \
  --out runs/eval_v6_2 \
  --thread v6_2 \
  --no-interrupt 2>&1 | tee runs/eval_v6_2.log
```

⚠️ 必须**新开 Git Bash**（已装 MikTeX）。ZCode harness 子进程不继承新 PATH，且 600s 超时跑不完一次 17 分钟的端到端。

预计 15-25 分钟。

### Task 4.2：验收点

- [ ] **Step 1: paper.tex 用 xelatex 真编出 PDF**

```bash
cd runs/eval_v6_2
xelatex -interaction=nonstopmode paper.tex
xelatex -interaction=nonstopmode paper.tex   # 跑两遍以解决交叉引用
ls -la paper.pdf
```

期望：第二遍编完后 `paper.pdf` 存在且 >50KB。如果失败：
- 看 `paper.log` 最后 30 行：定位是 unicode（map 漏了某符号）还是 LaTeX 命令错误（writer 漏包 `$...$`）
- map 漏了：在 `_UNICODE_MATH_MAP` 里加上对应字符 → 重新 render → 重编（不必重跑 LLM）
- writer 漏包：是 prompt 没治住的"已知风险"，可手编 paper.tex 验证，但不算 v6.2 通过

- [ ] **Step 2: PDF 可读**

```bash
start paper.pdf    # Windows 打开
```

肉眼检查：
- 中文显示正常（SimSun 渲染）
- 西文正常（Times New Roman）
- 公式正常（`$\sigma_d$` 渲染为 σ_d 数学斜体）
- 图片正常嵌入

- [ ] **Step 3: 测试套件回归没破**

```bash
pytest -q
```

Expected: 78 passed, 1 skipped。

---

## 已知非本计划范围

- paper_critic 注入 stdout：见 `2026-06-29-paper-critic-stdout-injection.md`
- writer 漏包 `$...$` 时 LaTeX 仍编不过：本计划用 prompt rule 治源 + latex_node unicode 兜底，**两层都失守**（map 不覆盖且 prompt 没听话）时仍会失败；下一轮工单处理
- evaluation/figure_critic 不接 stdout：本计划不动
- LaTeX 评论文献 BibTeX：用纯文本 references，本计划不动

---

## 自我审查

**1. Spec 覆盖：**
- prompt 治源（IRON RULE 4）→ Task 3.2 ✓
- latex_node 确定性兜底（unicode → inline math）→ Task 1.2 ✓
- 字体后备（fontspec + Times New Roman）→ Task 2.1 ✓
- v6.2 端到端验收点（真编出 PDF）→ Phase 4 ✓

**2. Placeholder 扫描：** 无 TBD/TODO；所有 step 给出可粘贴代码或精确命令。Task 4.2 Step 1 的「map 漏了：在 `_UNICODE_MATH_MAP` 里加上对应字符」是真实操作指引而非占位。

**3. 类型一致性：**
- `_wrap_unicode_math(s: str) -> str` ↔ `latex_node` 调用 `_wrap_unicode_math(state.problem)` 与 `_wrap_unicode_math(v)`（v 是 model_dump 后的 str 字段）：签名一致 ✓
- wrap 在 escape **之前**：Task 1.2 Step 2 注释说明了顺序敏感性（`$` 在 escape 表里）✓
- `PaperSections(**{k: _wrap_unicode_math(v) if isinstance(v, str) else v ...})`：保留 non-str 字段不动（PaperSections 所有字段都是 str，但 isinstance guard 防御未来扩展）✓
- `SensitivityRun(...)` 显式重建：因为 `interpretation` 要 wrap，其他字段照搬。`figure_path` 是 `str | None`，照搬不出问题 ✓
- `IRON RULE 4` 文案与测试 keyword 命中：测试用 `["数学符号", "$...$", "希腊字母"]` 3 个 anchor，prompt 文案至少含 2 个 → ✓
