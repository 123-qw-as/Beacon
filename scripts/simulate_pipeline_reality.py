"""
模拟测试脚本：模拟真实 pipeline 中 table_assembler → evaluation → latex 全流程。
使用真实 xelatex 编译、真实 LaTeX 模板渲染，但 state 数据用模拟构造。
不修改源代码，仅测试。

目标：找出导致 PDF 截断/编译失败的根因。
"""
import sys, os, json, re, shutil, tempfile, time
from pathlib import Path

# 确保 src 在路径上
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from math_agent.state import (
    MathModelingState, PaperSections, ModelVersion, SensitivityRun,
    CodeArtifact, FigureArtifact, EvaluationReport, Assumption
)
from math_agent.nodes.table_assembler import table_assembler_node
from math_agent.nodes.latex_node import latex_node
from math_agent.nodes.latex_transform import _prepare_section, _md_table_to_latex

# ============================================================
# 辅助函数
# ============================================================
def run_simulation(state: MathModelingState, label: str, workdir: Path):
    """运行 table_assembler → latex 流程，返回结果。"""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # 1) table_assembler
    result = table_assembler_node(state)
    if result.get("table_warnings"):
        for w in result["table_warnings"]:
            print(f"  [WARN] {w}")
    state.paper = result["paper"]
    print("  table_assembler: OK")

    # 2) latex_node
    state.output_dir = str(workdir)
    errors = state.errors[:]
    try:
        latex_result = latex_node(state)
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    # 检查 errors
    new_errors = state.errors[len(errors):]
    if new_errors:
        return {"success": False, "latex_errors": new_errors}

    # 检查论文和日志
    tex_path = workdir / "paper.tex"
    pdf_path = workdir / "paper.pdf"
    partial_pdf = workdir / "paper.partial.pdf"
    md_path = workdir / "paper.md"

    issues = []
    if tex_path.exists():
        tex_content = tex_path.read_text(encoding="utf-8")
        print(f"  paper.tex: {len(tex_content)} chars")
    else:
        issues.append("paper.tex not generated")

    # 检查是否生成了 partial PDF (编译有错但产生了部分输出)
    partial_exists = partial_pdf.exists()
    pdf_exists = pdf_path.exists()

    if partial_exists:
        issues.append(f"paper.partial.pdf 存在 — LaTeX 编译有错误但产生了部分 PDF")
        print(f"  ⚠ partial PDF 存在 — 编译有错误（请检查 .tex）")
        pdf_to_check = partial_pdf
    elif pdf_exists:
        print(f"  ✅ paper.pdf 生成成功: {pdf_path.stat().st_size} bytes")
        pdf_to_check = pdf_path
    else:
        issues.append("paper.pdf 未生成 — 编译完全失败")
        print(f"  ❌ paper.pdf 未生成")
        pdf_to_check = None

    return {"success": len(issues) == 0, "issues": issues, "pdf_path": pdf_to_check, "tex_path": tex_path}


def build_basic_state(problem_text: str) -> MathModelingState:
    """构建一个基本的 state，包含所有字段的默认值。"""
    return MathModelingState(
        problem=problem_text,
        background="某城市投放了 5000 辆共享单车，分布在 200 个站点。",
        questions=[
            "建立模型预测各站点未来一小时的需求量。",
            "在不超过 100 辆运力的前提下，设计最优调度方案。"
        ],
        paper=PaperSections(),
        output_dir=None,
        latex_template="default",
    )


# ============================================================
# 测试用例
# ============================================================

def test_1_basic_paper(workdir: Path):
    """Case 1: 最简论文 — 所有 section 有纯文本，无特殊字符。"""
    state = build_basic_state("城市共享单车调度优化")
    state.paper = PaperSections(
        abstract="本文针对城市共享单车调度问题，建立了基于需求预测的动态调度模型。",
        problem_restatement="共享单车调度问题是指在需求不确定的条件下，合理分配单车资源。",
        assumptions="假设各站点需求服从泊松分布。",
        notation="符号说明见下表。",
        model_section="本文以最小化总调度成本为目标，建立整数规划模型。",
        solution="采用遗传算法对模型进行求解。",
        sensitivity="对需求均值和调度成本系数进行敏感性分析。",
        conclusion="本文模型能够有效降低调度成本，提高单车利用率。",
        references="[1] 张三, 李四. 共享单车调度优化研究. 运筹学学报, 2023.\n[2] 王五. 城市交通需求预测方法. 交通工程, 2022."
    )
    return run_simulation(state, "Case 1: 最简论文（纯文本）", workdir / "case1_basic")


def test_2_unicode_greek(workdir: Path):
    """Case 2: 含 Unicode 希腊字母和数学符号。"""
    state = build_basic_state("基于排队论的共享单车调度优化")
    state.paper = PaperSections(
        abstract="本文引入排队论模型，以 λ 表示到达率，μ 表示服务率。",
        problem_restatement="考虑 M/M/1 排队系统，顾客到达服从 Poisson 过程，参数 λ = 0.5。",
        assumptions="假设站点容量 C 为 30 辆，α = 0.05 为显著性水平。",
        notation="",
        model_section="""
本文建立目标函数：
min Z = Σ_i Σ_j c_{ij} x_{ij}

约束条件：
Σ_j x_{ij} ≤ S_i, ∀i ∈ I
Σ_i x_{ij} ≥ D_j, ∀j ∈ J
x_{ij} ≥ 0, x_{ij} ∈ ℤ

其中 S_i 是站点 i 的库存，D_j 是站点 j 的需求。
""",
        solution="使用 Gurobi 求解器求解上述整数规划模型。",
        sensitivity="对参数 λ 和 μ 进行敏感性分析。",
        conclusion="本文模型在 α = 0.05 水平下显著优于对照方案。结论：λ 越大，所需的调度车辆越多。",
        references="[1] 排队论基础. 清华大学出版社."
    )
    return run_simulation(state, "Case 2: Unicode 希腊字母 + 数学符号", workdir / "case2_unicode")


def test_3_markdown_tables(workdir: Path):
    """Case 3: 含 markdown 表格的论文（模拟 writer 输出）。"""
    state = build_basic_state("需求预测与调度优化")
    state.paper = PaperSections(
        abstract="本文对比了三种方案在不同场景下的表现。",
        problem_restatement="调度问题需要同时考虑需求和供给。",
        assumptions="假设各站点需求独立。",
        notation="",
        model_section="""
本文采用以下符号：

| 符号 | 含义 | 单位 |
|------|------|------|
| S_i | 站点 i 的库存 | 辆 |
| D_i | 站点 i 的需求 | 辆 |
| x_{ij} | 从 i 到 j 调度量 | 辆 |
| c_{ij} | 从 i 到 j 单位成本 | 元/辆 |

主要公式如下。
""",
        solution="",
        sensitivity="",
        conclusion="""
对比结果如下：

| 方案 | 总成本（元） | 满足率（%） | 计算时间（秒） |
|------|-------------|-------------|---------------|
| 无调度 | 28500 | 68.5 | 0 |
| 简单预测 | 19300 | 82.3 | 12 |
| 贪婪启发式 | 15700 | 89.1 | 45 |
| 本文方案 | 12300 | 94.7 | 120 |

本文方案在各项指标上均表现最优。
""",
        references="[1] 调度算法综述."
    )
    return run_simulation(state, "Case 3: Markdown 表格", workdir / "case3_tables")


def test_4_table_with_special_chars(workdir: Path):
    # Case 4: 变量表含 LaTeX 特殊字符（\\mathbf, $, _, & 等）
    state = build_basic_state("含特殊符号的模型")

    # 模拟 model_versions 含 LaTeX 变量名
    state.model_versions.append(ModelVersion(
        stage="final",
        description="最终模型",
        equations=["max Z = Σ_i Σ_j (r_{ij} - c_{ij}) x_{ij}"],
        variables={
            "S_i": "站点 i 的库存（辆）",
            "D_j": "站点 j 的需求（辆）",
            "x_{ij}": "从 i 调度到 j 的数量（辆）",
            "c_{ij}": "单位运输成本（元/辆）",
            "r_{ij}": "单位收入（元/辆）",
            "\\mathbf{h}_i": "站点 i 的优先级系数",
            "F_{i,t}": "t 时刻站点 i 的流量（辆/小时）",
            "\\alpha_i": "站点 i 的衰减系数",
            "p_i^{(k)}": "第 k 次迭代站点 i 的概率",
            "demand_rate_λ": "需求率（辆/小时）",
        }
    ))

    state.paper = PaperSections(
        abstract="本文模型包含多个变量和参数。",
        problem_restatement="问题重述。",
        assumptions="假设。",
        notation="符号说明如下。",
        model_section="本文建立如下模型。",
        solution="求解过程。",
        sensitivity="敏感性分析。",
        conclusion="结论。",
        references="[1] 参考文献."
    )
    return run_simulation(state, "Case 4: 变量表含 LaTeX 特殊字符", workdir / "case4_special_chars")


def test_5_sensitivity_table(workdir: Path):
    """Case 5: 敏感性分析 + 表格。"""
    state = build_basic_state("敏感性分析测试")
    state.paper = PaperSections(
        abstract="本文对关键参数进行敏感性分析。",
        problem_restatement="问题重述。",
        assumptions="假设。",
        notation="符号说明。",
        model_section="模型建立。",
        solution="模型求解。\n\n对比结果如下：\n\n| 方案 | 总成本 |\n|------|--------|\n| 本文方案 | 12300 |\n| 对照方案 | 15700 |",
        sensitivity="敏感性分析结果如下。",
        conclusion="结论。",
        references="[1] 参考文献."
    )

    # 添加敏感性运行数据
    state.sensitivity_runs.append(SensitivityRun(
        parameter="需求率 λ",
        values=[0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0],
        metric="总成本（元）",
        results=[8500, 9200, 10500, 12300, 15800, 23500, 45200],
        interpretation="当需求率 λ 从 0.1 增至 5.0 时，总成本从 8500 元增至 45200 元，增长约 4.3 倍。总体敏感度为高。"
    ))
    state.sensitivity_runs.append(SensitivityRun(
        parameter="单位调度成本 c",
        values=[0.5, 1.0, 2.0, 3.0, 5.0],
        metric="总成本（元）",
        results=[10200, 12300, 16500, 20700, 29100],
        interpretation="调度成本系数对总成本影响显著。"
    ))

    return run_simulation(state, "Case 5: 敏感性分析表 + 对比表", workdir / "case5_sensitivity")


def test_6_writer_realistic_output(workdir: Path):
    """Case 6: 模拟 writer 真实输出 — 含各种 markdown 格式混合。"""
    state = build_basic_state("基于混合整数规划的城市共享单车动态调度")

    state.model_versions.append(ModelVersion(
        stage="final",
        description="混合整数规划模型",
        equations=[
            "\\min Z = \\sum_{i \\in I} \\sum_{j \\in J} c_{ij} x_{ij} + \\sum_{i \\in I} p_i y_i",
            "\\text{s.t.} \\quad \\sum_{j \\in J} x_{ij} \\leq S_i + y_i, \\quad \\forall i \\in I",
            "\\sum_{i \\in I} x_{ij} \\geq D_j, \\quad \\forall j \\in J",
            "x_{ij} \\in \\mathbb{Z}_{\\geq 0}, \\quad y_i \\in \\{0, 1\\}"
        ],
        variables={
            "x_{ij}": "从站点 i 到 j 的调度量（辆）",
            "y_i": "站点 i 是否启用备用库存",
            "c_{ij}": "从 i 到 j 的单位运输成本（元/辆）",
            "p_i": "站点 i 的备用库存启用成本（元）",
            "S_i": "站点 i 的初始库存（辆）",
            "D_j": "站点 j 的需求量（辆）",
        }
    ))

    state.code_artifacts.append(CodeArtifact(
        purpose="主方案求解",
        code="import numpy as np\n# ... 求解代码 ...",
        stdout="RESULT: ours | 总成本: 12345 元 | 满足率: 94.7% | 计算时间: 128s",
        success=True,
        category="figure",
    ))
    state.code_artifacts.append(CodeArtifact(
        purpose="无调度 baseline",
        code="",
        stdout="RESULT: no_schedule | 总成本: 28500 元 | 满足率: 68.5%",
        success=True,
        category="baseline:no_schedule",
    ))

    state.paper = PaperSections(
        abstract="""
本文针对城市共享单车动态调度问题，建立了**混合整数规划模型**。模型以最小化总调度成本为目标，
同时考虑站点库存约束和需求不确定性。为了提高求解效率，设计了基于**遗传算法**的启发式求解策略。
实验结果表明，本文方案相比对照方案降低调度成本 **56.7%**，满足率提升至 **94.7%**。
""",
        problem_restatement="""

## 问题背景

城市共享单车系统面临**潮汐现象**：早高峰时段居住区车辆大量流出至办公区，
导致居住区车辆短缺而办公区车辆堆积。调度中心需在有限运力条件下，
合理分配车辆以最大化服务水平和运营效率。

### 已知条件

- 共享单车总数：5000 辆
- 站点数量：200 个
- 调度车辆运力上限：100 辆/次
- 历史需求数据：30 天

## 问题要求

1. 建立模型预测各站点未来一小时的需求量 `D_j`。
2. 设计最优调度方案，在运力约束下最大化需求满足率。
""",
        assumptions="""
1. 假设各站点需求服从**泊松分布**，参数 λ_i 可通过历史数据估计。
2. 假设调度车辆行驶速度恒定，不受交通状况影响。
3. 假设用户取车行为受车辆可用性影响，即存在**需求弹性**。
4. 假设站点容量 C_i 固定，无法临时扩容。
""",
        notation="""
本文使用以下符号：

| 符号 | 含义 | 单位 |
|------|------|------|
| `S_i` | 站点 i 的初始库存 | 辆 |
| `D_j` | 站点 j 的需求量 | 辆 |
| `x_{ij}` | 从 i 到 j 的调度量 | 辆 |
| `c_{ij}` | 从 i 到 j 的运输成本 | 元/辆 |
| `y_i` | 是否启用站点 i 备用库存 | 0/1 |
| `p_i` | 站点 i 的备用库存成本 | 元 |
| `λ_i` | 站点 i 的需求到达率 | 辆/小时 |
| `μ_i` | 站点 i 的服务率 | 辆/小时 |
""",
        model_section="""
本文建立基于混合整数规划的调度优化模型。

## 目标函数

\\[ \\min Z = \\sum_{i \\in I} \\sum_{j \\in J} c_{ij} x_{ij} + \\sum_{i \\in I} p_i y_i \\]

其中第一项表示运输成本，第二项表示备用库存启用成本。

## 约束条件

### 库存约束

\\[ \\sum_{j \\in J} x_{ij} \\leq S_i + y_i, \\quad \\forall i \\in I \\]

### 需求约束

\\[ \\sum_{i \\in I} x_{ij} \\geq D_j, \\quad \\forall j \\in J \\]

### 变量范围

\\[ x_{ij} \\in \\mathbb{Z}_{\\geq 0}, \\quad y_i \\in \\{0, 1\\} \\]

### 需求预测

采用 **ARIMA(1,1,1)** 模型对需求进行预测：

\\[ (1 - \\phi_1 B)(1 - B) D_t = (1 + \\theta_1 B) \\varepsilon_t \\]

其中 `B` 为滞后算子，`φ_1` 为自回归系数，`θ_1` 为移动平均系数。
""",
        solution="""
## 求解算法

采用**遗传算法**求解上述混合整数规划模型，关键参数如下：

| 参数 | 取值 |
|------|------|
| 种群大小 | 200 |
| 交叉概率 | 0.85 |
| 变异概率 | 0.05 |
| 最大迭代次数 | 500 |

## 实验结果

对比本文方案与对照方案：

| 方案 | 总成本（元） | 满足率（%） | 计算时间（秒） |
|------|-------------|-------------|---------------|
| 无调度 | 28500 | 68.5 | 0 |
| 简单平均预测 | 19300 | 82.3 | 12 |
| 贪婪启发式 | 15700 | 89.1 | 45 |
| **本文方案** | **12345** | **94.7** | **128** |

实验结果显示，本文方案在满足率和总成本上均显著优于对照方案。
""",
        sensitivity="""
## 敏感性分析

对关键参数进行敏感性分析：

| 参数 | 取值范围 | 指标 | 指标变化范围 | 敏感性评级 |
|------|---------|------|-------------|-----------|
| 需求率 λ | [0.1, 5.0] | 总成本 | [8500, 45200] | 高 |
| 调度成本 c | [0.5, 5.0] | 总成本 | [10200, 29100] | 高 |
| 站点容量 C | [20, 50] | 满足率 | [0.85, 0.98] | 中 |

分析表明，需求率和调度成本对总成本影响显著，而站点容量对满足率影响中等。
""",
        conclusion="""
本文针对城市共享单车动态调度问题，提出了基于混合整数规划的优化模型，并采用遗传算法进行求解。
主要结论如下：

1. **模型有效性**：本文方案相比"无调度"方案降低总成本 **56.7%**，需求满足率从 **68.5%** 提升至 **94.7%**。
2. **算法效率**：遗传算法在 128 秒内收敛到稳定解，满足实际调度的时间要求。
3. **敏感性**：需求率 λ 是影响总成本最敏感的参数，实际应用中应重点关注需求预测精度。
4. **应用价值**：本文模型可直接部署到城市共享单车调度平台，具有显著的经济效益和社会效益。

未来工作将考虑多时段耦合的联合调度方案，以及引入实时需求响应的动态调度策略。
""",
        references="""
[1] 张伟, 李明. 共享单车调度优化方法综述. 系统工程理论与实践, 2023.
[2] 王芳. 基于需求预测的动态调度模型. 运筹与管理, 2022.
[3] F. Glover. Future paths for integer programming. *Computers & Operations Research*, 1986.
[4] 马丽. 遗传算法在组合优化中的应用. 清华大学学报, 2021.
"""
    )

    return run_simulation(state, "Case 6: Writer 真实输出（混合格式）", workdir / "case6_realistic")


def test_7_long_paper(workdir: Path):
    """Case 7: 长论文 — 模拟 writer 产生大量内容。"""
    state = build_basic_state("长论文测试")

    # 生成长内容
    long_text = "本文对调度问题进行了深入研究。" * 500

    state.paper = PaperSections(
        abstract=long_text[:2000],
        problem_restatement="## 问题背景\n\n" + long_text[:1000],
        assumptions="""
1. 假设一：需求独立分布。
2. 假设二：车辆不损坏。
""" * 50,
        notation="符号说明。",
        model_section="""
## 模型建立

\\[ \\min Z = \\sum_i \\sum_j c_{ij} x_{ij} \\]

### 推导过程

令 `S_i` 表示库存，`D_j` 表示需求。
""",
        solution="求解过程。" + "测试内容。" * 300,
        sensitivity="敏感性分析。" * 200,
        conclusion="结论。" * 200,
        references="[1] 参考1.\n[2] 参考2.\n" * 50,
    )

    return run_simulation(state, "Case 7: 长论文（溢出测试）", workdir / "case7_long")


def test_8_forbidden_words_and_sanitize(workdir: Path):
    """Case 8: 禁用词清洗 + 特殊转义场景。"""
    state = build_basic_state("测试论文")

    state.model_versions.append(ModelVersion(
        stage="final",
        description="模型",
        equations=["E = mc^2"],
        variables={
            "D_i": "需求量（件）",
            "R_{max}": "最大收益（元）",
        }
    ))

    state.sensitivity_runs.append(SensitivityRun(
        parameter="参数 α",
        values=[0.1, 0.5, 1.0],
        metric="收益",
        results=[100, 200, 300],
        interpretation="结果解读。"
    ))

    state.code_artifacts.append(CodeArtifact(
        purpose="求解代码",
        code="print('hello')",
        stdout="RESULT: ours | cost: 100\nRESULT: no_schedule | cost: 300",
        success=True,
        category="figure",
    ))
    state.code_artifacts.append(CodeArtifact(
        purpose="对照",
        code="",
        stdout="RESULT: no_schedule | cost: 300",
        success=True,
        category="baseline:no_schedule",
    ))

    state.paper = PaperSections(
        abstract="""
本文使用了PaperCritic进行内部评审，结果显示Claim缺乏Evidence支撑。
Reasoning过程不够严谨。代码[1]需要优化。代码2已修复。
李华同学负责数据收集。回应评审意见：已修改。
""",
        problem_restatement="问题重述。",
        assumptions="假设。",
        notation="符号说明。",
        model_section="模型部分包含依赖项 `sensitivity_cost.png` 文件。",
        solution="解决方案。",
        sensitivity="敏感性分析。",
        conclusion="结论。",
        references="[1] Smith, J. Evidence-based Reasoning in Operations Research. 2020."
    )

    return run_simulation(state, "Case 8: 禁用词 + 路径转义", workdir / "case8_forbidden")


def test_9_nested_math_and_tables(workdir: Path):
    """Case 9: 复杂内联数学 + 表格混合。"""
    state = build_basic_state("数学公式密集论文")

    state.model_versions.append(ModelVersion(
        stage="final",
        description="复杂模型",
        equations=["complex"],
        variables={}
    ))

    state.paper = PaperSections(
        abstract="本文研究 `α_i`、`β_{ij}`、`γ_i^{(k)}` 等参数的影响。",
        problem_restatement="""
在排队论中，参数 `λ_i` 表示到达率，`μ_j` 表示服务率。
系统稳态概率 `π_i` 满足：

\\[ \\pi_i = \\frac{\\lambda_i}{\\mu_j} \\pi_{i-1} \\]

其中 `π_0` 可通过归一化条件确定。
""",
        assumptions="""
假设 `α > 0`，`β ≥ 0`，`γ_i^{(t)}` 为第 t 次迭代的站点 i 参数。
系统容量 `C_max = 100`。

| 参数 | 下限 | 上限 |
|------|------|------|
| `α` | 0 | 10 |
| `β` | 0 | 5 |
| `γ_i^{(t)}` | 0 | 1 |
""",
        notation="符号说明。",
        model_section="""
本文模型如下。

## 目标函数

`Z = Σ_i α_i x_i + Σ_j β_j y_j`

其中 `α_i` 是权重系数，`β_j` 是惩罚系数。

## 约束条件

`x_i ≤ C_max`, `∀i`

利用 `λ_i` 和 `μ_j` 计算系统性能指标。

`ρ = λ / (k · μ)`，其中 k 为服务台数量。

\\[ W_q = \\frac{C(k, \\lambda/\\mu)}{k\\mu - \\lambda} \\]

其中 `C(k, ρ)` 是 Erlang C 公式。
""",
        solution="""
## 参数设置

| 参数 | 符号 | 取值 |
|------|------|------|
| 到达率 | `λ` | 0.5 |
| 服务率 | `μ` | 0.8 |
| 服务台数 | `k` | 3 |
| 系统容量 | `C_max` | 100 |

### 运行结果

| 指标 | 值 |
|------|-----|
| 系统负荷 `ρ` | 0.208 |
| 平均等待时间 `W_q` | 0.003 |
| 平均排队长度 `L_q` | 0.0015 |
""",
        sensitivity="""
对 `λ` 和 `μ` 进行敏感性分析：

当 `λ` 从 0.1 变化到 0.9，`W_q` 的变化如下：

| `λ` | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|-----|-----|-----|-----|-----|-----|
| `W_q` | 0.000 | 0.001 | 0.003 | 0.010 | 0.050 |
""",
        conclusion="""
本文研究了 `α`、`β`、`γ` 等参数对系统性能的影响。
当 `λ < μ` 时，系统稳定；当 `λ → μ` 时，`W_q → ∞`。
""",
        references="[1] 排队论基础."
    )

    return run_simulation(state, "Case 9: 多层嵌套数学 + 表格", workdir / "case9_nested_math")


def test_10_image_paths_and_captions(workdir: Path):
    """Case 10: 含图片路径和长 caption 的论文。"""
    state = build_basic_state("含图片的论文")

    state.figures.append(FigureArtifact(
        path=r"C:\Users\lwh86\Desktop\progame\build-agent\analysis_pdf\sensitivity_cost.png",
        purpose="调度成本敏感性分析图",
        caption="不同需求率下调度成本随站点容量的变化趋势，曲线呈下降趋势，说明站点容量增大时调度成本降低。",
        quality_score=8,
        quality_issues=["图例位置需微调"],
        analysis="""
图1展示了不同需求率 `λ` 下，站点容量 `C` 对调度成本的影响。
可以看出：

- 当 `λ = 0.5` 时，调度成本在 `C = 30` 处达到最优。
- 当 `λ = 2.0` 时，调度成本随 `C` 单调递减。
- 整体上，站点容量越大，调度成本越低，但边际效益递减。
"""
    ))

    state.figures.append(FigureArtifact(
        path=r"C:\Users\lwh86\Desktop\progame\build-agent\analysis_pdf\convergence_curve.png",
        purpose="遗传算法收敛曲线图",
        caption="遗传算法在 500 代内的收敛曲线，在大约 128 代时收敛到稳定解，将适应度收敛曲线分为快降区、慢降区和稳定区三个阶段。",
        quality_score=9,
        quality_issues=[],
        analysis="遗传算法的收敛曲线显示，算法在 128 代左右达到收敛，验证了算法的收敛性。"
    ))

    state.paper = PaperSections(
        abstract="本文包含实验图表。",
        problem_restatement="问题。",
        assumptions="假设。",
        notation="符号。",
        model_section="模型。",
        solution="实验结果详见下图。",
        sensitivity="敏感性分析。",
        conclusion="结论。",
        references="[1] 参考."
    )

    return run_simulation(state, "Case 10: 图片路径 + 长 caption", workdir / "case10_images")


def test_11_gmcm_template(workdir: Path):
    """Case 11: GMCM 国赛模板。"""
    state = build_basic_state("基于混合整数规划的共享单车动态调度优化")
    state.latex_template = "gmcm"
    state.school = "XX大学"
    state.team_id = "2026001"
    state.members = "张三,李四,王五"

    state.paper = PaperSections(
        abstract="本文针对城市共享单车调度问题，建立了混合整数规划模型。",
        problem_restatement="问题重述。",
        assumptions="假设。",
        notation="符号说明。",
        model_section="模型建立。",
        solution="求解。",
        sensitivity="敏感性分析。",
        conclusion="结论。",
        keywords="共享单车, 调度优化, 混合整数规划, 遗传算法",
        references="[1] 参考."
    )

    return run_simulation(state, "Case 11: GMCM 国赛模板", workdir / "case11_gmcm")


def test_12_problematic_unicode_combos(workdir: Path):
    """Case 12: 各种有问题 Unicode 组合。"""
    state = build_basic_state("含复杂 Unicode 的论文")

    state.paper = PaperSections(
        abstract="""
本文使用 α、β、γ、δ、ε、ζ、η、θ、λ、μ、ν、ξ、π、ρ、σ、τ、υ、φ、χ、ψ、ω
等希腊字母表示参数。同时使用 ∑、∏、∫、∈、⊆、∪、∩、∀、∃、∞、∂、∇、→、≥、≤、≠、≈、± 等数学符号。
""",
        problem_restatement="""
参数关系如下：

当 λ → ∞ 时，μ ≥ λ 且 σ² ≠ 0。系统稳定性条件为 ∀i ∈ I: ρ_i ≤ 1。
若 ∃j: ρ_j > 1，则系统不稳定。

||| 测试 ||| 这个应该不是表格 |||
""",
        assumptions="""
假设 ∀i: α_i > 0。∃ 至少一个 β_j = 0。
""",
        notation="",
        model_section="""
## 数学公式

文中使用 `D_i^{(k)}` 表示第 k 次迭代的站点 i 需求。
`S_i^{(0)}` 表示初始库存。

混合符号：`α_i` 和 `β_j^{(t)}` 的关系为 `α_i = β_j^{(t)} · γ_{ij}`。

\\[ R_{ij}(t) = \\frac{D_i(t)}{\\sum_k D_k(t)} \\cdot C_{ij} \\]

\\[ L = \\sum_{i \\in I} \\left( D_i - \\sum_{j \\in J} x_{ij} \\right)^2 \\]

`S_i`、`D_j`、`x_{ij}` 构成决策变量集合。
""",
        solution="""
## 结果表

| 指标 | 符号 | 值 |
|------|------|------|
| 均值 | `μ` | 0.5 |
| 方差 | `σ²` | 0.12 |
| 偏度 | `γ₁` | 0.3 |
| 峰度 | `κ` | 2.8 |
| 效率 | `η` | 0.95 |

## 特殊情况

a_b_c 这种三段式下标的处理。
T_1, T_2, ..., T_n 的时间序列。
sensitivity_analysis_2024.png 文件名不要炸。
""",
        sensitivity="""
`x_{ij}^{(k)}` 的收敛性分析：

当 `k → ∞` 时，`x_{ij}^{(k)} → x_{ij}^*`。
""",
        conclusion="""
本文使用 `λ=0.5`、`μ=0.8`、`ρ=0.625` 进行实验。
结果表明 `ρ < 1` 时系统稳定。

`α = (β + γ) / δ` 为关键关系式。
""",
        references="""
[1] 含特殊字符的论文标题: `λ_estimation` 方法研究.
[2] 使用 `α=0.05` 作为显著性水平.
"""
    )

    return run_simulation(state, "Case 12: Unicode 组合 + 边界情况", workdir / "case12_unicode_combos")


def test_13_extreme_escaping(workdir: Path):
    """Case 13: 各种需要转义的极端情况。"""
    state = build_basic_state("转义测试论文")

    # 模拟 table_assembler 输出的变量表含特殊字符
    state.model_versions.append(ModelVersion(
        stage="final",
        description="测试模型",
        equations=["test"],
        variables={
            "a": "值 A",
            "b_1": "值 B_1（单位）",
            "c_{ij}": "C_{ij} 值（件）",
            "\\mathbf{x}": "向量 X（单位）",
            "F_{i,t}": "F_{i,t} 流量（辆/小时）",
            "100%": "百分比（%）",
            "#1": "排名第一（#1）",
            "dollar$ign": "美元符号（$100）",
        }
    ))

    # 模拟 sensitivity 和 baseline 数据
    state.sensitivity_runs.append(SensitivityRun(
        parameter="参数 a（含特殊: 100% $ign）",
        values=[1, 2, 3],
        metric="metric_test_1",
        results=[10, 20, 30],
        interpretation="结果解读（含 100% 特殊字符）。"
    ))

    state.code_artifacts.append(CodeArtifact(
        purpose="测试代码",
        code="print('test')",
        stdout="RESULT: ours | metric: 100 | rate: 95%\nRESULT: no_schedule | metric: 200 | rate: 70%",
        success=True,
        category="figure",
    ))
    state.code_artifacts.append(CodeArtifact(
        purpose="对照",
        code="",
        stdout="RESULT: no_schedule | metric: 200 | rate: 70%",
        success=True,
        category="baseline:no_schedule",
    ))

    state.paper = PaperSections(
        abstract="测试各种特殊字符 & % # _ { } $ 的转义处理。",
        problem_restatement="包含 100% 完成度 & 排名 #1 的 $100 预算。",
        assumptions="",
        notation="符号说明区域。",
        model_section="模型包含 a & b 关系。使用 #pragma 指令。百分号 % 要转义。下划线 _ 在文件名中：sensitivity_test_2024.png。",
        solution="结果包含 50% 提升。测试文件路径：output/results/v3_test/final_report.pdf。",
        sensitivity="""
敏感性测试结果：参数变化 100% 时，指标变化 50%。
文件名示例：sensitivity_analysis_report_2024_v3_final.png
""",
        conclusion=(
        "结论：系统性能提升 30% ~ 50%。\n"
        "注意：文件路径中的反斜杠需要转义。\n"
    ),
        references="[1] 测试文献：100% 提升率的实验研究（2024）."
    )

    return run_simulation(state, "Case 13: 极端转义情况", workdir / "case13_extreme_escaping")


# ============================================================
# 主运行逻辑
# ============================================================
def main():
    base_workdir = Path(__file__).resolve().parent.parent / "analysis_pdf"
    base_workdir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print(f"  math-agent 仿真测试 — 模拟 table_assembler → evaluation → latex 全流程")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  xelatex: {shutil.which('xelatex')}")
    print(f"  Python: {sys.version}")
    print(f"  工作目录: {base_workdir}")
    print("=" * 70)

    test_cases = [
        ("Basic paper (plain text)", test_1_basic_paper),
        ("Unicode Greek + math symbols", test_2_unicode_greek),
        ("Markdown tables", test_3_markdown_tables),
        ("LaTeX special chars in variables", test_4_table_with_special_chars),
        ("Sensitivity + comparison tables", test_5_sensitivity_table),
        ("Writer realistic mixed output", test_6_writer_realistic_output),
        ("Long paper overflow test", test_7_long_paper),
        ("Forbidden words + path escaping", test_8_forbidden_words_and_sanitize),
        ("Nested math + tables", test_9_nested_math_and_tables),
        ("Image paths + long captions", test_10_image_paths_and_captions),
        ("GMCM national template", test_11_gmcm_template),
        ("Problematic Unicode combos", test_12_problematic_unicode_combos),
        ("Extreme escaping cases", test_13_extreme_escaping),
    ]

    results = []
    for label, test_fn in test_cases:
        # 清理旧的 case 目录
        case_dir = base_workdir / f"{test_fn.__name__}"
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        try:
            r = test_fn(case_dir)
            results.append((label, r, None))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            results.append((label, {"success": False, "error": str(e)}, tb))
            # 保存 traceback
            with open(case_dir / "exception.txt", "w", encoding="utf-8") as f:
                f.write(f"{e}\n\n{tb}")

    # 汇总
    print("\n\n" + "=" * 70)
    print("  测试结果汇总")
    print("=" * 70)

    success_count = 0
    fail_count = 0

    for label, r, tb in results:
        ok = r.get("success", False)
        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            success_count += 1
        else:
            fail_count += 1
        print(f"  {status}  {label}")
        if not ok:
            if r.get("error"):
                print(f"         异常: {r['error']}")
            if r.get("latex_errors"):
                for e in r["latex_errors"]:
                    print(f"         LaTeX: {e[:200]}")
            if r.get("issues"):
                for iss in r["issues"]:
                    print(f"         Issue: {iss}")

    print(f"\n  总计: {success_count} passed / {fail_count} failed / {len(results)} total")
    print(f"\n  详细输出目录: {base_workdir}")
    print("=" * 70)

    # 保存汇总 JSON
    summary = []
    for label, r, _ in results:
        summary.append({
            "label": label,
            "success": r.get("success", False),
            "error": r.get("error"),
            "latex_errors": r.get("latex_errors", []),
            "issues": r.get("issues", []),
        })
    with open(base_workdir / f"simulation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
