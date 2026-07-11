"""模拟真实的流水线后段：table_assembler → evaluation → human_review → latex

不 mock xelatex（实际编译），mock 所有的 LLM 调用。
生成 realistic 的 state 数据并验证 PDF 输出是否正常。
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

# ── 加入 src 到 sys.path ──
_HERE = Path(__file__).resolve().parent
_SRC = (_HERE.parent / "src").resolve()
sys.path.insert(0, str(_SRC))


def build_realistic_state(workdir: Path):
    """构造一个接近真实输出的 MathModelingState，含所有字段。"""
    from math_agent.state import (
        MathModelingState, PaperSections, ModelVersion, SensitivityRun,
        FigureArtifact, CodeArtifact, EvaluationReport, Assumption,
    )

    now = "2026-07-04"

    state = MathModelingState(
        problem="城市共享单车调度优化问题：给定各站点的历史需求时序数据，"
                "设计调度方案最小化运营成本同时最大化用户满意度。",
        questions=["建立需求预测模型", "设计调度优化方案", "敏感性分析与鲁棒性检验"],
        output_dir=str(workdir),
        latex_template="default",
        stage_target="final",
        iteration=0,
        writer_iteration=0,
    )

    # ── model_versions ──
    state.model_versions.append(ModelVersion(
        stage="basic",
        description="建立基础 M/M/c 排队模型，刻画车辆到达与服务过程。",
        equations=[r"\lambda_i(t) = \alpha_i + \beta_i \sin(\omega t)"],
        variables={"lambda_i": "到达率(辆/小时)", "mu": "服务率(辆/小时)"},
    ))
    state.model_versions.append(ModelVersion(
        stage="improved",
        description="引入时变需求与容量约束的排队网络模型。",
        equations=[r"P_0 = \left[\sum_{n=0}^{c-1}\frac{(c\rho)^n}{n!} + \frac{(c\rho)^c}{c!(1-\rho)}\right]^{-1}"],
        variables={"lambda_i": "到达率(辆/小时)", "mu": "服务率(辆/小时)", "c": "服务台数(个)"},
    ))
    state.model_versions.append(ModelVersion(
        stage="final",
        description="最终版：带容量约束的时变需求 M/M/c 排队网络模型 + 整数规划调度优化。",
        equations=[
            r"\min Z = \sum_{i=1}^{n} \sum_{j=1}^{m} c_{ij} x_{ij} + \sum_{i=1}^{n} h_i s_i^+ + p_i s_i^-",
            r"\text{s.t. } \sum_{j=1}^{m} x_{ij} \leq cap_i, \forall i",
            r"s_i^+ = \max(0, D_i - S_i), s_i^- = \max(0, S_i - D_i)",
        ],
        variables={
            "x_{ij}": "调度量(辆)", "c_{ij}": "运输成本(元/辆)",
            "D_i": "需求量(辆)", "S_i": "初始库存(辆)",
            "s_i^+": "缺货量(辆)", "s_i^-": "过剩量(辆)",
            "cap_i": "运力上限(辆)", "h_i": "持有成本(元/辆)",
            "p_i": "缺货惩罚(元/辆)",
        },
        figure_purposes=["需求时序图", "调度路径图", "成本构成饼图"],
    ))

    # ── assumptions ──
    state.assumptions = [
        Assumption(statement="各站点需求相互独立", rationale="空间距离较远", sensitivity_relevant=False),
        Assumption(statement="需求服从 Poisson 分布", rationale="KS 检验 p>0.05", sensitivity_relevant=True),
        Assumption(statement="运输时间忽略不计", rationale="调度周期内完成", sensitivity_relevant=True),
    ]

    # ── sensitivity_runs ──
    state.sensitivity_runs = [
        SensitivityRun(parameter="lambda", values=[50, 75, 100, 125, 150], metric="总成本",
                       results=[1200, 1450, 1750, 2100, 2500],
                       interpretation="到达率 lambda 从 50 增至 150 时，总成本从 1200 升至 2500，增幅约 108%，高敏感。"
                                      "建议在实际运营中重点监控高峰时段需求波动。"),
        SensitivityRun(parameter="c", values=[3, 4, 5, 6, 7], metric="总成本",
                       results=[2100, 1850, 1650, 1520, 1450],
                       interpretation="服务台数 c 从 3 增至 7 时，总成本从 2100 降至 1450，递减趋势显著。"
                                      "但 c>6 后边际效益递减，建议 c=6 为最优配置。"),
        SensitivityRun(parameter="h_i", values=[5, 10, 15, 20, 25], metric="总成本",
                       results=[1600, 1650, 1750, 1900, 2100],
                       interpretation="持有成本 h_i 变动对总成本影响较温和（低敏感），"
                                      "说明模型对持有成本估计误差有较强的鲁棒性。"),
    ]

    # ── figures ──
    # Create actual PNG files
    fig1_path = str(workdir / "figures" / "demand_ts.png")
    fig2_path = str(workdir / "figures" / "schedule_path.png")
    fig3_path = str(workdir / "figures" / "cost_pie.png")
    Path(fig1_path).parent.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    Image.new("RGB", (640, 480), "white").save(fig1_path)
    Image.new("RGB", (640, 480), "white").save(fig2_path)
    Image.new("RGB", (640, 480), "white").save(fig3_path)

    def _latex_path(p: str) -> str:
        return r"\detokenize{" + p.replace("\\", "/") + "}"

    state.figures = [
        FigureArtifact(path=_latex_path(fig1_path), purpose="demand_timeseries",
                       caption="各站点需求时序图：展示 24 小时内各站点的车辆需求变化趋势",
                       quality_score=9, quality_issues=[],
                       analysis="图1展示了各站点在24小时内的需求变化趋势。高峰期集中在早8-9点和晚18-19点。"
                                "站点A和站点C的需求波动较大，站点B相对平稳。建议在高峰期前增加车辆投放。"),
        FigureArtifact(path=_latex_path(fig2_path), purpose="dispatch_path",
                       caption="最优调度路径方案：基于整数规划求解的车辆调度路径图",
                       quality_score=8, quality_issues=["颜色对比度偏低"],
                       analysis="图2展示了最优调度路径方案。总调度距离为 45.6 km，共涉及 12 个站点的"
                                "车辆再平衡。红色箭头表示调出车辆，蓝色箭头表示调入车辆。"),
        FigureArtifact(path=_latex_path(fig3_path), purpose="cost_breakdown",
                       caption="成本构成分析：运输成本、持有成本和缺货惩罚的占比",
                       quality_score=9, quality_issues=[],
                       analysis="图3展示了总成本的构成比例。运输成本占 52%，持有成本占 28%，"
                                "缺货惩罚占 20%。运输成本是主要支出，建议优化调度路线以降低该部分成本。"),
    ]

    # ── code_artifacts (including baselines) ──
    state.code_artifacts = [
        CodeArtifact(purpose="需求预测与调度主方案", code="print('main solution')",
                     stdout="RESULT: ours cost=1750.0 wait_time=12.5 satisfaction=0.85\n"
                            "RESULT: baseline=no_schedule cost=3200.0 wait_time=45.0 satisfaction=0.55",
                     success=True, category="figure",
                     artifact_paths=[fig1_path, fig2_path, fig3_path]),
        CodeArtifact(purpose="对照方案-无调度(no_schedule)", code="print('no_schedule')",
                     stdout="RESULT: no_schedule cost=3200.0 wait_time=45.0 satisfaction=0.55",
                     success=True, category="baseline:no_schedule"),
        CodeArtifact(purpose="对照方案-简单平均预测(simple_pred)", code="print('simple_pred')",
                     stdout="RESULT: simple_pred cost=2100.0 wait_time=22.0 satisfaction=0.72",
                     success=True, category="baseline:simple_pred"),
        CodeArtifact(purpose="对照方案-贪婪启发式(greedy)", code="print('greedy')",
                     stdout="RESULT: greedy cost=1950.0 wait_time=18.5 satisfaction=0.78",
                     success=True, category="baseline:greedy"),
    ]

    # ── paper sections (包含各种真实的 markdown 格式内容) ──
    state.paper = PaperSections(
        abstract="本文研究了城市共享单车系统的调度优化问题，提出了一种融合排队论与整数规划的"
                 "混合建模方法。首先，利用 M/M/c 排队网络刻画各站点的车辆到达与服务过程，"
                 "捕捉需求的时变特性。其次，以最小化运营成本为目标函数，考虑运力约束和库存平衡，"
                 "建立了整数规划调度模型。**模型优点**：能够刻画需求的时变性和随机性，适用于"
                 "大规模网络。**缺点**：假设各站点需求独立，未考虑空间相关性。"
                 "敏感性分析表明到达率 `lambda` 对总成本影响最大（高敏感），"
                 "而持有成本 `h_i` 的影响较温和（低敏感）。",

        problem_restatement='共享单车系统在城市交通中扮演着\u201c最后一公里\u201d的重要角色。然而，'
                            '由于需求在时间和空间上的不均衡分布，部分站点出现\u201c潮汐现象\u201d：'
                            '高峰期一车难求，低谷期车辆堆积。\n\n'
                            "**问题1**：如何利用历史需求数据建立各站点的需求预测模型？\n"
                            "**问题2**：在不超过运力约束的前提下，如何设计最优的车辆调度方案？\n"
                            "**问题3**：如何评估各参数对模型输出的影响程度？",

        assumptions="**假设1**（需求独立性）：各站点的车辆需求相互独立。"
                    "**依据**：站点间距均大于 500m，空间相关性较弱。"
                    "**合理性**：KS 检验 p>0.05 不显著。\n\n"
                    "**假设2**（Poisson 到达）：各站点用户到达过程服从 Poisson 分布。"
                    "**依据**：对历史数据的到达间隔进行指数分布拟合，Q-Q 图吻合良好。\n\n"
                    "**假设3**（运输时间可忽略）：调度车辆在各站点间的运输时间相对于"
                    "调度周期（1小时）可忽略。",

        notation="本文使用的符号体系说明如下。其中 `D_i` 表示站点的需求量，`S_i` 表示库存量。"
                 "主要变量包括：\n\n"
                 "| 符号 | 含义 | 单位 |\n"
                 "|------|------|------|\n"
                 "| D_i | 需求量 | 辆 |\n"
                 "| S_i | 库存量 | 辆 |\n"
                 "| x_{ij} | 调度量 | 辆 |\n"
                 "| c_{ij} | 运输成本 | 元/辆 |\n"
                 "| cap_i | 运力上限 | 辆 |\n"
                 "| λ_i | 到达率 | 辆/小时 |\n"
                 "| μ | 服务率 | 辆/小时 |\n\n"
                 "其中 i 和 j 均表示站点编号，取值范围为 1 到 N（N=15 个站点）。",

        model_section="**第一阶段：基础 M/M/c 排队模型**\n\n"
                      "在每个站点单独建立 M/M/c 排队模型。设到达率为 λ_i，每个服务台的服务率为 μ，"
                      "有 c 个并行服务台。系统稳态概率为：\n\n"
                      "$$P_0 = \\left[\\sum_{n=0}^{c-1}\\frac{(c\\rho)^n}{n!}"
                      "+ \\frac{(c\\rho)^c}{c!(1-\\rho)}\\right]^{-1}$$\n\n"
                      "其中 ρ = λ/(cμ) 为系统利用率。\n\n"
                      "**第二阶段：时变需求排队网络**\n\n"
                      "引入时变到达率 λ_i(t) = α_i + β_i sin(ωt)，采用 PSA 近似。\n\n"
                      "**最终模型：整数规划调度优化**\n\n"
                      "目标函数为最小化总成本 Z：\n\n"
                      "$$\\min Z = \\sum_{i=1}^n \\sum_{j=1}^m c_{ij} x_{ij}"
                      "+ \\sum_{i=1}^n (h_i s_i^+ + p_i s_i^-)$$\n\n"
                      "约束条件包括运力约束、库存平衡约束和整数约束。",

        solution="采用两阶段求解策略：**第一阶段**用 Kalman 滤波对时变到达率进行在线估计；"
                 "**第二阶段**将估计参数代入整数规划模型，使用分支定界法求解。\n\n"
                 "基线对照实验结果：\n\n"
                 "| 方案 | 总成本(元) | 平均等待时间(min) | 用户满意度 |\n"
                 "|------|------------|-------------------|------------|\n"
                 "| 无调度 | 3200 | 45.0 | 0.55 |\n"
                 "| 简单平均预测 | 2100 | 22.0 | 0.72 |\n"
                 "| 贪婪启发式 | 1950 | 18.5 | 0.78 |\n"
                 "| 本文方案 | 1750 | 12.5 | 0.85 |\n\n"
                 "实验结果表明，本文方案在各项指标上均优于对照方案，"
                 "总成本降低了 **45.3%**，用户满意度提升了 **54.5%**。",

        sensitivity="对模型中的关键参数进行单因素敏感性分析，探讨各参数对总成本的影响。\n\n"
                    "**到达率 λ（高敏感）**：当 λ 从 50 增至 150 时，成本从 1200 升至 2500，"
                    "增幅 108%。说明需求高峰期系统压力显著增大，建议部署动态定价机制。\n\n"
                    "**服务台数 c（中敏感）**：c 从 3 增至 7 时，成本递减但边际递减。"
                    "最优配置为 c=6。\n\n"
                    "**持有成本 h_i（低敏感）**：变动对总成本影响温和，模型鲁棒性强。",

        conclusion="本文构建了基于排队论与整数规划的共享单车调度优化模型，"
                   "从需求预测到调度方案形成完整闭环。\n\n"
                   "**主要贡献**：\n"
                   "- 首次将时变 M/M/c 排队网络引入共享单车调度建模\n"
                   "- 提出了 Kalman 滤波 + 整数规划的两阶段求解框架\n"
                   "- 系统性比较了三种基线方案，验证了本文方案的有效性\n\n"
                   "**模型局限性**： \n"
                   "- 假设各站点需求独立，未考虑空间—时间相关性\n"
                   "- 雨天、节假日等特殊场景的适应性有待验证\n\n"
                   "**未来方向**：\n"
                   "1. 引入时空 Graph Neural Network 捕捉站点间依赖\n"
                   "2. 考虑动态定价与车辆调度的联合优化\n"
                   "3. 扩展为多目标优化（成本 + 满意度 + 碳排放）",

        references="[1] Kleinrock L. Queueing Systems, Volume I: Theory. Wiley, 1975.\n"
                   "[2] Box G E P, Jenkins G M. Time Series Analysis: Forecasting and Control. "
                   "Wiley, 2015.\n"
                   "[3] Boyd S, Vandenberghe L. Convex Optimization. Cambridge University Press, 2004.\n"
                   "[4] 姜启源. 数学模型(第四版). 高等教育出版社, 2011.\n"
                   "[5] 司守奎. 数学建模算法与应用. 国防工业出版社, 2015.",

        keywords="排队论,整数规划,Kalman滤波,敏感性分析,共享单车调度",
    )

    # ── evaluation ──
    state.evaluation = EvaluationReport(
        assumption_reasonableness=8,
        modeling_creativity=8,
        result_correctness=8,
        writing_clarity=8,
        extra_depth=7,
        overall=7.9,
        issues=[],
        suggestions=["增加更多对照实验", "补充鲁棒性分析"],
    )

    return state


def validate_pdf_integrity(pdf_path: str) -> list[str]:
    """检查 PDF 文件完整性。返回问题列表，空列表 = OK。"""
    issues = []
    path = Path(pdf_path)
    if not path.exists():
        return ["PDF 文件不存在"]
    if path.stat().st_size < 1000:
        issues.append(f"PDF 文件过小 ({path.stat().st_size} bytes)")
    raw = path.read_bytes()
    header = raw[:8]
    if not header.startswith(b"%PDF-"):
        issues.append(f"PDF 头不合法: {header!r}")
    if b"%%EOF" not in raw[-100:]:
        issues.append(f"PDF 尾部缺少 %%EOF（文件可能截断）")
    # 页数检查
    pages_count = raw.count(b"/Type /Page") - raw.count(b"/Type /Pages")
    if pages_count == 0:
        issues.append(f"PDF 似乎无有效页面 (pages={pages_count})")
    return issues


def check_latex_log(log: str) -> list[str]:
    """从 xelatex log 中提取问题。返回问题列表。"""
    issues = []
    lines = log.split("\n")
    for line in lines:
        if line.startswith("! "):
            issues.append(line[:150])
    # 检查关键错误模式
    patterns = [
        ("Missing $ inserted", "缺少 $ 符号"),
        ("Emergency stop", "紧急停止"),
        ("Fatal error", "致命错误"),
        ("undefined control sequence", "未定义的控制序列"),
        ("Undefined control sequence", "未定义的控制序列"),
        ("! LaTeX Error:", "LaTeX 错误"),
        ("! Package", "宏包错误"),
        ("No pages of output", "无输出页面"),
        ("! Font", "字体错误"),
    ]
    for pattern, desc in patterns:
        if pattern in log:
            issues.append(f"{desc}: {pattern}")
    return issues


def test_full_pipeline(workdir: Path) -> tuple[bool, str, str, list[str]]:
    """跑完整流水线，返回 (success, pdf_path, tex_content, issues)."""
    issues = []

    # Step 1: Build state
    print("=" * 60)
    print("步骤 1/5: 构造 realistic state")
    state = build_realistic_state(workdir)
    print(f"  模型版本: {len(state.model_versions)}")
    print(f"  敏感性参数: {len(state.sensitivity_runs)}")
    print(f"  图片: {len(state.figures)}")
    print(f"  Code artifacts: {len(state.code_artifacts)}")
    print(f"  Evaluation: {state.evaluation}")
    print()

    # Step 2: table_assembler
    print("步骤 2/5: 运行 table_assembler")
    from math_agent.nodes.table_assembler import table_assembler_node
    ta_out = table_assembler_node(state)
    paper = ta_out["paper"]
    warnings = ta_out.get("table_warnings", [])
    print(f"  注入的表格: 符号表={bool('模型变量表' in paper.notation)}, "
          f"敏感表={bool('敏感性结果汇总' in paper.sensitivity)}, "
          f"对比表={bool('各方案结果对比' in paper.solution)}")
    if warnings:
        print(f"  清洗警告: {len(warnings)}")
        for w in warnings[:5]:
            print(f"    - {w}")
    print()

    # Step 3: Run latex_node
    print("步骤 3/5: 运行 latex_node")
    from math_agent.nodes.latex_node import latex_node
    # latex_node 内部会调用 compile_latex（真正的 xelatex）
    try:
        delta = latex_node(state)
    except Exception as e:
        import traceback
        return (False, "", "", [f"latex_node 异常: {e}\n{traceback.format_exc()}"])

    if delta.get("errors"):
        issues.extend(delta["errors"])
        print(f"  错误: {delta['errors']}")
    else:
        print("  latex_node 成功")
    print()

    # Step 4: Check outputs
    print("步骤 4/5: 检查产出文件")
    tex_path = workdir / "paper.tex"
    md_path = workdir / "paper.md"
    pdf_path = workdir / "paper.pdf"
    partial_pdf = workdir / "paper.partial.pdf"

    if tex_path.exists():
        tex_content = tex_path.read_text(encoding="utf-8")
        print(f"  paper.tex: {len(tex_content)} bytes")
    else:
        tex_content = ""
        print("  paper.tex: 缺失!")
        issues.append("paper.tex 未生成")

    if md_path.exists():
        print(f"  paper.md: {md_path.stat().st_size} bytes")

    # Check various PDF paths
    found_pdf = None
    if pdf_path.exists():
        found_pdf = pdf_path
        print(f"  paper.pdf: {pdf_path.stat().st_size} bytes (成功)")
    if partial_pdf.exists():
        found_pdf = partial_pdf
        print(f"  paper.partial.pdf: {partial_pdf.stat().st_size} bytes (编译有错误)")
    if not found_pdf:
        issues.append("未找到 PDF 输出（paper.pdf 或 paper.partial.pdf）")

    # Step 5: Validate
    print()
    print("步骤 5/5: 验证 PDF 完整性")
    if found_pdf:
        pdf_issues = validate_pdf_integrity(str(found_pdf))
        if pdf_issues:
            issues.extend(pdf_issues)
            for iss in pdf_issues:
                print(f"  ⚠ {iss}")
        else:
            print(f"  ✅ PDF 完整性验证通过 ({found_pdf.name})")
    print()

    success = len(issues) == 0
    return success, str(found_pdf) if found_pdf else "", tex_content, issues


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="latex_test_"))
    # 保留 workdir，方便调试
    result_dir = _HERE / "runs"
    result_dir.mkdir(exist_ok=True)

    print(f"工作目录: {workdir}")
    print(f"xelatex path: {shutil.which('xelatex')}")
    print()

    success, pdf_path, tex_content, issues = test_full_pipeline(workdir)

    # Copy artifacts to results
    run_name = "paper_output"
    run_dir = result_dir / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(workdir, run_dir, dirs_exist_ok=True)

    print("=" * 60)
    if success:
        print("✅ 全流程通过！PDF 已生成且完整性验证 OK")
    else:
        print(f"❌ 发现 {len(issues)} 个问题:")
        for i, iss in enumerate(issues, 1):
            print(f"  {i}. {iss}")

    print(f"\n所有产出已保存到: {run_dir}")
    print(f"  直接打开: {str(run_dir / 'paper.pdf')} 查看结果")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
