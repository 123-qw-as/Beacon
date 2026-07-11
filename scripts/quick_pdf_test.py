#!/usr/bin/env python
r"""快速验证 PDF 编译链路：mock state → table_assembler → latex → PDF。

不跑 graph、不调 LLM。用含 LaTeX 命令（\tau \text \beta \bigl \bar \boldsymbol）
的真实风格 paper 文本，验证：
1. \b \t \f JSON escape fix 生效（paper 文本里没有 backspace/tab 损坏）
2. 变量表 LaTeX 转义生效（\mathbf{} 等不崩 tabularx）
3. 对比表注入生效
4. 附录精选生效
5. PDF 完整编译（>4 页）

用法：python scripts/quick_pdf_test.py
耗时：<30 秒
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from math_agent.state import (
    MathModelingState, ModelVersion, PaperSections, CodeArtifact,
    FigureArtifact, SensitivityRun, Assumption, DerivationStep,
)
from math_agent.nodes.table_assembler import table_assembler_node
from math_agent.nodes.latex_node import latex_node


def build_mock_state(output_dir: str) -> MathModelingState:
    """构造含 LaTeX 命令的 mock state，模拟真实 writer 输出。"""
    s = MathModelingState(problem="共享单车调度测试", output_dir=output_dir)

    # model_versions：含 \bar \boldsymbol 等 LaTeX 命令的变量
    s.model_versions.append(ModelVersion(
        stage="final",
        description="XGBoost + 随机规划",
        equations=[r"\min \sum_{\tau=1}^{T} \sum_{i=1}^{N} \bigl(c_{pen} \cdot p_i + c_{wait} \cdot w_v \bigr)"],
        variables={
            r"$\hat{D}_{i,t}^{out}$": "站点i时段t的借出需求预测值(辆)",
            r"$\bar{D}_{i,t}$": "历史同期需求均值(辆)",
            r"$\boldsymbol{f}$": "调度量向量(辆)",
            r"$\mathcal{B}_{\epsilon}(\hat{P})$": "Wasserstein模糊集",
            r"$\beta$": "回归系数",
            r"$\tau$": "分位数水平",
            "x_i": "调度量(辆)",
            "N_s": "站点总数(个)",
        },
        figure_purposes=["需求时序图", "调度路径图"],
        derivation_steps=[
            DerivationStep(
                title="参数估计",
                motivation="用MLE估计",
                statement=r"$\hat{\theta} = \arg\max \sum_i \log p(d_i | \theta)$",
                result=r"$\hat{\beta} = 0.85$",
            ),
        ],
    ))

    # code_artifacts：含主方案 + 3 个对照方案
    s.code_artifacts = [
        CodeArtifact(
            purpose="需求预测与调度优化",
            code="import numpy as np\nprint('main solution')\nprint(f'RESULT: baseline=ours total_cost=750.5 service_rate=0.95')",
            stdout="main solution\nRESULT: baseline=ours total_cost=750.5 service_rate=0.95",
            success=True,
            category="figure",
        ),
        CodeArtifact(
            purpose="无调度对照方案",
            code="print('no schedule')",
            stdout="RESULT: baseline=no_schedule total_cost=1245.3 service_rate=0.82",
            success=True,
            category="baseline:no_schedule",
        ),
        CodeArtifact(
            purpose="简单平均预测对照方案",
            code="print('simple pred')",
            stdout="RESULT: baseline=simple_pred total_cost=1100.0 service_rate=0.85",
            success=True,
            category="baseline:simple_pred",
        ),
        CodeArtifact(
            purpose="贪婪启发式对照方案",
            code="print('greedy')",
            stdout="RESULT: baseline=greedy total_cost=980.0 service_rate=0.91",
            success=True,
            category="baseline:greedy",
        ),
    ]

    # figures：mock 一张
    from PIL import Image
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "test_fig.png"
    Image.new("RGB", (320, 240), "white").save(fig_path)
    s.figures = [
        FigureArtifact(path=str(fig_path), purpose="需求时序图", caption="各站点需求时序", quality_score=8),
    ]

    # sensitivity_runs
    s.sensitivity_runs = [
        SensitivityRun(parameter="alpha", values=[0.01, 0.1, 0.5],
                       metric="MAE", results=[3.2, 2.4, 2.8]),
        SensitivityRun(parameter="beta", values=[0.5, 1.0, 2.0],
                       metric="total_cost", results=[1200, 1245, 1180]),
    ]

    # paper：含 LaTeX 命令的真实风格文本（模拟 writer 输出）
    s.paper = PaperSections(
        abstract="本文研究共享单车潮汐式需求预测与调度优化。采用 GCN+XGBoost 时空联合预测模型，"
                 "MAE 达 2.4 辆。建立 MILP 调度模型，总成本降至 750.5 元。",
        problem_restatement="## 问题分析\n某城市投放 5000 辆单车，分布在 200 个站点。"
                            "决策变量为 $x_{i,\\tau}$（调度量），目标为 $\\min \\sum_i c_i x_i$，"
                            "约束包括容量 $\\sum_i x_i \\leq Q_v$ 和库存 $S_{i,t} \\geq 0$。",
        assumptions="**假设1**：需求服从 Poisson 分布。\n**依据**：历史数据拟合 KS=0.92。\n\n"
                    "**假设2**：调度车行驶速度恒定 20 km/h。\n**依据**：城市道路平均限速。",
        notation="| 符号 | 含义 | 单位 | 类型 |\n|---|---|---|---|\n"
                 "| $x_{i,t}$ | 调度量 | 辆 | 决策变量 |\n"
                 "| $D_{i,t}$ | 需求量 | 辆 | 随机变量 |",
        model_section=(
            "## 基础预测模型\n"
            "采用 XGBoost 梯度提升树捕捉时间依赖。预测形式化为：\n"
            "\\[ \\hat{D}_{i,t}^{out} = f(\\mathbf{X}_{i,t}; \\theta) \\]\n"
            "其中 $\\mathbf{X}_{i,t}$ 是特征向量，$\\theta$ 是模型参数。用 $\\beta$ 表示回归系数，"
            "$\\tau$ 表示分位数水平。参数估计采用 MLE：\n"
            "\\[ \\hat{\\theta} = \\arg\\max \\sum_i \\log p(d_i | \\theta) \\]\n\n"
            "## 优化调度模型\n"
            "调度问题形式化为 MILP：\n"
            "\\[ \\min \\sum_{\\tau=1}^{T} \\sum_{i=1}^{N} \\bigl(c_{pen} \\cdot p_i + c_{wait} \\cdot w_v \\bigr) \\]\n"
            "约束包括调度车容量 $\\sum_i x_i \\leq Q_v$ 和库存平衡 $S_{i,t+1} = S_{i,t} + x_i - D_i$。\n\n"
            "## 动态与不确定性扩展\n"
            "引入 Wasserstein 模糊集 $\\mathcal{B}_{\\epsilon}(\\hat{P})$ 处理需求不确定性。"
            "鲁棒对应为：\n"
            "\\[ \\min \\max_{P \\in \\mathcal{B}_{\\epsilon}} \\mathbb{E}_P[f(x, \\xi)] \\]\n"
            "用 $\\bar{D}_{i,t}$ 表示历史均值，$\\boldsymbol{f}$ 表示调度量向量。"
        ),
        solution=(
            "## 求解算法与流程\n"
            "采用 Gurobi 求解 MILP，复杂度 $O(2^n)$。关键步骤：\n"
            "1. 输入预测值 $\\hat{D}_{i,t}$\n"
            "2. 构建约束矩阵\n"
            "3. 调用 solver 求解\n"
            "4. 输出调度方案\n\n"
            "## 实验设计与对比分析\n"
            "对比 3 个对照方案：无调度、简单平均预测、贪婪启发式。\n"
            "本文方案总成本 750.5 元，显著优于无调度的 1245.3 元。"
        ),
        sensitivity="对参数 $\\alpha$ 和 $\\beta$ 做敏感性分析。"
                    "$\\alpha$ 在 [0.01, 0.5] 范围内，MAE 从 3.2 降至 2.4，敏感性评级为高。"
                    "$\\beta$ 在 [0.5, 2.0] 范围内，总成本变化约 5%，评级为中。",
        conclusion="模型优点：(1) 时空联合预测精度高，MAE=2.4；(2) MILP 求解保证全局最优。"
                   "缺点：(1) 假设 Poisson 到达；(2) 计算复杂度随站点数指数增长。"
                   "推广方向：(1) 网约车派单；(2) 外卖骑手调度。",
        references="[1] Smith J. Bike sharing demand forecasting. Transportation Research, 2023.\n"
                   "[2] Boyd S. Convex Optimization. Cambridge, 2004.\n"
                   "[3] Bertsekas D. Nonlinear Programming. Athena, 2016.",
        keywords="共享单车, 时空预测, 鲁棒优化, 敏感性分析, MILP",
    )

    s.assumptions = [
        Assumption(statement="需求服从 Poisson 分布", rationale="历史数据拟合 KS=0.92"),
        Assumption(statement="调度车速度恒定 20km/h", rationale="城市道路平均限速"),
    ]

    return s


def main():
    import re

    with tempfile.TemporaryDirectory(prefix="quick_pdf_") as output_dir:
        print(f"[1] 构建 mock state（含 LaTeX 命令）...")
        s = build_mock_state(output_dir)

        # 验证 paper 文本里没有 backspace/tab 损坏
        for field in ['model_section', 'solution', 'notation']:
            val = getattr(s.paper, field, '')
            assert chr(8) not in val, f"{field} 含 backspace！"
            assert chr(9) not in val, f"{field} 含 tab！"
        print("    ✓ paper 文本无 backspace/tab 损坏")

        print(f"[2] 运行 table_assembler...")
        result = table_assembler_node(s)
        s.paper = result["paper"]
        warnings = result["table_warnings"]

        # 验证表格注入
        assert "| 符号 | 含义 | 单位 |" in s.paper.notation, "变量表未注入"
        assert "| 参数 | 取值范围 |" in s.paper.sensitivity, "敏感性表未注入"
        assert "| 方案 |" in s.paper.solution, "对比表未注入"
        print(f"    ✓ 变量表 + 敏感性表 + 对比表 均已注入")
        print(f"    ✓ table_warnings: {len(warnings)} 条")

        # 验证对比表含真实数字
        assert "750.5" in s.paper.solution, "对比表缺本文方案数字"
        assert "1245.3" in s.paper.solution, "对比表缺无调度数字"
        assert "980.0" in s.paper.solution, "对比表缺贪婪数字"
        print(f"    ✓ 对比表含 4 个方案真实数字")

        print(f"[3] 运行 latex_node 编译 PDF...")
        latex_result = latex_node(s)

        if latex_result.get("errors"):
            err = latex_result["errors"][0]
            seen = set()
            print("    ✗ 编译失败，错误：")
            for line in err.split("\n"):
                if line.startswith("!") and line not in seen:
                    seen.add(line)
                    print(f"      {line}")
                    if len(seen) >= 5:
                        break
            m = re.search(r"Output written on \S+ \((\d+) page", err)
            if m:
                print(f"      渲染页数: {m.group(1)}")
            return 1

        print(f"    ✓ 编译成功！")

        # 检查 PDF
        pdf_path = Path(output_dir) / "paper.pdf"
        if pdf_path.exists():
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            pages = len(reader.pages)
            size = pdf_path.stat().st_size
            print(f"[4] PDF: {size} bytes, {pages} 页")

            if pages < 5:
                print(f"    ⚠ 页数偏少（<5），可能有问题")
                return 1
            else:
                print(f"    ✓ 页数正常（≥5）")
        else:
            print(f"    ✗ PDF 不存在")
            return 1

        # 检查 paper.md 也生成了
        md_path = Path(output_dir) / "paper.md"
        if md_path.exists():
            md = md_path.read_text(encoding="utf-8")
            assert "关键算法代码" in md or "附录" in md, "附录缺失"
            assert "关键输出摘要" in md or "curated" in md.lower() or len(md) < 100000, "附录可能未精选"
            print(f"[5] paper.md: {len(md)} chars，附录精选 ✓")

        print(f"\n{'='*60}")
        print("✅ 全部通过！PDF 编译链路正常。")
        print(f"{'='*60}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
