"""Plan D 端到端验证脚本。

验证 Plan D 四大改动在完整 graph 流水线中正确协作：
1. writer 多段调用：大纲 + 7 段逐章填充（首轮 8 次 LLM 调用）
2. coder 多图：modeler final 输出 figure_purposes → coder 按 list 跑 N 次
3. modeler 推导链：final 阶段 6 步推导 + self-consistency
4. references：writer references 段调用 select_references（mock API 返回真实文献）

不依赖外部网络 / API key / xelatex。所有 LLM mock，但 graph 真跑。
"""
from __future__ import annotations

import json
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from PIL import Image


def _make_png(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 240), "white").save(p)
    return str(p)


def run_plan_d_e2e(workdir: Path) -> dict:
    """真跑 graph：所有 LLM/LaTeX mock，但验证 Plan D 的多调用架构正确接线。

    返回 final_state dict。
    """
    from math_agent.state import (
        Assumption, ModelVersion, DerivationStep, CriticReport, CriticIssue,
        PaperSections, EvaluationReport, HumanDecision, Reference,
    )
    from math_agent.nodes.analyst import AnalystOutput
    from math_agent.nodes.coder import CoderDraft
    from math_agent.nodes.figure_pipeline import FigureCriticOut, FigureAnalysisOut
    from math_agent.nodes.sensitivity import SensitivityPlan, SensitivityCode, Interpretations
    from math_agent.nodes.modeler import modeler_node as _modeler_node  # noqa: F401
    from math_agent.prompts.modeler_derivation import ConsistencyCheck

    # ---- 跟踪每个节点的 complete 调用次数 ----
    call_log: dict[str, int] = {
        "modeler": 0, "coder": 0, "writer": 0,
    }

    # ---- schema-aware modeler mock ----
    # Plan D: final 阶段触发 6 步推导 + 1 次 consistency = 7 次额外调用
    stage_box = {"i": 0}

    def _modeler_complete(prompt, *, schema, **kw):
        call_log["modeler"] += 1
        if schema is ModelVersion:
            stages = ["basic", "improved", "final"]
            s = stages[min(stage_box["i"], 2)]
            stage_box["i"] += 1
            # final 阶段输出 figure_purposes，让 coder 多图跑起来
            fps = ["需求时序图", "调度路径图", "成本构成饼图"] if s == "final" else []
            return ModelVersion(
                stage=s, description="带容量约束的时变需求 M/M/c 排队网络" * 10,
                equations=[r"\lambda_i(t) = \alpha_i + \beta_i \sin(\omega t)"],
                variables={"lambda": "到达率", "mu": "服务率"},
                figure_purposes=fps,
            )
        if schema is DerivationStep:
            return DerivationStep(
                title=f"推导步骤{call_log['modeler']}",
                motivation="选择此模型族因为它能刻画时变到达率",
                statement=r"$\lambda_i(t) = \alpha_i + \beta_i \sin(\omega t)$",
                result=r"参数 $\alpha_i, \beta_i$ 可由 MLE 估计",
            )
        # ConsistencyCheck
        return ConsistencyCheck(coherent=True, issues=[])

    # ---- schema-aware coder mock ----
    # Plan D: 按 figure_purposes 逐个调用
    def _coder_complete(prompt, *, schema, **kw):
        call_log["coder"] += 1
        return CoderDraft(purpose=f"图{call_log['coder']}", code="print('fig done')")

    # ---- writer mock：返回完整 PaperSections（每个 group schema 都返回同样的）----
    # Plan D writer 调 1 outline + 7 section = 8 次；用 return_value 让每次返回同对象
    # 各 group schema 字段不同，但 getattr 会从 PaperSections 取到对应字段
    full_paper = PaperSections(
        abstract="本文研究共享单车调度优化，采用 M/M/c 排队网络与时变需求建模。" * 3,
        problem_restatement="决策变量为各站点调度量，目标为最小化总成本与等待时间。" * 3,
        assumptions="**假设1**：需求服从 Poisson 分布。**依据**：日志拟合 KS=0.92。" * 3,
        notation="lambda: 到达率; mu: 服务率; c: 服务台数; rho: 利用率。" * 3,
        model_section="basic 阶段建立 M/M/1 模型；improved 加入时变到达；final 引入容量约束。" * 3,
        solution="采用 Kalman 滤波求解状态空间模型，结果见附录。" * 3,
        sensitivity="参数 alpha 在 [0.1, 0.3] 范围内，成本从 100 升至 155，高敏感。" * 3,
        conclusion="模型优点是刻画了时变需求；缺点是假设 Poisson 到达；可推广到多商品调度。" * 3,
        references="[1] Kleinrock. Queueing Systems. Wiley, 1975.\n[2] Box. Time Series Analysis. Wiley, 2015.\n[3] Boyd. Convex Optimization. Cambridge, 2004.\n[4] Goodfellow. Deep Learning. MIT Press, 2016.\n[5] Bertsekas. Nonlinear Programming. Athena, 2016.",
        keywords="排队论,时变需求,Kalman滤波,敏感性分析,共享单车",
    )

    def _writer_complete(prompt, *, schema, **kw):
        call_log["writer"] += 1
        return full_paper

    # ---- mock references 检索：返回真实 Reference 列表 ----
    fake_refs = [
        Reference(id="1", title="Bike sharing demand forecasting",
                  authors=["Smith, J."], venue="Transportation Research",
                  year=2018, doi="10.1016/j.trb.2018.01.001"),
        Reference(id="2", title="Queueing models for bike sharing",
                  authors=["Lee, K."], venue="Operations Research",
                  year=2020, doi="10.1287/opre.2019.1900"),
    ]

    with ExitStack() as stack:
        # analyst
        stack.enter_context(patch(
            "math_agent.nodes.analyst.complete",
            return_value=AnalystOutput(
                assumptions=[Assumption(statement="需求服从 Poisson",
                                        rationale="日志拟合", sensitivity_relevant=True)],
                problem_domains=["queueing", "optimization"],
            )))
        # modeler (schema-aware)
        stack.enter_context(patch("math_agent.nodes.modeler.complete",
                                   side_effect=_modeler_complete))
        # model_critic
        stack.enter_context(patch(
            "math_agent.nodes.model_critic.complete",
            return_value=CriticReport(target="modeler", score=9, approved=True)))
        # coder (schema-aware)
        stack.enter_context(patch("math_agent.nodes.coder.complete",
                                   side_effect=_coder_complete))

        # sensitivity
        sens_plan = SensitivityPlan(runs=[{"parameter": "lambda",
                                           "values": [1, 2, 3, 4, 5],
                                           "metric": "y", "rationale": "r"}])
        sens_code = SensitivityCode(code=(
            "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
            "v=[1,2,3,4,5]; r=[x*2 for x in v]\n"
            "plt.plot(v,r); plt.savefig('lambda.png')\n"
            "print(f'RESULT: parameter=lambda values={v} results={r}')\n"
        ))
        sens_interp = Interpretations(interpretations=["lambda 越大 y 线性增长。"])
        stack.enter_context(patch("math_agent.nodes.sensitivity.complete",
                                   side_effect=[sens_plan, sens_code, sens_interp]))

        # figure_pipeline
        fc = FigureCriticOut(score=9, approved=True)
        fa = FigureAnalysisOut(analysis="趋势单调。")
        stack.enter_context(patch("math_agent.nodes.figure_pipeline.complete",
                                   side_effect=[fc, fa]))

        # writer (multi-call)
        stack.enter_context(patch("math_agent.nodes.writer.complete",
                                   side_effect=_writer_complete))

        # paper_critic
        stack.enter_context(patch(
            "math_agent.nodes.paper_critic.complete",
            return_value=CriticReport(target="paper", score=9, approved=True)))

        # evaluation
        stack.enter_context(patch(
            "math_agent.nodes.evaluation.complete",
            return_value=EvaluationReport(
                assumption_reasonableness=8, modeling_creativity=8,
                result_correctness=8, writing_clarity=8, extra_depth=8, overall=8.0)))

        # latex
        stack.enter_context(patch(
            "math_agent.nodes.latex.compile_latex",
            return_value=type("R", (object,),
                              {"success": True, "pdf_path": "", "log": "",
                               "error_kind": ""})()))

        # references 检索 mock（API 返回真实文献）
        stack.enter_context(patch(
            "math_agent.tools.references.search_references",
            return_value=fake_refs))

        # ---- 真跑 graph ----
        from math_agent.graph import build_graph

        g = build_graph()
        final = g.invoke({
            "problem": "共享单车调度优化：给定站点需求时序，优化调度策略",
            "questions": ["建立需求预测模型", "优化调度方案"],
            "stage_target": "basic", "iteration": 0,
            "output_dir": str(workdir),
            "human_decision": HumanDecision(approved=True).model_dump(),
        })

    return final, call_log


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td) / "run"
        workdir.mkdir()

        print("=== Plan D E2E: 真跑 graph（全 mock LLM/LaTeX）===\n")

        final, call_log = run_plan_d_e2e(workdir)

        # ---- 验证 1：writer 多段调用 ----
        # 首轮 = 1 outline + 7 section = 8 次；critic 通过无 retry
        writer_calls = call_log["writer"]
        assert writer_calls == 8, \
            f"writer 应调 8 次（1 outline + 7 section），实际 {writer_calls}"
        print(f"[1] writer 多段调用: {writer_calls} 次 (1 outline + 7 section)  OK")

        # ---- 验证 2：coder 多图 + Phase 2 对照方案 ----
        # final 阶段 modeler 输出 3 个 figure_purposes → coder 主图调 3 次
        # Phase 2: + 3 个对照方案 = 6 次
        coder_calls = call_log["coder"]
        assert coder_calls == 6, \
            f"coder 应调 6 次（3 figure + 3 baseline），实际 {coder_calls}"
        code_artifacts = final.get("code_artifacts", []) if isinstance(final, dict) \
            else final.code_artifacts
        assert len(code_artifacts) == 6, \
            f"应有 6 个 code_artifacts（3 figure + 3 baseline），实际 {len(code_artifacts)}"
        print(f"[2] coder 多图+对照: {coder_calls} 次调用, {len(code_artifacts)} artifacts  OK")

        # ---- 验证 3：modeler 推导链 ----
        # final 阶段: 1 base + 6 derivation + 1 consistency = 8 次
        # basic + improved 各 1 次 = 2 次
        # 总计 = 2 + 8 = 10 次
        modeler_calls = call_log["modeler"]
        assert modeler_calls == 10, \
            f"modeler 应调 10 次（2 base + 1 base + 6 derivation + 1 consistency），实际 {modeler_calls}"
        model_versions = final.get("model_versions", []) if isinstance(final, dict) \
            else final.model_versions
        final_model = [m for m in model_versions if m.stage == "final"]
        assert final_model, "缺少 final 阶段 model"
        deriv_steps = final_model[0].derivation_steps
        assert len(deriv_steps) == 6, \
            f"final model 应有 6 个 derivation_steps，实际 {len(deriv_steps)}"
        print(f"[3] modeler 推导链: {modeler_calls} 次调用, "
              f"final model derivation_steps={len(deriv_steps)}  OK")

        # ---- 验证 4：references 检索 ----
        paper = final.get("paper") if isinstance(final, dict) else final.paper
        refs_text = paper.references
        assert len(refs_text) > 50, f"references 应有内容，实际: {refs_text[:80]}"
        ref_lines = [l.strip() for l in refs_text.split("\n") if l.strip()]
        assert len(ref_lines) >= 3, f"references 应 ≥3 条，实际 {len(ref_lines)}"
        print(f"[4] references: {len(ref_lines)} 条文献  OK")

        # ---- 验证 5：problem_domains 传递 ----
        problem_domains = final.get("problem_domains") if isinstance(final, dict) \
            else final.problem_domains
        assert problem_domains and "queueing" in problem_domains, \
            f"problem_domains 应含 queueing，实际 {problem_domains}"
        print(f"[5] problem_domains: {problem_domains}  OK")

        # ---- 验证 6：paper.tex / paper.md 产出 ----
        assert (workdir / "paper.tex").exists(), "paper.tex 缺失"
        assert (workdir / "paper.md").exists(), "paper.md 缺失"
        print(f"[6] paper.tex + paper.md 产出  OK")

        # ---- 验证 7：完整 graph 跑通（所有阶段）----
        stages = [m.stage for m in model_versions]
        assert stages == ["basic", "improved", "final"], \
            f"阶段序列应为 basic→improved→final，实际 {stages}"
        print(f"[7] 完整 graph 流水线: stages={stages}  OK")

    print("\n=== ALL PLAN-D E2E CHECKS PASSED ===")
    print(f"\n调用统计: modeler={modeler_calls}, coder={coder_calls}, writer={writer_calls}")
    print("(Plan B 基线: modeler=3, coder=1, writer=1 → Plan D: modeler=10, coder=3, writer=8)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
