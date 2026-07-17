"""用已验证附件重建内容增强版绿色物流论文，供回归与视觉验收。"""
from __future__ import annotations

import argparse
from pathlib import Path

from math_agent.nodes.coder import (
    _green_logistics_template_code,
    _safe_baseline_draft,
    _safe_solver_model_contract,
    _validated_execution,
)
from math_agent.nodes.finalizer import _pdf_body_metrics
from math_agent.nodes.latex_node import latex_node
from math_agent.nodes.model_code_consistency import _verified_green_contract_report
from math_agent.nodes.sensitivity import (
    SensitivityPlan,
    _build_canonical_replay_code,
    _parse_results,
)
from math_agent.nodes.table_assembler import table_assembler_node
from math_agent.nodes.writer import (
    _verified_abstract_problem,
    _verified_assumptions_notation,
    _verified_conclusion_section,
    _verified_green_references,
    _verified_model_section,
    _verified_sensitivity_section,
    _verified_solution,
)
from math_agent.prompts.coder_baseline import BASELINE_SPECS
from math_agent.state import (
    CodeArtifact,
    FigureArtifact,
    MathModelingState,
    PaperSections,
    SensitivityRun,
)
from math_agent.tools.runner import run_python


SENSITIVITY_RUNS = [
    {
        "parameter": "速度时变函数的比例因子（整体速度水平）",
        "values": [0.8, 0.9, 1.0, 1.1, 1.2],
        "metric": "total_cost",
    },
    {
        "parameter": "绿色区限行时段开始时间（小时）",
        "values": [7.0, 7.5, 8.0, 8.5, 9.0],
        "metric": "total_cost",
    },
    {
        "parameter": "软时间窗单位惩罚成本系数（元/分钟）",
        "values": [0.625, 0.7291667, 0.8333333, 0.9375, 1.0416667],
        "metric": "total_cost",
    },
]


FIGURE_TEXT = {
    "green_delivery_network.png": ("城市绿色物流配送路径", "主方案路径、客户点、配送中心与绿色区的空间关系。"),
    "data_profile.png": ("附件数据画像", "需求、时间窗与绿色区客户分布均来自本次附件读取。"),
    "algorithm_flow.png": ("求解与验证流程", "展示构造、2-opt、硬约束审计以及随机和动态实验。"),
    "dynamic_stress.png": ("动态局部重插压力测试", "展示距离、晚到和响应时间的独立事件样本分布。"),
    "robustness_diagnostics.png": ("随机交通稳健性", "展示固定路线在蒙特卡洛交通情景下的服务率、晚到和成本分布。"),
    "service_diagnostics.png": ("服务与线路资源诊断", "展示载重、容积利用率及违约任务晚到强度。"),
}


def _artifact(purpose: str, code: str, result, *, category: str, role: str) -> CodeArtifact:
    return CodeArtifact(
        purpose=purpose,
        code=code,
        stdout=result.stdout,
        stderr=result.stderr,
        success=result.success,
        artifact_paths=result.artifact_paths if role == "primary" else [],
        read_paths=result.read_paths,
        category=category,
        evidence_role=role,
        batch=1,
    )


def build(source_state: Path, data_dir: Path, out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"输出目录非空，拒绝覆盖：{out}")
    out.mkdir(parents=True, exist_ok=True)
    state = MathModelingState.model_validate_json(source_state.read_text(encoding="utf-8"))
    state.output_dir = str(out.resolve())
    state.data_dir = str(data_dir.resolve())
    state.latex_template = "gmcm"
    for info in state.data_files:
        info.path = str((data_dir / info.filename).resolve())

    main_code = _green_logistics_template_code(str(data_dir))
    expected = [data_dir / name for name in ("订单信息.xlsx", "距离矩阵.xlsx", "时间窗.xlsx", "客户坐标信息.xlsx")]
    main_result = run_python(
        main_code,
        workdir=out / "executions" / "primary",
        timeout=240,
        expected_input_paths=expected,
    )
    if not main_result.success:
        raise RuntimeError(main_result.stderr)
    main_valid, main_reason, _ = _validated_execution(
        state,
        {"kind": "figure", "category": "figure"},
        main_result,
        code=main_code,
        require_data_usage=True,
    )
    if not main_valid:
        raise RuntimeError(f"主方案未通过正式证据门禁：{main_reason}")
    primary = _artifact("内容增强主方案", main_code, main_result, category="figure", role="primary")

    baselines: list[CodeArtifact] = []
    for name, category, _ in BASELINE_SPECS:
        draft = _safe_baseline_draft({"name": name, "category": category}, main_code)
        if draft is None:
            raise RuntimeError(f"无法生成基线：{category}")
        result = run_python(
            draft.code,
            workdir=out / "executions" / f"baseline_{category}",
            timeout=240,
            expected_input_paths=expected,
        )
        if not result.success:
            raise RuntimeError(f"基线 {category} 失败：{result.stderr}")
        baseline_valid, baseline_reason, _ = _validated_execution(
            state,
            {"kind": "baseline", "category": category},
            result,
            code=draft.code,
            require_data_usage=False,
        )
        if not baseline_valid:
            raise RuntimeError(f"基线 {category} 未通过正式数值门禁：{baseline_reason}")
        baselines.append(_artifact(name, draft.code, result, category=f"baseline:{category}", role="baseline"))
    state.code_artifacts = [primary, *baselines]

    aligned = _safe_solver_model_contract(state, [primary])
    if aligned is not None:
        state.model_versions = [aligned]

    plan = SensitivityPlan(runs=SENSITIVITY_RUNS)
    sensitivity_code = _build_canonical_replay_code(plan, main_code)
    sensitivity_result = run_python(
        sensitivity_code,
        workdir=out / "sensitivity",
        timeout=300,
        expected_input_paths=expected,
    )
    if not sensitivity_result.success:
        raise RuntimeError(f"敏感性分析失败：{sensitivity_result.stderr}")
    parsed = {name: (values, results) for name, values, results in _parse_results(sensitivity_result.stdout)}
    sensitivity_runs: list[SensitivityRun] = []
    for index, entry in enumerate(plan.runs):
        values, results = parsed[entry.parameter]
        sensitivity_runs.append(SensitivityRun(
            parameter=entry.parameter,
            values=values,
            metric=entry.metric,
            results=results,
            figure_path=str((out / "sensitivity" / f"sensitivity_scan_{index}.png").resolve()),
        ))
    state.sensitivity_runs = sensitivity_runs

    figures: list[FigureArtifact] = []
    for raw_path in primary.artifact_paths:
        path = Path(raw_path)
        purpose, analysis = FIGURE_TEXT.get(path.name, (path.stem, "该图来自正式主方案执行。"))
        figures.append(FigureArtifact(
            path=str(path.resolve()), purpose=purpose, caption=purpose,
            quality_score=9, analysis=analysis,
        ))
    state.figures = figures

    state.paper = PaperSections(
        abstract=_verified_abstract_problem(state).abstract,
        problem_restatement=_verified_abstract_problem(state).problem_restatement,
        assumptions=_verified_assumptions_notation(state).assumptions,
        notation=_verified_assumptions_notation(state).notation,
        model_section=_verified_model_section(),
        solution=_verified_solution(state).solution,
        sensitivity=_verified_sensitivity_section(state),
        conclusion=_verified_conclusion_section(state),
        references=_verified_green_references(),
        keywords="异构车辆路径；分割配送；路线内2-opt；蒙特卡洛稳健性；动态事件重调度",
    )
    state.paper = table_assembler_node(state)["paper"]

    report = _verified_green_contract_report(state.latest_model(), [primary], baselines)
    state.model_code_reports = [report] if report is not None else []
    latex_delta = latex_node(state)
    if latex_delta.get("errors"):
        raise RuntimeError("；".join(latex_delta["errors"]))
    total_pages, body_pages, nonempty_pages, body_chars = _pdf_body_metrics(out / "paper.pdf")
    if body_pages < 20 or nonempty_pages != body_pages or body_chars < 15000:
        raise RuntimeError(
            "论文篇幅门禁失败："
            f"total={total_pages}, body={body_pages}, nonempty={nonempty_pages}, chars={body_chars}"
        )
    (out / "final_state.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")
    print(main_result.stdout, end="")
    print(sensitivity_result.stdout, end="")
    print(f"PDF={out / 'paper.pdf'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.source_state, args.data_dir, args.out)


if __name__ == "__main__":
    main()
