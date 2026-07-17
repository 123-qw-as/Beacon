"""latex 节点：渲染 .tex → 编译 .pdf → 失败时回退到 Markdown。"""
from __future__ import annotations

import shutil
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

from math_agent.nodes.latex_transform import (
    _prepare_section, _prepare_inline_text, _prepare_title, _gmcm_bibliography,
)
from math_agent.nodes.rendering import (
    _curate_code, _curate_stdout, _latex_path, _truncate_caption, _latex_plain_text,
)
from math_agent.nodes.sensitivity import _render_verified_figure
from math_agent.nodes.writer import render_markdown, _has_green_safe_solver
from math_agent.state import FigureArtifact, MathModelingState, PaperSections, SensitivityRun
from math_agent.tools.latex_compile import compile_latex
from math_agent.tools.runner import extract_valid_result_lines, infer_entity_upper_bound

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=select_autoescape([]))


def _latest_sensitivity_runs(state: MathModelingState) -> list[SensitivityRun]:
    """Select the newest formal run per parameter from append-only history."""
    latest: dict[str, SensitivityRun] = {}
    for run in state.sensitivity_runs:
        latest[run.parameter] = run
    return list(latest.values())


def _formal_figures(
    state: MathModelingState, sensitivity_runs: list[SensitivityRun]
) -> list[FigureArtifact]:
    """Select figures backed by the current formal evidence set."""
    artifact_paths = {
        str(Path(path).resolve())
        for artifact in state.latest_code_artifacts()
        if artifact.success and artifact.evidence_role in {"primary", "supporting"}
        for path in artifact.artifact_paths
    }
    sensitivity_history_paths = {
        str(Path(run.figure_path).resolve())
        for run in state.sensitivity_runs
        if run.figure_path
    }
    by_path: dict[str, FigureArtifact] = {}
    for figure in state.figures:
        path = str(Path(figure.path).resolve())
        if path in sensitivity_history_paths:
            continue
        if not artifact_paths or path in artifact_paths:
            by_path[path] = figure

    figures = list(by_path.values())
    seen = set(by_path)
    for run in sensitivity_runs:
        if not run.figure_path:
            continue
        path = str(Path(run.figure_path).resolve())
        if path in seen:
            continue
        seen.add(path)
        figures.append(FigureArtifact(
            path=run.figure_path,
            purpose=f"敏感性分析：{run.parameter}",
            caption=f"{run.parameter}的单因素敏感性结果",
            analysis="",
        ))
    return figures


def _refresh_verified_cost_figure(state: MathModelingState, figures: list[FigureArtifact]) -> None:
    """Re-render a supporting cost chart strictly from the primary RESULT breakdown."""
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    primary = next((
        artifact for artifact in reversed(state.latest_code_artifacts())
        if artifact.success and artifact.evidence_role == "primary"
        and "BEACON_GREEN_LOGISTICS_SAFE_SOLVER" in artifact.code
    ), None)
    if primary is None:
        return
    match = re.search(
        r"(?m)^BREAKDOWN:\s+Z_fix=([0-9.]+)\s+Z_wait=([0-9.]+)\s+"
        r"Z_late=([0-9.]+)\s+Z_energy=([0-9.]+)\s+Z_carbon=([0-9.]+)",
        primary.stdout,
    )
    target = next((
        figure for figure in figures
        if any(
            marker in Path(figure.path).stem.casefold()
            for marker in ("cost_composition", "cost_pie")
        )
    ), None)
    if match is None or target is None:
        return
    values = [float(value) for value in match.groups()]
    labels = ["固定启动成本", "等待成本", "惩罚成本", "能源成本", "碳税成本"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    fig, ax = plt.subplots(figsize=(10, 6.4), dpi=180)
    wedges, _, _ = ax.pie(
        values, labels=None, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.72,
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
    )
    total = sum(values)
    legend = [
        f"{label}：{value:.2f} 元（{100.0 * value / total:.1f}%）"
        for label, value in zip(labels, values)
    ]
    ax.legend(wedges, legend, loc="center left", bbox_to_anchor=(0.93, 0.5),
              frameon=False, fontsize=10)
    ax.set_title("多约束分割配送模型——成本构成", fontsize=15, pad=14)
    ax.axis("equal")
    fig.subplots_adjust(left=0.04, right=0.74, top=0.90, bottom=0.05)
    fig.savefig(target.path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _verified_comparison_figure(
    state: MathModelingState, workdir: Path
) -> FigureArtifact | None:
    """由通过 RESULT 门禁的主方案与基线生成同口径比较图。"""
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    upper_bound = infer_entity_upper_bound(state.data_files)
    rows: list[tuple[str, dict[str, float]]] = []
    for artifact in state.latest_code_artifacts():
        if not artifact.success or artifact.evidence_role not in {"primary", "baseline"}:
            continue
        expected = (
            artifact.category.split(":", 1)[1]
            if artifact.evidence_role == "baseline" and ":" in artifact.category
            else "ours"
        )
        lines = extract_valid_result_lines(
            artifact.stdout,
            stderr=artifact.stderr,
            expected_identifier=expected,
            max_entity_count=upper_bound,
        )
        if not lines:
            continue
        values = {
            match.group(1): float(match.group(2))
            for match in re.finditer(
                r"([A-Za-z_][\w]*)=(-?\d+(?:\.\d+)?)", lines[0]
            )
        }
        rows.append((expected, values))
    order = {name: index for index, name in enumerate(
        ("ours", "no_schedule", "simple_pred", "greedy")
    )}
    rows.sort(key=lambda item: order.get(item[0], 99))
    if len(rows) < 2 or rows[0][0] != "ours":
        return None

    labels = [name for name, _ in rows]
    costs = np.asarray([item.get("total_cost", 0.0) for _, item in rows])
    carbons = np.asarray([item.get("total_carbon", 0.0) for _, item in rows])
    timewins = np.asarray([item.get("timewin_rate", 0.0) for _, item in rows])
    if costs[0] <= 0 or carbons[0] <= 0:
        return None
    cost_index = 100.0 * costs / costs[0]
    carbon_index = 100.0 * carbons / carbons[0]
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756"][:len(rows)]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=180)
    for ax, values, title, ylabel in (
        (axes[0], cost_index, "总成本指数", "主方案=100"),
        (axes[1], carbon_index, "碳排放指数", "主方案=100"),
        (axes[2], 100.0 * timewins, "时间窗满足率", "%"),
    ):
        bars = ax.bar(labels, values, color=colors, width=0.68)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("主方案与同口径基线比较", fontsize=15)
    fig.tight_layout()
    target = workdir / "baseline_comparison.png"
    fig.savefig(target, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return FigureArtifact(
        path=str(target),
        purpose="主方案与同口径基线比较",
        caption="主方案与三类同口径基线的成本、碳排放和时间窗率比较",
        analysis=(
            "总成本和碳排放采用主方案归一化指数，时间窗率保留百分比。"
            "图中数值全部由通过门禁的正式 RESULT 行计算；指数只用于消除量纲差异，"
            "不能替代原始结果表，也不构成统计显著性检验。"
        ),
    )


def latex_node(state: MathModelingState) -> dict:
    workdir = Path(state.output_dir or ".")
    workdir.mkdir(parents=True, exist_ok=True)

    # ponytail: 为图重打一份 LaTeX 安全视图，避免 caption/path 里的 _/&/% 炸编译
    # caption 尽量截到最近的句/短语边界，避免"曲线呈下降趋势，"这种半截逗号结尾
    # analysis 走完整 _prepare_section（v8 实测：figure_pipeline LLM 在图说里
    # 会写 sensitivity_capacity.png 这种文件名，必须 escape）
    formal_sens = _latest_sensitivity_runs(state)
    if _has_green_safe_solver(state):
        formal_sens = [
            run.model_copy(update={
                "figure_path": _render_verified_figure(
                    run, Path(run.figure_path).resolve().parent
                ) if run.figure_path else run.figure_path,
            })
            for run in formal_sens
        ]
    formal_figures = _formal_figures(state, formal_sens)
    _refresh_verified_cost_figure(state, formal_figures)
    comparison_figure = _verified_comparison_figure(state, workdir)
    if comparison_figure is not None:
        formal_figures.append(comparison_figure)
    safe_figures = [
        FigureArtifact(
            path=_latex_path(f.path),
            purpose=_prepare_inline_text(f.purpose),
            caption=_prepare_inline_text(_truncate_caption(f.caption or f.purpose, max_chars=55)),
            quality_score=f.quality_score,
            quality_issues=list(f.quality_issues),
            analysis=_prepare_section(f.analysis),
        )
        for f in formal_figures
    ]

    # paper 各段做 markdown → LaTeX 的确定性预处理（粗体/表格/列表/标题/数学）
    safe_paper = PaperSections(**{
        k: _prepare_section(v) if isinstance(v, str) else v
        for k, v in state.paper.model_dump().items()
    })
    safe_sens = [
        SensitivityRun(
            parameter=_latex_plain_text(r.parameter) or "",
            values=r.values, metric=r.metric,
            results=r.results,
            # The verified green-logistics paper already derives its complete
            # sensitivity interpretation from the numeric arrays. Do not
            # append older free-form model prose a second time.
            interpretation=(
                "" if _has_green_safe_solver(state)
                else _prepare_section(r.interpretation)
            ),
            figure_path=r.figure_path,
        )
        for r in formal_sens
    ]

    # title 取 problem 第一行（避免把整段问题描述塞进 \title{}）
    title_line = (
        "多约束异构车队绿色配送与动态局部重调度"
        if _has_green_safe_solver(state)
        else state.problem.split("\n", 1)[0].strip()
    )

    # 选模板：default 用 article 简版；gmcm 用 gmcmthesis 国赛规范
    use_gmcm = state.latex_template == "gmcm"
    tmpl_name = "gmcm.tex.j2" if use_gmcm else "paper.tex.j2"
    if use_gmcm:
        safe_paper.references = _gmcm_bibliography(safe_paper.references)

    upper_bound = infer_entity_upper_bound(state.data_files)
    primary_artifacts = [
        artifact
        for artifact in state.latest_code_artifacts()
        if (
            artifact.success
            and artifact.evidence_role == "primary"
            and extract_valid_result_lines(
                artifact.stdout,
                stderr=artifact.stderr,
                max_entity_count=upper_bound,
            )
        )
    ]

    render_kwargs = dict(
        # 标题保留数学段，并完整转义纯文本字符。
        problem=_prepare_title(title_line),
        paper=safe_paper, figures=safe_figures, sensitivity_runs=safe_sens,
        code_artifacts=[
            {
                "purpose": _prepare_inline_text(a.purpose), "code": a.code, "stdout": a.stdout,
                "success": a.success, "artifact_paths": a.artifact_paths,
                "curated_code": _curate_code(a.code, max_lines=55),
                "curated_stdout": _curate_stdout(a.stdout),
            }
            for a in primary_artifacts
        ],
    )
    if use_gmcm:
        # 队员逗号拆分到 a/b/c，空位补占位
        mem = (state.members or "").split(",")
        mem = [m.strip() for m in mem] + ["", "", ""]
        render_kwargs.update(
            school=_latex_plain_text(state.school or "XX大学"),
            team_id=_latex_plain_text(state.team_id or "No.00000001"),
            member_a=_latex_plain_text(mem[0] or "队员A"),
            member_b=_latex_plain_text(mem[1] or "队员B"),
            member_c=_latex_plain_text(mem[2] or "队员C"),
            keywords=_latex_plain_text((state.paper.keywords or "数学建模").strip()),
        )
        # cls 必须和 .tex 在同一目录才能被 xelatex 找到；封面图也要带上
        cls_src = _TEMPLATE_DIR / "gmcmthesis.cls"
        cls_dst = workdir / "gmcmthesis.cls"
        shutil.copyfile(cls_src, cls_dst)
        fig_src = _TEMPLATE_DIR / "gmcm_figures"
        fig_dst = workdir / "figures"
        if fig_src.is_dir():
            fig_dst.mkdir(parents=True, exist_ok=True)
            for f in fig_src.iterdir():
                shutil.copyfile(f, fig_dst / f.name)

    tmpl = _env.get_template(tmpl_name)
    tex = tmpl.render(**render_kwargs)
    tex_path = workdir / "paper.tex"
    tex_path.write_text(tex, encoding="utf-8")

    # 始终也写一份 Markdown，作为降级 / 备查。它必须与 TeX 使用同一组最新正式
    # 图和敏感性证据，不能重新渲染 append-only 历史。绿色物流安全求解器的完整
    # 敏感性解释已经由 paper.sensitivity 从数值数组确定性生成，不重复追加自由文本。
    markdown_sens = [
        run.model_copy(update={
            "interpretation": "" if _has_green_safe_solver(state) else run.interpretation,
        })
        for run in formal_sens
    ]
    (workdir / "paper.md").write_text(
        render_markdown(
            state,
            figures=formal_figures,
            sensitivity_runs=markdown_sens,
            problem_override=title_line,
        ),
        encoding="utf-8",
    )

    res = compile_latex(tex_path)
    if not res.success:
        return {"errors": [f"latex compile failed: {res.log[:500]}"]}
    return {}
