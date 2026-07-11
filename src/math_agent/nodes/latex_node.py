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
from math_agent.nodes.writer import render_markdown
from math_agent.state import FigureArtifact, MathModelingState, PaperSections, SensitivityRun
from math_agent.tools.latex_compile import compile_latex

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=select_autoescape([]))


def latex_node(state: MathModelingState) -> dict:
    workdir = Path(state.output_dir or ".")
    workdir.mkdir(parents=True, exist_ok=True)

    # ponytail: 为图重打一份 LaTeX 安全视图，避免 caption/path 里的 _/&/% 炸编译
    # caption 尽量截到最近的句/短语边界，避免"曲线呈下降趋势，"这种半截逗号结尾
    # analysis 走完整 _prepare_section（v8 实测：figure_pipeline LLM 在图说里
    # 会写 sensitivity_capacity.png 这种文件名，必须 escape）
    safe_figures = [
        FigureArtifact(
            path=_latex_path(f.path),
            purpose=_prepare_inline_text(f.purpose),
            caption=_prepare_inline_text(_truncate_caption(f.caption or f.purpose, max_chars=55)),
            quality_score=f.quality_score,
            quality_issues=list(f.quality_issues),
            analysis=_prepare_section(f.analysis),
        )
        for f in state.figures
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
            results=r.results, interpretation=_prepare_section(r.interpretation),
            figure_path=r.figure_path,
        )
        for r in state.sensitivity_runs
    ]

    # title 取 problem 第一行（避免把整段问题描述塞进 \title{}）
    title_line = state.problem.split("\n", 1)[0].strip()

    # 选模板：default 用 article 简版；gmcm 用 gmcmthesis 国赛规范
    use_gmcm = state.latex_template == "gmcm"
    tmpl_name = "gmcm.tex.j2" if use_gmcm else "paper.tex.j2"
    if use_gmcm:
        safe_paper.references = _gmcm_bibliography(safe_paper.references)

    render_kwargs = dict(
        # 标题保留数学段，并完整转义纯文本字符。
        problem=_prepare_title(title_line),
        paper=safe_paper, figures=safe_figures, sensitivity_runs=safe_sens,
        code_artifacts=[
            {
                "purpose": _prepare_inline_text(a.purpose), "code": a.code, "stdout": a.stdout,
                "success": a.success, "artifact_paths": a.artifact_paths,
                "curated_code": _curate_code(a.code),
                "curated_stdout": _curate_stdout(a.stdout),
            }
            for a in state.latest_code_artifacts() if a.success
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

    # 始终也写一份 Markdown，作为降级 / 备查
    (workdir / "paper.md").write_text(render_markdown(state), encoding="utf-8")

    res = compile_latex(tex_path)
    if not res.success:
        return {"errors": [f"latex compile failed: {res.log[:500]}"]}
    return {}
