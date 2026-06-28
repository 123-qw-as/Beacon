from math_agent.state import (
    MathModelingState,
    ModelVersion,
    CodeArtifact,
    PaperSections,
    Assumption,
    SensitivityRun,
    FigureArtifact,
)
from math_agent.nodes.writer import writer_node, render_markdown
from math_agent.prompts.writer import build_prompt


def _rich_state() -> MathModelingState:
    s = MathModelingState(problem="共享单车调度优化")
    s.assumptions.extend([
        Assumption(statement="需求服从 Poisson 分布", rationale="日志拟合 KS=0.92"),
        Assumption(statement="车辆调度成本线性", rationale="厂商提供的运维台账"),
        Assumption(statement="站点容量上界 50", rationale="物理桩位实测"),
    ])
    s.model_versions.append(ModelVersion(
        stage="final",
        description="带容量约束的时变需求 M/M/c 排队网络",
        equations=[r"\lambda_i(t) = \alpha_i + \beta_i \sin(\omega t)"],
        variables={"lambda": "到达率", "mu": "服务率"},
        notes="相对 improved 引入了时变到达",
    ))
    s.code_artifacts.append(CodeArtifact(
        purpose="求解最优调度", code="...", success=True,
        stdout="A" * 1000 + "FINAL_RESULT=42.7",
    ))
    s.code_artifacts.append(CodeArtifact(
        purpose="验证收敛", code="...", success=True,
        stdout="B" * 1000 + "CONVERGED_AT_ITER=18",
    ))
    s.sensitivity_runs.append(SensitivityRun(
        parameter="alpha", values=[0.1, 0.2, 0.3], metric="total_cost",
        results=[100.0, 120.0, 155.0], interpretation="成本对 alpha 高敏感",
    ))
    s.figures.append(FigureArtifact(
        path="figs/f1.png", purpose="敏感性曲线",
        caption="alpha-cost 曲线", analysis="alpha 超过 0.25 后成本陡增 30%",
    ))
    return s


def test_writer_fills_paper(mocker):
    fake = PaperSections(
        abstract="a"*200, problem_restatement="b"*200, assumptions="c"*200,
        notation="d"*200, model_section="e"*200, solution="f"*200,
        sensitivity="s"*200, conclusion="g"*200, references="h",
    )
    mocker.patch("math_agent.nodes.writer.complete", return_value=fake)
    s = MathModelingState(problem="p")
    s.model_versions.append(ModelVersion(stage="final", description="d"))
    s.code_artifacts.append(CodeArtifact(purpose="x", code="c", success=True, stdout="42"))
    delta = writer_node(s)
    assert isinstance(delta["paper"], PaperSections)
    assert delta["paper"].abstract.startswith("a")


def test_render_markdown_contains_sections():
    s = MathModelingState(problem="P")
    s.paper = PaperSections(abstract="A", problem_restatement="B", assumptions="C",
                            notation="D", model_section="E", solution="F",
                            sensitivity="S", conclusion="H", references="I")
    s.code_artifacts.append(CodeArtifact(purpose="x", code="print(1)", success=True, stdout="1"))
    md = render_markdown(s)
    assert "## 摘要" in md and "## 7. 模型评价" in md
    assert "## 6. 敏感性分析" in md
    assert "print(1)" in md


# ---- build_prompt: 上游素材完整传递 ----

def test_prompt_includes_each_assumption_with_rationale():
    s = _rich_state()
    p = build_prompt(s)
    for a in s.assumptions:
        assert a.statement in p
        assert a.rationale in p


def test_prompt_keeps_per_artifact_stdout_tail():
    s = _rich_state()
    p = build_prompt(s)
    assert "FINAL_RESULT=42.7" in p
    assert "CONVERGED_AT_ITER=18" in p


def test_prompt_includes_sensitivity_numbers_and_interpretation():
    s = _rich_state()
    p = build_prompt(s)
    assert "alpha" in p
    assert "[0.1, 0.2, 0.3]" in p
    assert "[100.0, 120.0, 155.0]" in p
    assert "成本对 alpha 高敏感" in p


def test_prompt_includes_figure_analysis():
    s = _rich_state()
    p = build_prompt(s)
    assert "alpha 超过 0.25 后成本陡增 30%" in p


# ---- build_prompt: IRON RULES 与字数预算 ----

def test_prompt_contains_iron_rules():
    p = build_prompt(_rich_state())
    assert "IRON RULES" in p
    assert "禁编造数据" in p
    assert "禁占位" in p


def test_prompt_contains_word_budget_per_section():
    p = build_prompt(_rich_state())
    # 关键预算锚点
    assert "250–400" in p          # abstract
    assert "800–1500" in p         # model_section
    assert "300–600" in p          # sensitivity


def test_prompt_contains_chinese_style_blocklist():
    p = build_prompt(_rich_state())
    assert "深入探讨" in p
    assert "至关重要" in p
    assert "众所周知" in p
