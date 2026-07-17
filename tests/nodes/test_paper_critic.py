from math_agent.state import (
    MathModelingState, PaperSections, FigureArtifact, SensitivityRun, CriticReport,
    CodeArtifact,
)
from math_agent.nodes.paper_critic import paper_critic_node
from math_agent.prompts.paper_critic import build_prompt


def test_paper_critic_appends_report(mocker):
    fake = CriticReport(target="paper", score=8, issues=[], suggestions=[], approved=True)
    mocker.patch("math_agent.nodes.paper_critic.complete", return_value=fake)
    s = MathModelingState(problem="p")
    s.paper = PaperSections(
        abstract="a"*200, problem_restatement="b"*200, assumptions="c"*200,
        notation="d"*200, model_section="e"*200, solution="f"*200,
        sensitivity="g"*200, conclusion="h"*200, references="-",
    )
    s.figures.append(FigureArtifact(path="x.png", purpose="t"))
    s.sensitivity_runs.append(SensitivityRun(
        parameter="a", values=[1], metric="m", results=[1],
    ))
    delta = paper_critic_node(s)
    assert delta["critic_reports"][0].target == "paper"
    assert delta["critic_reports"][0].approved is True


def test_paper_critic_handles_missing_paper(mocker):
    s = MathModelingState(problem="p")
    delta = paper_critic_node(s)
    assert delta["errors"]
    assert delta["critic_reports"][0].score == 0
    assert delta["critic_reports"][0].approved is False


def _paper_with_numbers():
    return PaperSections(
        abstract="目标成本52.6（放缩因子22.12，对应实际成本718）。",
        problem_restatement="x"*200, assumptions="x"*200, notation="x"*200,
        model_section="x"*200, solution="x"*200, sensitivity="x"*200,
        conclusion="x"*200, references="-",
    )


def test_prompt_includes_code_stdout_block():
    """build_prompt 第 4 个形参 code_stdout：注入 stdout 文本块。"""
    real_stdout = "优化成功，目标总成本 = 52.7174\n扰动 +20% → 目标成本 53.7718"
    prompt = build_prompt(_paper_with_numbers(), 0, 0, real_stdout)
    assert "52.7174" in prompt
    assert "53.7718" in prompt
    assert "代码运行" in prompt or "stdout" in prompt.lower()


def test_prompt_omits_stdout_block_when_empty():
    """没有 success=True code_artifact 时不渲染 stdout 区块（避免噪声）。"""
    prompt = build_prompt(_paper_with_numbers(), 0, 0, "")
    assert "代码运行真实输出" not in prompt


def test_paper_critic_node_passes_all_valid_result_evidence(mocker):
    """评审与 writer 使用同一批有效 RESULT，失败或无协议输出均不进入 prompt。"""
    captured = {}

    def _capture(prompt, **kw):
        captured["prompt"] = prompt
        return CriticReport(target="paper", score=7, issues=[], suggestions=[], approved=False)

    mocker.patch("math_agent.nodes.paper_critic.complete", side_effect=_capture)
    s = MathModelingState(problem="p")
    s.paper = _paper_with_numbers()
    s.code_artifacts.append(CodeArtifact(purpose="x", code="...", success=False,
                                          stdout="OLD_FAILED", stderr="error"))
    s.code_artifacts.append(CodeArtifact(purpose="y", code="...", success=True,
                                          stdout="RESULT: baseline=ours total_cost=52.7174 service_rate=0.95", stderr=""))
    s.code_artifacts.append(CodeArtifact(
        purpose="z", code="...", success=True,
        stdout="RESULT: baseline=greedy total_cost=60 service_rate=0.90",
        category="baseline:greedy",
    ))
    delta = paper_critic_node(s)
    assert delta["critic_reports"][0].target == "paper"
    assert "total_cost=52.7174" in captured["prompt"]
    assert "baseline=greedy" in captured["prompt"]
    assert "OLD_FAILED" not in captured["prompt"]
