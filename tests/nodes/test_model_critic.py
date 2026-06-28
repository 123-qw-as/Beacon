from math_agent.state import MathModelingState, ModelVersion, CriticReport
from math_agent.nodes.model_critic import model_critic_node


def test_model_critic_appends_report(mocker):
    fake = CriticReport(target="modeler", score=8, issues=[], suggestions=[], approved=True)
    mocker.patch("math_agent.nodes.model_critic.complete", return_value=fake)
    s = MathModelingState(problem="p")
    s.model_versions.append(ModelVersion(stage="basic", description="d"))
    delta = model_critic_node(s)
    assert delta["critic_reports"][0].approved is True
    # critic 必须记录所针对的阶段，下游 modeler_node 据此过滤反馈
    assert delta["critic_reports"][0].stage == "basic"


def test_model_critic_tags_stage_from_latest_model(mocker):
    """即便 LLM 未返回 stage 字段，节点也要按当前最新 model 的 stage 强制覆写。"""
    fake = CriticReport(target="modeler", score=6, approved=False)  # stage 默认 None
    mocker.patch("math_agent.nodes.model_critic.complete", return_value=fake)
    s = MathModelingState(problem="p")
    s.model_versions.append(ModelVersion(stage="improved", description="d"))
    delta = model_critic_node(s)
    assert delta["critic_reports"][0].stage == "improved"


def test_model_critic_no_model_raises(mocker):
    import pytest
    s = MathModelingState(problem="p")
    with pytest.raises(ValueError, match="model_versions"):
        model_critic_node(s)
