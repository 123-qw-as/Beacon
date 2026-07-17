"""验证昂贵子步骤在进程/网络故障后不会从大节点起点重跑。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from PIL import Image

from math_agent.errors import LLMConnectionError
from math_agent.nodes.coder import (
    CoderDraft, coder_execute_node, coder_generate_node, coder_prepare_node,
)
from math_agent.nodes.modeler import (
    modeler_consistency_node,
    modeler_derivation_node,
    modeler_prepare_node,
)
from math_agent.nodes.figure_pipeline import (
    FigureAnalysisOut,
    FigureCriticOut,
    figure_analysis_node,
    figure_critic_node,
    figure_prepare_node,
)
from math_agent.nodes.sensitivity import (
    Interpretations,
    SensitivityCode,
    SensitivityPlan,
    sensitivity_code_execute_node,
    sensitivity_code_generate_node,
    sensitivity_interpret_node,
    sensitivity_plan_node,
)
from math_agent.prompts.modeler_derivation import ConsistencyCheck
from math_agent.routing import (
    after_coder_work, after_figure_work, after_modeler_work, after_sensitivity_work,
)
from math_agent.state import (
    Assumption,
    CodeArtifact,
    DerivationStep,
    MathModelingState,
    ModelVersion,
    ProblemBlueprint,
)


def _config(name: str) -> dict:
    return {"configurable": {"thread_id": name}}


def test_modeler_recover_keeps_completed_derivations(tmp_path, monkeypatch):
    calls = {"draft": 0, "derive": 0, "check": 0}
    fail_once = {"value": True}

    def fake_complete(_prompt, *, schema, **_kwargs):
        if schema is ModelVersion:
            calls["draft"] += 1
            return ModelVersion(stage="final", description="model")
        if schema is DerivationStep:
            calls["derive"] += 1
            if calls["derive"] == 3 and fail_once["value"]:
                fail_once["value"] = False
                raise LLMConnectionError("injected 502")
            return DerivationStep(title=f"s{calls['derive']}", motivation="m", statement="x", result="r")
        if schema is ConsistencyCheck:
            calls["check"] += 1
            return ConsistencyCheck(coherent=True, issues=[])
        raise AssertionError(schema)

    monkeypatch.setattr("math_agent.nodes.modeler.complete", fake_complete)
    graph = StateGraph(MathModelingState)
    graph.add_node("prepare", modeler_prepare_node)
    graph.add_node("derive", modeler_derivation_node)
    graph.add_node("check", modeler_consistency_node)
    graph.set_entry_point("prepare")
    graph.add_conditional_edges("prepare", after_modeler_work,
                                {"derive": "derive", "check": "check", "done": END})
    graph.add_conditional_edges("derive", after_modeler_work,
                                {"derive": "derive", "check": "check", "done": END})
    graph.add_edge("check", END)

    with SqliteSaver.from_conn_string(str(tmp_path / "modeler.sqlite")) as saver:
        compiled = graph.compile(checkpointer=saver)
        with pytest.raises(LLMConnectionError):
            compiled.invoke({
                "problem": "p", "stage_target": "final",
                "problem_blueprint": ProblemBlueprint(core_task="t"),
            }, config=_config("modeler"))
        snapshot = compiled.get_state(_config("modeler"))
        assert len(snapshot.values["modeler_completed_derivations"]) == 2

        final = compiled.invoke(None, config=_config("modeler"))

    assert calls == {"draft": 1, "derive": 7, "check": 1}
    assert len(final["model_versions"]) == 1
    assert len(final["model_versions"][0].derivation_steps) == 6


def test_coder_recover_does_not_repeat_finished_purposes(tmp_path, monkeypatch):
    prompts: list[str] = []
    fail_once = {"value": True}

    def fake_complete(prompt, **_kwargs):
        prompts.append(prompt)
        if "P3" in prompt and fail_once["value"]:
            fail_once["value"] = False
            raise LLMConnectionError("injected 502")
        purpose = next((p for p in ("P1", "P2", "P3") if p in prompt), "unknown")
        return CoderDraft(
            purpose=purpose,
            code="print('RESULT: baseline=ours total_cost=10 service_rate=0.9')",
        )

    monkeypatch.setattr("math_agent.nodes.coder.complete", fake_complete)
    monkeypatch.setattr("math_agent.nodes.coder._baseline_items", lambda: [])
    monkeypatch.setattr("math_agent.nodes.coder.run_python", lambda *_a, **_k: SimpleNamespace(
        stdout="RESULT: baseline=ours total_cost=10 service_rate=0.9",
        stderr="", success=True, artifact_paths=[], error_kind="",
    ))

    graph = StateGraph(MathModelingState)
    graph.add_node("prepare", coder_prepare_node)
    graph.add_node("generate", coder_generate_node)
    graph.add_node("execute", coder_execute_node)
    graph.set_entry_point("prepare")
    routes = {"generate": "generate", "execute": "execute", "done": END}
    graph.add_conditional_edges("prepare", after_coder_work, routes)
    graph.add_conditional_edges("generate", after_coder_work, routes)
    graph.add_conditional_edges("execute", after_coder_work, routes)

    initial = {
        "problem": "p",
        "model_versions": [ModelVersion(stage="final", description="m",
                                        figure_purposes=["P1", "P2", "P3"])],
        "output_dir": str(tmp_path),
    }
    with SqliteSaver.from_conn_string(str(tmp_path / "coder.sqlite")) as saver:
        compiled = graph.compile(checkpointer=saver)
        with pytest.raises(LLMConnectionError):
            compiled.invoke(initial, config=_config("coder"))
        snapshot = compiled.get_state(_config("coder"))
        assert [a.purpose for a in snapshot.values["coder_work_artifacts"]] == ["P1", "P2"]
        final = compiled.invoke(None, config=_config("coder"))

    assert sum("P1" in prompt for prompt in prompts) == 1
    assert sum("P2" in prompt for prompt in prompts) == 1
    assert len(final["code_artifacts"]) == 3


def test_coder_execution_crash_reuses_checkpointed_draft(tmp_path, monkeypatch):
    calls = {"complete": 0, "run": 0}

    def fake_complete(*_args, **_kwargs):
        calls["complete"] += 1
        return CoderDraft(
            purpose="P1",
            code="print('RESULT: baseline=ours total_cost=10 service_rate=0.9')",
        )

    def fake_run(*_args, **_kwargs):
        calls["run"] += 1
        if calls["run"] == 1:
            raise RuntimeError("injected process kill")
        return SimpleNamespace(
                               stdout="RESULT: baseline=ours total_cost=10 service_rate=0.9",
                               stderr="", success=True,
                               artifact_paths=[], error_kind="")

    monkeypatch.setattr("math_agent.nodes.coder.complete", fake_complete)
    monkeypatch.setattr("math_agent.nodes.coder.run_python", fake_run)
    monkeypatch.setattr("math_agent.nodes.coder._baseline_items", lambda: [])
    graph = StateGraph(MathModelingState)
    graph.add_node("prepare", coder_prepare_node)
    graph.add_node("generate", coder_generate_node)
    graph.add_node("execute", coder_execute_node)
    graph.set_entry_point("prepare")
    routes = {"generate": "generate", "execute": "execute", "done": END}
    graph.add_conditional_edges("prepare", after_coder_work, routes)
    graph.add_conditional_edges("generate", after_coder_work, routes)
    graph.add_conditional_edges("execute", after_coder_work, routes)
    initial = {
        "model_versions": [ModelVersion(stage="final", description="m", figure_purposes=["P1"])],
        "output_dir": str(tmp_path),
    }
    with SqliteSaver.from_conn_string(str(tmp_path / "coder-execute.sqlite")) as saver:
        compiled = graph.compile(checkpointer=saver)
        with pytest.raises(RuntimeError, match="process kill"):
            compiled.invoke(initial, config=_config("coder-execute"))
        snapshot = compiled.get_state(_config("coder-execute"))
        assert snapshot.values["coder_phase"] == "execute"
        assert snapshot.values["coder_pending_draft"]["purpose"] == "P1"
        compiled.invoke(None, config=_config("coder-execute"))

    assert calls == {"complete": 1, "run": 2}


def test_sensitivity_recover_only_retries_interpretation(tmp_path, monkeypatch):
    calls = {"plan": 0, "code": 0, "interpret": 0, "run": 0}

    def fake_complete(_prompt, *, schema, **_kwargs):
        if schema is SensitivityPlan:
            calls["plan"] += 1
            return SensitivityPlan(runs=[{
                "parameter": "x", "values": [1, 2], "metric": "y", "rationale": "r",
            }])
        if schema is None:
            calls["code"] += 1
            return "print('RESULT')"
        if schema is Interpretations:
            calls["interpret"] += 1
            if calls["interpret"] == 1:
                raise LLMConnectionError("injected 502")
            return Interpretations(interpretations=["稳定"])
        raise AssertionError(schema)

    def fake_run(*_args, **_kwargs):
        calls["run"] += 1
        return SimpleNamespace(
            success=True, stdout="RESULT: parameter=x values=[1, 2] results=[2, 4]",
            stderr="", artifact_paths=[], error_kind="",
        )

    monkeypatch.setattr("math_agent.nodes.sensitivity.complete", fake_complete)
    monkeypatch.setattr("math_agent.nodes.sensitivity.run_python", fake_run)
    graph = StateGraph(MathModelingState)
    graph.add_node("plan", sensitivity_plan_node)
    graph.add_node("code_generate", sensitivity_code_generate_node)
    graph.add_node("code_execute", sensitivity_code_execute_node)
    graph.add_node("interpret", sensitivity_interpret_node)
    graph.set_entry_point("plan")
    routes = {"code_generate": "code_generate", "code_execute": "code_execute",
              "interpret": "interpret", "done": END}
    graph.add_conditional_edges("plan", after_sensitivity_work, routes)
    graph.add_conditional_edges("code_generate", after_sensitivity_work, routes)
    graph.add_conditional_edges("code_execute", after_sensitivity_work, routes)
    graph.add_edge("interpret", END)

    initial = {
        "problem": "p", "assumptions": [Assumption(statement="a")],
        "model_versions": [ModelVersion(stage="final", description="m")],
        "output_dir": str(tmp_path),
    }
    with SqliteSaver.from_conn_string(str(tmp_path / "sensitivity.sqlite")) as saver:
        compiled = graph.compile(checkpointer=saver)
        with pytest.raises(LLMConnectionError):
            compiled.invoke(initial, config=_config("sensitivity"))
        snapshot = compiled.get_state(_config("sensitivity"))
        assert len(snapshot.values["sensitivity_pending_runs"]) == 1
        final = compiled.invoke(None, config=_config("sensitivity"))

    assert calls == {"plan": 1, "code": 1, "interpret": 2, "run": 1}
    assert final["sensitivity_runs"][0].interpretation == "稳定"


def test_figure_recover_does_not_repeat_finished_image(tmp_path, monkeypatch):
    p1, p2 = tmp_path / "p1.png", tmp_path / "p2.png"
    Image.new("RGB", (32, 32), "white").save(p1)
    Image.new("RGB", (32, 32), "white").save(p2)
    calls: list[tuple[str, str]] = []
    fail_once = {"value": True}

    def fake_complete(prompt, *, schema, **_kwargs):
        purpose = "P1" if "P1" in prompt else "P2"
        kind = "critic" if schema is FigureCriticOut else "analysis"
        calls.append((purpose, kind))
        if purpose == "P2" and kind == "critic" and fail_once["value"]:
            fail_once["value"] = False
            raise LLMConnectionError("injected 502")
        if schema is FigureCriticOut:
            return FigureCriticOut(score=9, approved=True)
        return FigureAnalysisOut(analysis=f"{purpose} analysis")

    monkeypatch.setattr("math_agent.nodes.figure_pipeline.complete", fake_complete)
    graph = StateGraph(MathModelingState)
    graph.add_node("prepare", figure_prepare_node)
    graph.add_node("critic", figure_critic_node)
    graph.add_node("analysis", figure_analysis_node)
    graph.set_entry_point("prepare")
    graph.add_conditional_edges("prepare", after_figure_work,
                                {"critic": "critic", "analysis": "analysis", "done": END})
    graph.add_conditional_edges("critic", after_figure_work,
                                {"critic": "critic", "analysis": "analysis", "done": END})
    graph.add_conditional_edges("analysis", after_figure_work,
                                {"critic": "critic", "analysis": "analysis", "done": END})
    initial = {"code_artifacts": [
        CodeArtifact(purpose="P1", code="", success=True, artifact_paths=[str(p1)], batch=1),
        CodeArtifact(purpose="P2", code="", success=True, artifact_paths=[str(p2)], batch=1),
    ]}
    with SqliteSaver.from_conn_string(str(tmp_path / "figure.sqlite")) as saver:
        compiled = graph.compile(checkpointer=saver)
        with pytest.raises(LLMConnectionError):
            compiled.invoke(initial, config=_config("figure"))
        snapshot = compiled.get_state(_config("figure"))
        assert len(snapshot.values["figure_work_results"]) == 1
        final = compiled.invoke(None, config=_config("figure"))

    assert calls.count(("P1", "critic")) == 1
    assert calls.count(("P1", "analysis")) == 1
    assert len(final["figures"]) == 2
