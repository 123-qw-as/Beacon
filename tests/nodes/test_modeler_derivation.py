"""Plan D Phase 4：modeler 推导链测试。

final 阶段：1 base + 6 derivation + 1 consistency = 8 次 complete。
basic / improved：不跑推导。
"""
from math_agent.state import MathModelingState, Assumption, ModelVersion, DerivationStep, ProblemBlueprint
from math_agent.nodes.modeler import modeler_node


def _state_final():
    s = MathModelingState(problem="p", stage_target="final")
    s.problem_blueprint = ProblemBlueprint(core_task="test")
    s.assumptions.append(Assumption(statement="a", rationale="r"))
    return s


def _state_stage(stage):
    s = MathModelingState(problem="p", stage_target=stage)
    s.problem_blueprint = ProblemBlueprint(core_task="test")
    s.assumptions.append(Assumption(statement="a", rationale="r"))
    return s


def test_modeler_final_stage_runs_derivation_chain(mocker):
    """final stage: 1 base call + 6 derivation steps + 1 consistency = 8 complete calls."""
    base_model = ModelVersion(stage="final", description="d" * 200, equations=["x=1"])
    fake_steps = [DerivationStep(title=f"step{i}", motivation="m", statement="s", result="r")
                  for i in range(6)]
    fake_consistency = type("C", (), {"coherent": True, "issues": []})()
    # complete is called: 1 (base) + 6 (steps) + 1 (consistency) = 8 times
    # side_effect returns in order.
    mocker.patch("math_agent.nodes.modeler.complete",
                 side_effect=[base_model] + fake_steps + [fake_consistency])
    s = _state_final()
    delta = modeler_node(s)
    model = delta["model_versions"][0]
    assert len(model.derivation_steps) == 6
    assert all(isinstance(ds, DerivationStep) for ds in model.derivation_steps)


def test_modeler_basic_stage_skips_derivation(mocker):
    """basic stage: only 1 complete call, no derivation."""
    mocker.patch("math_agent.nodes.modeler.complete",
                 return_value=ModelVersion(stage="basic", description="d" * 200))
    s = _state_stage("basic")
    delta = modeler_node(s)
    model = delta["model_versions"][0]
    assert model.derivation_steps == []


def test_modeler_inconsistent_derivation_records_notes(mocker):
    """If consistency check says not coherent, derivation_notes is populated."""
    base_model = ModelVersion(stage="final", description="d" * 200)
    fake_steps = [DerivationStep(title=f"s{i}", motivation="m", statement="s")
                  for i in range(6)]
    fake_consistency = type("C", (), {"coherent": False, "issues": ["step3 与 step1 矛盾"]})()
    mocker.patch("math_agent.nodes.modeler.complete",
                 side_effect=[base_model] + fake_steps + [fake_consistency])
    s = _state_final()
    delta = modeler_node(s)
    model = delta["model_versions"][0]
    assert "step3" in model.derivation_notes or "矛盾" in model.derivation_notes


def test_modeler_derivation_feeds_completed_steps(mocker):
    """Each derivation step prompt includes the previously completed steps."""
    base_model = ModelVersion(stage="final", description="d" * 200, equations=["x=1"])
    fake_steps = [DerivationStep(title=f"s{i}", motivation=f"m{i}", statement=f"st{i}", result=f"r{i}")
                  for i in range(6)]
    fake_consistency = type("C", (), {"coherent": True, "issues": []})()
    spy = mocker.patch("math_agent.nodes.modeler.complete",
                       side_effect=[base_model] + fake_steps + [fake_consistency])
    s = _state_final()
    modeler_node(s)
    # The 3rd derivation call (index 3 in spy, since 0=base, 1=step1, 2=step2, 3=step3)
    # should contain step1 and step2's results in its prompt
    third_step_prompt = spy.call_args_list[3].args[0]
    assert "s0" in third_step_prompt  # step1 title
    assert "s1" in third_step_prompt  # step2 title


def test_modeler_final_consistent_derivation_notes_empty(mocker):
    """When consistency check passes, derivation_notes stays empty (default)."""
    base_model = ModelVersion(stage="final", description="d" * 200)
    fake_steps = [DerivationStep(title=f"s{i}", motivation="m", statement="s", result="r")
                  for i in range(6)]
    fake_consistency = type("C", (), {"coherent": True, "issues": []})()
    mocker.patch("math_agent.nodes.modeler.complete",
                 side_effect=[base_model] + fake_steps + [fake_consistency])
    s = _state_final()
    delta = modeler_node(s)
    model = delta["model_versions"][0]
    assert model.derivation_notes == ""


def test_modeler_improved_stage_skips_derivation(mocker):
    """improved stage: only 1 complete call, no derivation."""
    mocker.patch("math_agent.nodes.modeler.complete",
                 return_value=ModelVersion(stage="improved", description="d" * 200))
    s = _state_stage("improved")
    delta = modeler_node(s)
    model = delta["model_versions"][0]
    assert model.derivation_steps == []
    assert model.derivation_notes == ""
