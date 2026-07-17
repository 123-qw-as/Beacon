from math_agent.checkpointing import checkpoint_serializer, sqlite_saver
from math_agent.state import Assumption, MathModelingState, PaperSections


def test_checkpoint_serializer_roundtrips_registered_state_types():
    serde = checkpoint_serializer()
    state = MathModelingState(
        problem="p",
        assumptions=[Assumption(statement="a")],
        paper=PaperSections(abstract="x"),
    )

    typed = serde.dumps_typed(state)
    restored = serde.loads_typed(typed)

    assert isinstance(restored, MathModelingState)
    assert isinstance(restored.assumptions[0], Assumption)
    assert isinstance(restored.paper, PaperSections)


def test_sqlite_saver_uses_beacon_serializer(tmp_path):
    with sqlite_saver(tmp_path / "checkpoints.sqlite") as saver:
        typed = saver.serde.dumps_typed(Assumption(statement="safe"))
        restored = saver.serde.loads_typed(typed)
    assert isinstance(restored, Assumption)
