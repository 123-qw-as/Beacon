from math_agent.state import MathModelingState, HumanDecision
from math_agent.nodes.human_review import human_review_node


def test_human_review_passes_through_existing_decision():
    s = MathModelingState(problem="p")
    s.human_decision = HumanDecision(approved=True, notes="looks good")
    delta = human_review_node(s)
    assert delta == {}


def test_human_review_records_error_when_no_decision():
    s = MathModelingState(problem="p")
    delta = human_review_node(s)
    assert delta["errors"]
