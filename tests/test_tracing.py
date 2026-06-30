import json

from math_agent.tracing import Tracer


def test_tracer_logs_llm_event(workdir):
    t = Tracer(thread_id="t1", out_dir=workdir)
    t.log_llm(model="gpt-4o", prompt_tokens=100, completion_tokens=50, latency_ms=1234)
    t.flush()
    rep = json.loads((workdir / "trace.json").read_text(encoding="utf-8"))
    assert rep["thread_id"] == "t1"
    assert rep["llm_calls"] == 1
    assert rep["tokens"]["prompt"] == 100
    assert rep["tokens"]["completion"] == 50


def test_tracer_aggregates_per_model(workdir):
    t = Tracer(thread_id="t1", out_dir=workdir)
    t.log_llm(model="A", prompt_tokens=10, completion_tokens=5, latency_ms=100)
    t.log_llm(model="A", prompt_tokens=20, completion_tokens=10, latency_ms=200)
    t.log_llm(model="B", prompt_tokens=5, completion_tokens=2, latency_ms=50)
    t.flush()
    rep = json.loads((workdir / "trace.json").read_text(encoding="utf-8"))
    assert rep["per_model"]["A"]["calls"] == 2
    assert rep["per_model"]["A"]["prompt_tokens"] == 30
    assert rep["per_model"]["B"]["calls"] == 1


def test_tracer_logs_node_phase(workdir):
    t = Tracer(thread_id="t1", out_dir=workdir)
    with t.node("analyst"):
        pass
    t.flush()
    rep = json.loads((workdir / "trace.json").read_text(encoding="utf-8"))
    assert rep["nodes"][0]["name"] == "analyst"
    assert rep["nodes"][0]["duration_ms"] >= 0


def test_tracer_current_handle_set_get_reset():
    from math_agent.tracing import set_current, get_current, reset_current
    assert get_current() is None
    t = Tracer(thread_id="x", out_dir=".")
    tok = set_current(t)
    try:
        assert get_current() is t
    finally:
        reset_current(tok)
    assert get_current() is None
