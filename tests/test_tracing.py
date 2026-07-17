import json
import pytest

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


def test_tracer_can_append_existing_trace(workdir):
    first = Tracer(thread_id="t1", out_dir=workdir)
    first.log_llm(model="A", prompt_tokens=10, completion_tokens=5, latency_ms=20)
    with first.node("analyst"):
        pass
    first.flush()

    resumed = Tracer(thread_id="t1", out_dir=workdir, append_existing=True)
    resumed.log_llm(model="A", prompt_tokens=7, completion_tokens=3, latency_ms=10)
    with resumed.node("modeler"):
        pass
    resumed.flush()

    rep = json.loads((workdir / "trace.json").read_text(encoding="utf-8"))
    assert rep["llm_calls"] == 2
    assert rep["tokens"] == {"prompt": 17, "completion": 8}
    assert [n["name"] for n in rep["nodes"]] == ["analyst", "modeler"]


def test_wrapped_node_preserves_failed_node_name():
    import pytest
    from math_agent.graph import _wrap
    from math_agent.tracing import clear_failed_node, get_last_node

    clear_failed_node()
    wrapped = _wrap(lambda _state: (_ for _ in ()).throw(RuntimeError("boom")), "coder")
    with pytest.raises(RuntimeError, match="boom"):
        wrapped(None)
    assert get_last_node() == "coder"
    clear_failed_node()


def test_wrapped_node_attaches_name_across_thread_boundary():
    from concurrent.futures import ThreadPoolExecutor
    from math_agent.graph import _wrap

    wrapped = _wrap(
        lambda _state: (_ for _ in ()).throw(RuntimeError("boom")),
        "modeler",
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(wrapped, None)
        with pytest.raises(RuntimeError) as caught:
            future.result()
    assert getattr(caught.value, "_math_agent_failed_node") == "modeler"


def test_tracer_ignores_valid_json_with_wrong_shape(workdir):
    (workdir / "trace.json").write_text("[]", encoding="utf-8")
    tracer = Tracer(thread_id="t1", out_dir=workdir, append_existing=True)
    tracer.log_llm(model="A", prompt_tokens=1, completion_tokens=2, latency_ms=3)
    assert tracer.llm_calls == 1


def test_tracer_normalizes_partial_model_stats(workdir):
    (workdir / "trace.json").write_text(json.dumps({
        "thread_id": "t1",
        "llm_calls": 1,
        "tokens": None,
        "per_model": {"A": {"calls": 1, "latency_ms": "invalid"}},
        "nodes": "invalid",
    }), encoding="utf-8")
    tracer = Tracer(thread_id="t1", out_dir=workdir, append_existing=True)
    tracer.log_llm(model="A", prompt_tokens=4, completion_tokens=5, latency_ms=6)
    assert tracer.per_model["A"]["calls"] == 2
    assert tracer.per_model["A"]["prompt_tokens"] == 4
