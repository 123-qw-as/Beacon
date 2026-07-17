"""
Fast end-to-end test: minimize iterations, use fast models.
"""
from __future__ import annotations
import json, sys, time, os
from pathlib import Path

# Tune for speed
os.environ["MATH_AGENT_RAG_ENABLED"] = "0"
os.environ["MATH_AGENT_DEFAULT_MODEL"] = "openai/ocg/mimo-v2.5-pro"
os.environ["MATH_AGENT_STRONG_MODEL"] = "openai/ocg/mimo-v2.5-pro"
os.environ["MATH_AGENT_LLM_TIMEOUT"] = "600"

# Override config to minimize iterations
os.environ["MATH_AGENT_MAX_MODEL_ITERATIONS"] = "0"  # skip modeler retries
os.environ["MATH_AGENT_MAX_WRITER_ITERATIONS"] = "0"
os.environ["MATH_AGENT_MAX_BLUEPRINT_ITERATIONS"] = "1"
os.environ["MATH_AGENT_MAX_CODE_VERIFY_ITERATIONS"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Re-import config so env overrides take effect
import importlib
import math_agent.config
importlib.reload(math_agent.config)
from math_agent.config import (
    MAX_MODEL_ITERATIONS, MAX_WRITER_ITERATIONS,
    MAX_BLUEPRINT_ITERATIONS, MAX_CODE_VERIFY_ITERATIONS,
)
print(f"MAX_MODEL_ITERATIONS={MAX_MODEL_ITERATIONS}", flush=True)
print(f"MAX_WRITER_ITERATIONS={MAX_WRITER_ITERATIONS}", flush=True)
print(f"MAX_BLUEPRINT_ITERATIONS={MAX_BLUEPRINT_ITERATIONS}", flush=True)
print(f"MAX_CODE_VERIFY_ITERATIONS={MAX_CODE_VERIFY_ITERATIONS}", flush=True)

out = Path("runs/e2e_fast")
out.mkdir(parents=True, exist_ok=True)
log_file = out / "pipeline.log"

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")

log("Starting fast pipeline...")

from langgraph.checkpoint.sqlite import SqliteSaver
from math_agent.graph import build_graph

# SUPER simple problem to test end-to-end flow
problem_text = """Simple Bike Sharing Optimization

We have 5000 bikes across 200 stations with 3 trucks (capacity 50 each). 
Goal: minimize cost while meeting demand.
"""
questions = [
    "Build a model to predict hourly demand at each station.",
    "Design an optimal dispatch plan to rebalance bikes.",
]

initial = {
    "problem": problem_text,
    "background": "City bike sharing system with 5000 bikes, 200 stations.",
    "questions": questions,
    "stage_target": "basic",
    "iteration": 0,
    "output_dir": str(out),
    "latex_template": "default",
}
cfg = {"configurable": {"thread_id": "default"}}

t_start = time.time()
with SqliteSaver.from_conn_string(str(out / "checkpoints.sqlite")) as saver:
    g = build_graph(checkpointer=saver, interrupt_before=[])
    step = 0
    for event in g.stream(initial, config=cfg):
        for node_name, output in event.items():
            step += 1
            elapsed = time.time() - t_start
            msg = f"[+{elapsed:.0f}s][step{step}] Node: {node_name}"
            if isinstance(output, dict):
                for k, v in output.items():
                    if hasattr(v, "model_dump"):
                        if hasattr(v, "core_task"):
                            msg += f" | blueprint core_task={v.core_task[:80]}"
                        elif hasattr(v, "score"):
                            msg += f" | {k}: score={v.score} approved={v.approved}"
                        elif hasattr(v, "overall"):
                            msg += f" | evaluation overall={v.overall}"
                    elif k == "stage_target":
                        msg += f" | stage_target={v}"
                    elif k == "writer_section_queue":
                        msg += f" | sections queued={len(v)}"
                    elif isinstance(v, list) and v:
                        msg += f" | {k}: count={len(v)}"
            log(msg)

    snap = g.get_state(cfg)
    log(f"Pipeline done. Next: {snap.next}")
    log(f"Total time: {time.time()-t_start:.0f}s")

    # Print artifacts found
    for fname in ["trace.json", "paper.md", "paper.pdf", "state_summary.json"]:
        p = out / fname
        if p.exists():
            log(f"✓ Artifact {fname}: {p.stat().st_size} bytes")
        else:
            log(f"✗ Artifact {fname}: NOT FOUND")
    
    # Show code artifacts count
    if snap.values:
        d = snap.values if isinstance(snap.values, dict) else snap.values.__dict__
        ca = d.get('code_artifacts', [])
        log(f"Code artifacts: {len(ca)} ({sum(1 for a in ca if getattr(a,'success',False))} successful)")
        fi = d.get('figures', [])
        log(f"Figures: {len(fi)}")
        pp = d.get('paper', None)
        if pp and hasattr(pp, 'model_dump'):
            log(f"Paper sections present")
