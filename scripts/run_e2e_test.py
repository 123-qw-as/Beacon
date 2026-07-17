"""End-to-end pipeline test with real LLM.
Writes progress to runs/e2e_real/pipeline.log for monitoring.
"""
from __future__ import annotations
import json, sys, time, os
from pathlib import Path

os.environ["MATH_AGENT_RAG_ENABLED"] = "0"
os.environ["MATH_AGENT_DEFAULT_MODEL"] = "openai/ocg/mimo-v2.5-pro"
os.environ["MATH_AGENT_STRONG_MODEL"] = "openai/ocg/mimo-v2.5-pro"
os.environ["MATH_AGENT_LLM_TIMEOUT"] = "600"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

out = Path("runs/e2e_real")
out.mkdir(parents=True, exist_ok=True)
log_file = out / "pipeline.log"

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")

log("Starting pipeline...")

from langgraph.checkpoint.sqlite import SqliteSaver
from math_agent.graph import build_graph

spec = json.loads(Path("tests/fixtures/sample_problem.json").read_text(encoding="utf-8"))
initial = {
    "problem": spec["title"] + "\n" + "\n".join(spec["questions"]),
    "background": spec.get("background", ""),
    "questions": spec["questions"],
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
            msg = f"[+{elapsed:.0f}s][step{step}] {node_name}"
            if isinstance(output, dict):
                for k, v in output.items():
                    if hasattr(v, "model_dump"):
                        if hasattr(v, "core_task"):
                            msg += f" | blueprint: {v.core_task[:80]}"
                        elif hasattr(v, "score"):
                            msg += f" | {k}: score={v.score} approved={v.approved}"
                        elif hasattr(v, "purpose"):
                            msg += f" | {k}: {v.purpose[:60]}"
                    elif k == "stage_target":
                        msg += f" | stage_target={v}"
                    elif k == "iteration":
                        msg += f" | iteration={v}"
            log(msg)

    snap = g.get_state(cfg)
    log(f"Pipeline done. Next: {snap.next}")
    log(f"Total time: {time.time()-t_start:.0f}s")

    # Check artifacts
    for fname in ["trace.json", "paper.md", "paper.pdf", "state_summary.json"]:
        p = out / fname
        log(f"Artifact {fname}: {p.stat().st_size if p.exists() else 'NOT FOUND'}")
