#!/usr/bin/env python3
"""历史 checkpoint 应急迁移工具。

新运行由图内 finalizer 自动收口；本脚本只用于旧版本已经结束、无法经 graph recover
进入 finalizer 的运行。它不会重写原 checkpoint 历史。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from math_agent.checkpointing import sqlite_saver
from math_agent.nodes.evaluation import evaluation_node
from math_agent.nodes.finalizer import finalizer_node
from math_agent.nodes.latex_node import latex_node
from math_agent.nodes.paper_critic import paper_critic_node
from math_agent.nodes.sensitivity import _fallback_interpretation
from math_agent.nodes.table_assembler import table_assembler_node
from math_agent.state import MathModelingState


def _load_state(out: Path, thread: str) -> MathModelingState:
    with sqlite_saver(out / "checkpoints.sqlite") as saver:
        snapshot = saver.get_tuple({"configurable": {"thread_id": thread}})
    if snapshot is None or snapshot.checkpoint is None:
        raise ValueError(f"未找到 thread={thread} 的 checkpoint")
    return MathModelingState.model_validate(snapshot.checkpoint["channel_values"])


def _repair_sensitivity(state: MathModelingState) -> tuple[MathModelingState, dict]:
    repaired = state.model_copy(deep=True)
    before_runs = len(repaired.sensitivity_runs)
    before_pending = len(repaired.sensitivity_pending_runs)
    before_errors = list(repaired.errors)

    if not repaired.sensitivity_runs and repaired.sensitivity_pending_runs:
        new_runs = []
        for run in repaired.sensitivity_pending_runs:
            item = run.model_copy(deep=True)
            if not (item.interpretation or "").strip():
                item.interpretation = _fallback_interpretation(item)
            new_runs.append(item)
        repaired.sensitivity_runs = new_runs
        repaired.sensitivity_pending_runs = []
        repaired.sensitivity_plan_dump = {}
        repaired.sensitivity_pending_code = ""
        repaired.sensitivity_phase = "done"

    repaired.errors = [
        err for err in repaired.errors
        if not (isinstance(err, str) and err.startswith("sensitivity:") and "解读数量" in err)
    ]

    summary = {
        "before_sensitivity_runs": before_runs,
        "before_sensitivity_pending_runs": before_pending,
        "after_sensitivity_runs": len(repaired.sensitivity_runs),
        "after_sensitivity_pending_runs": len(repaired.sensitivity_pending_runs),
        "before_errors": before_errors,
        "after_errors": list(repaired.errors),
    }
    return repaired, summary


def _apply_override(state: MathModelingState, delta: dict) -> MathModelingState:
    for key, value in delta.items():
        setattr(state, key, value)
    return state


def _apply_append(state: MathModelingState, delta: dict, append_keys: set[str]) -> MathModelingState:
    for key, value in delta.items():
        if key in append_keys:
            current = list(getattr(state, key))
            current.extend(value)
            setattr(state, key, current)
        else:
            setattr(state, key, value)
    return state


def _backup_outputs(out: Path) -> list[str]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backups: list[str] = []
    for name in ("paper.md", "paper.tex", "paper.pdf"):
        src = out / name
        if not src.exists():
            continue
        dst = out / f"{name}.{stamp}.bak"
        dst.write_bytes(src.read_bytes())
        backups.append(str(dst))
    return backups


def main() -> int:
    parser = argparse.ArgumentParser(description="修复已完成运行的末尾状态，并重写论文产物")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--thread", default="default")
    parser.add_argument("--snapshot", type=Path, default=None, help="输出修复后状态 JSON")
    args = parser.parse_args()

    out = args.out.resolve()
    state = _load_state(out, args.thread)
    repaired, repair_summary = _repair_sensitivity(state)

    backups = _backup_outputs(out)

    repaired = _apply_override(repaired, table_assembler_node(repaired))
    repaired = _apply_append(repaired, paper_critic_node(repaired), {"critic_reports", "errors"})
    repaired = _apply_override(repaired, evaluation_node(repaired))
    repaired = _apply_append(repaired, latex_node(repaired), {"errors"})
    repaired = _apply_override(repaired, finalizer_node(repaired))

    snapshot_path = args.snapshot or (out / "final_state_repaired.json")
    snapshot_path.write_text(
        json.dumps(
            {
                "repair_summary": repair_summary,
                "backups": backups,
                "final_errors": repaired.errors,
                "sensitivity_runs": [r.model_dump() for r in repaired.sensitivity_runs],
                "paper": repaired.paper.model_dump(),
                "evaluation": repaired.evaluation.model_dump() if repaired.evaluation else None,
                "critic_reports_count": len(repaired.critic_reports),
                "figures_count": len(repaired.figures),
                "finalization": repaired.finalization.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[repair] 注意：这是旧 checkpoint 应急迁移，不是正常运行入口。")
    print(f"[repair] 输出目录: {out}")
    print(f"[repair] 备份文件数: {len(backups)}")
    print(
        "[repair] sensitivity_runs: "
        f"{repair_summary['before_sensitivity_runs']} -> {repair_summary['after_sensitivity_runs']}"
    )
    print(
        "[repair] sensitivity_pending_runs: "
        f"{repair_summary['before_sensitivity_pending_runs']} -> "
        f"{repair_summary['after_sensitivity_pending_runs']}"
    )
    print(f"[repair] 最终错误数: {len(repaired.errors)}")
    print(f"[repair] 状态快照: {snapshot_path}")
    if repaired.errors:
        for err in repaired.errors:
            print(f"  - {err}")
        return 1
    print("[repair] 成功")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
