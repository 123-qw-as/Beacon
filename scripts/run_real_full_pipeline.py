#!/usr/bin/env python3
"""带真实 LLM 的全流程测试 — 带详细进度日志。

用法:
  python scripts/run_real_full_pipeline.py
"""

import argparse
import sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from math_agent.checkpointing import sqlite_saver
from math_agent.graph import build_graph, _wrap as original_wrap
from math_agent.state import HumanDecision


# 全局计时
_T0 = time.time()
_LAST = [_T0]  # 用 list 绕过闭包限制
_COUNTER = [0]


def _logged_wrap(fn, name):
    """包装节点函数，每次被调用时打印耗时。"""
    inner = original_wrap(fn, name)

    def _wrapper(s):
        now = time.time()
        _COUNTER[0] += 1
        total = now - _T0
        step = now - _LAST[0]
        _LAST[0] = now
        print(f"[{total:6.1f}s] #{_COUNTER[0]:2d} {name:25s} +{step:5.1f}s")
        sys.stdout.flush()
        return inner(s)

    _wrapper.__name__ = inner.__name__
    return _wrapper


def main():
    parser = argparse.ArgumentParser(description="运行可恢复的 Beacon 真实完整流程")
    parser.add_argument("--out", type=Path, default=Path("runs/real_full"))
    parser.add_argument("--thread", default="real-full")
    args = parser.parse_args()
    workdir = args.out.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] 输出目录: {workdir}")
    sys.stdout.flush()

    # 注入日志包装器
    import math_agent.graph as mod
    mod._wrap = _logged_wrap

    initial = {
        "problem": (
            "在一个有 N 个节点的连通无向图中，每条边有一个非负权重。"
            "一位旅行者从节点 1 出发，需要访问所有其他节点至少一次，最终回到节点 1。"
            "请帮助旅行者找到一条总权重最小的闭合路径。"
            "分析和求解 TSP 问题的整数规划模型。"
        ),
        "stage_target": "basic",
        "iteration": 0,
        "output_dir": str(workdir),
        "human_decision": HumanDecision(approved=True),
    }

    try:
        config = {"configurable": {"thread_id": args.thread}}
        with sqlite_saver(workdir / "checkpoints.sqlite") as saver:
            g = build_graph(checkpointer=saver, interrupt_before=[])
            snapshot = g.get_state(config)
            if snapshot is not None and snapshot.values:
                print(f"[recover] 从已有 checkpoint 继续，下一节点: {snapshot.next}")
                final = g.invoke(None, config=config)
            else:
                final = g.invoke(initial, config=config)
        elapsed = time.time() - _T0
        print(f"\n[done] 全流程完成，耗时 {elapsed:.1f}s")
        print(f"模型版本: {len(final.get('model_versions', []))}")
        print(f"代码制品: {len(final.get('code_artifacts', []))}")
        print(f"审校报告: {len(final.get('critic_reports', []))}")

        tex = workdir / "paper.tex"
        md = workdir / "paper.md"
        print(f"\npaper.tex: {tex.exists()}, {tex.stat().st_size if tex.exists() else 0}B")
        print(f"paper.md:  {md.exists()}, {md.stat().st_size if md.exists() else 0}B")

        errors = final.get("errors", [])
        if errors:
            print(f"\n[ERROR] {len(errors)} 个错误:")
            for e in errors:
                print(f"  - {e}")
            return False
        print("\n[SUCCESS] 全流程通过")
        return True
    except Exception as e:
        elapsed = time.time() - _T0
        print(f"\n[FAIL] 耗时 {elapsed:.1f}s, 异常退出")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
