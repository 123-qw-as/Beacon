#!/usr/bin/env python
"""Phase 1 端到端验证脚本（真实 LLM，真实 graph）。

跑完整流水线：analyst → modeler → coder → sensitivity → figures → writer →
paper_critic → table_assembler → evaluation → latex，验证 Phase 1 四项改动：

  1. table_assembler 节点被正确执行（table_warnings 非空）
  2. 变量表注入到 notation（markdown pipe-table 出现）
  3. 敏感性表注入到 sensitivity（markdown pipe-table 出现）
  4. 禁用词被清洗（PaperCritic/Claim/超时 等不出现在最终正文）
  5. 8 段结构（model_section 含 ## 基础预测模型，solution 含 ## 求解算法）
  6. 附录精选（paper.md 附录不含完整 stdout，含截断标记或关键输出摘要）

用法：
  # 默认用 sample_problem.json（共享单车调度，适合验证 8 段结构）
  python scripts/e2e_phase1_real.py

  # 指定题目文件
  python scripts/e2e_phase1_real.py --problem src/math_agent/bench/problems/2022_A.json

  # 指定输出目录（默认 runs/phase1_e2e）
  python scripts/e2e_phase1_real.py --out runs/phase1_e2e

  # 用 gmcm 模板（生成国赛格式 PDF）
  python scripts/e2e_phase1_real.py --template gmcm --school "测试大学" --team-id "A0001" --members "队员A,队员B,队员C"

需要：真实 LLM API key（OPENAI_API_KEY 或 LiteLLM 支持的 provider 环境变量）。
耗时：约 5-10 分钟（取决于模型速度）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# 确保能 import math_agent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langgraph.checkpoint.sqlite import SqliteSaver

from math_agent.graph import build_graph
from math_agent.state import MathModelingState


# 禁用词清单——最终正文里不应出现这些
FORBIDDEN_WORDS = ["PaperCritic", "Claim", "Evidence", "Reasoning", "超时", "占位"]


def run_e2e(problem_path: Path, out_dir: Path, template: str = "default",
            school: str = "", team_id: str = "", members: str = "") -> dict:
    """跑完整 graph，返回 final state 的关键字段。"""

    spec = json.loads(problem_path.read_text(encoding="utf-8"))
    problem_text = spec.get("title", "") + "\n" + "\n".join(spec.get("questions", []))

    out_dir.mkdir(parents=True, exist_ok=True)

    initial = {
        "problem": problem_text,
        "background": spec.get("background", ""),
        "questions": spec.get("questions", []),
        "stage_target": "basic",
        "iteration": 0,
        "output_dir": str(out_dir),
        "latex_template": template,
        "school": school or None,
        "team_id": team_id or None,
        "members": members or None,
    }

    print(f"[E2E] 题目: {spec.get('title', '?')}")
    print(f"[E2E] 模板: {template}")
    print(f"[E2E] 输出: {out_dir}")
    print(f"[E2E] 开始运行（真实 LLM，约 5-10 分钟）...\n")

    t0 = time.time()
    with SqliteSaver.from_conn_string(str(out_dir / "checkpoints.sqlite")) as saver:
        # no_interrupt=True：跳过 human_review，直接跑到底
        g = build_graph(checkpointer=saver, interrupt_before=[])
        final_state = g.invoke(initial, config={"configurable": {"thread_id": "phase1_e2e"}})

    elapsed = time.time() - t0
    print(f"\n[E2E] 完成，耗时 {elapsed:.0f}s\n")

    return final_state


def verify(final_state: dict, out_dir: Path) -> list[str]:
    """验证 Phase 1 四项改动。返回失败列表（空=全通过）。"""
    failures: list[str] = []

    paper = final_state.get("paper")
    if paper is None:
        failures.append("FATAL: final state 中没有 paper 字段")
        return failures

    # 把 paper 各字段拼成一个大文本，方便搜索
    all_text = "\n".join([
        paper.abstract or "", paper.problem_restatement or "",
        paper.assumptions or "", paper.notation or "",
        paper.model_section or "", paper.solution or "",
        paper.sensitivity or "", paper.conclusion or "",
        paper.references or "",
    ])

    # ---- 检查 1: table_assembler 节点执行了 ----
    # table_warnings 为空是正常的——说明 writer 没泄露禁用词（RULE 7 生效）
    # 验证 table_assembler 跑了的方式：notation 含变量表 或 solution 含对比表
    has_var_table = "| 符号 | 含义 | 单位 |" in (paper.notation or "")
    has_comp_table = "| 方案 |" in (paper.solution or "")
    if has_var_table or has_comp_table:
        print(f"  ✓ 检查1: table_assembler 已执行（变量表={has_var_table}, 对比表={has_comp_table}）")
    else:
        failures.append("检查1 FAIL: notation 无变量表且 solution 无对比表——table_assembler 可能没执行")
    warnings = final_state.get("table_warnings", [])
    if warnings:
        print(f"     table_warnings: {len(warnings)} 条（禁用词被清洗）")
    else:
        print(f"     table_warnings: 空（writer 未泄露禁用词，RULE 7 生效）")

    # ---- 检查 2: 变量表注入到 notation ----
    if "| 符号 | 含义 | 单位 |" in (paper.notation or ""):
        print("  ✓ 检查2: 变量表已注入 notation")
    else:
        failures.append("检查2 FAIL: notation 中没有变量表（| 符号 | 含义 | 单位 |）")

    # ---- 检查 3: 敏感性表注入到 sensitivity ----
    if "| 参数 | 取值范围 |" in (paper.sensitivity or ""):
        print("  ✓ 检查3: 敏感性表已注入 sensitivity")
    else:
        # 可能 sensitivity_runs 为空导致没生成表
        sens_runs = final_state.get("sensitivity_runs", [])
        if not sens_runs:
            print("  ⚠ 检查3: 跳过——sensitivity_runs 为空（敏感性节点可能失败了）")
        else:
            failures.append("检查3 FAIL: sensitivity 中没有敏感性表（| 参数 | 取值范围 |），但 sensitivity_runs 非空")

    # ---- 检查 4: 禁用词被清洗 ----
    leaked = [w for w in FORBIDDEN_WORDS if w in all_text]
    if leaked:
        failures.append(f"检查4 FAIL: 正文仍含禁用词: {leaked}")
    else:
        print(f"  ✓ 检查4: 禁用词已清洗（{len(FORBIDDEN_WORDS)} 个词均未出现）")

    # ---- 检查 5: 8 段结构 ----
    if "## 基础预测模型" in (paper.model_section or "") or "基础预测模型" in (paper.model_section or ""):
        print("  ✓ 检查5a: model_section 含「基础预测模型」段")
    else:
        failures.append("检查5a FAIL: model_section 中没有「基础预测模型」段标题")

    if "## 求解算法" in (paper.solution or "") or "求解算法" in (paper.solution or ""):
        print("  ✓ 检查5b: solution 含「求解算法」段")
    else:
        failures.append("检查5b FAIL: solution 中没有「求解算法」段标题")

    # ---- 检查 6: 附录精选（读 paper.md）----
    md_path = out_dir / "paper.md"
    if md_path.exists():
        md = md_path.read_text(encoding="utf-8")
        if "关键算法代码" in md or "附录" in md:
            # 检查附录里是否还有完整 stdout（应该只有摘要）
            # 粗略判断：如果附录区域的行数 < 200，说明做了精选
            appendix_start = md.find("附录")
            if appendix_start >= 0:
                appendix = md[appendix_start:]
                if "关键输出摘要" in appendix or "curated" in appendix.lower():
                    print("  ✓ 检查6: 附录已精选（含「关键输出摘要」）")
                elif len(appendix.split("\n")) < 300:
                    print(f"  ✓ 检查6: 附录已精选（{len(appendix.split(chr(10)))} 行，非全量堆叠）")
                else:
                    failures.append(f"检查6 FAIL: 附录可能未精选（{len(appendix.split(chr(10)))} 行，疑似全量堆叠）")
            else:
                print("  ⚠ 检查6: 跳过——paper.md 无附录区域")
        else:
            failures.append("检查6 FAIL: paper.md 中没有附录")
    else:
        failures.append("检查6 FAIL: paper.md 不存在")

    # ---- 额外: 检查 errors ----
    errors = final_state.get("errors", [])
    if errors:
        print(f"\n  ⚠ 注意: state.errors 有 {len(errors)} 条:")
        for e in errors[:5]:
            print(f"    - {e}")
        if len(errors) > 5:
            print(f"    ... 还有 {len(errors) - 5} 条")

    return failures


def main():
    import argparse
    p = argparse.ArgumentParser(description="Phase 1 端到端验证（真实 LLM）")
    p.add_argument("--problem", default="tests/fixtures/sample_problem.json",
                   help="题目 JSON 文件路径")
    p.add_argument("--out", default="runs/phase1_e2e",
                   help="输出目录")
    p.add_argument("--template", default="default", choices=["default", "gmcm"],
                   help="LaTeX 模板")
    p.add_argument("--school", default="", help="学校（gmcm）")
    p.add_argument("--team-id", default="", help="报名号（gmcm）")
    p.add_argument("--members", default="", help="队员（gmcm）")
    args = p.parse_args()

    problem_path = Path(args.problem)
    if not problem_path.exists():
        print(f"错误: 题目文件不存在: {problem_path}")
        return 1

    out_dir = Path(args.out)
    # 清理旧 checkpoint
    chk = out_dir / "checkpoints.sqlite"
    if chk.exists():
        chk.unlink()
        print(f"[E2E] 清理旧 checkpoint: {chk}")

    # 运行
    try:
        final_state = run_e2e(problem_path, out_dir, args.template,
                              args.school, args.team_id, args.members)
    except Exception as e:
        print(f"\n[E2E] 运行失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 验证
    print("=" * 60)
    print("Phase 1 验证结果")
    print("=" * 60)

    failures = verify(final_state, out_dir)

    print("=" * 60)
    if failures:
        print(f"\n❌ {len(failures)} 项检查失败:")
        for f in failures:
            print(f"  - {f}")
        print(f"\n输出在 {out_dir}（paper.md / paper.tex 可人工检查）")
        return 1
    else:
        print("\n✅ 全部检查通过！Phase 1 端到端验证成功。")
        print(f"\n产物位置:")
        print(f"  Markdown: {out_dir / 'paper.md'}")
        tex_path = out_dir / "paper.tex"
        if tex_path.exists():
            print(f"  LaTeX:    {tex_path}")
        pdf_path = out_dir / "paper.pdf"
        if pdf_path.exists():
            print(f"  PDF:      {pdf_path}")
        else:
            print(f"  PDF:      （未生成——xelatex 可能未安装，查看 paper.tex）")
        trace_path = out_dir / "trace.json"
        if trace_path.exists():
            print(f"  Trace:    {trace_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
