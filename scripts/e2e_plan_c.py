"""Plan C 端到端验证脚本。

测试链路（一次跑透 4 块新增能力）：
1. 真实 sqlite-vec ingest（mock embedding 函数，但真写盘+真查询）
2. 启用 RAG → 真跑 graph：analyst/modeler/writer 真访问 sqlite-vec DB
3. Tracer 真接入 graph + llm.complete，写出 trace.json
4. CLI `report` 子命令真渲染 rich 表格
5. 真触发 LLMRateLimitError → tenacity 重试 → 成功
6. runner/latex error_kind 在真子进程上正确分类

不依赖外部网络 / API key / xelatex。
"""
from __future__ import annotations

import json
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch, MagicMock

from PIL import Image
from typer.testing import CliRunner


# ----- 1. ingest 真写 sqlite-vec -----

def step_ingest(corpus_dir: Path, db_path: Path) -> None:
    """用 mock embedding 函数，真写 sqlite-vec 库。"""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "queue.md").write_text(
        "M/M/c 排队论建模：到达率λ服从Poisson，服务率μ服从指数分布。"
        "稳态条件 ρ=λ/(cμ)<1。常用于交通调度、内涝排水容量评估。",
        encoding="utf-8",
    )
    (corpus_dir / "uav.md").write_text(
        "无人机蜂群覆盖问题常用 Voronoi 划分 + 鲁棒优化。"
        "风扰下采用 chance-constrained programming 提升覆盖率。",
        encoding="utf-8",
    )

    # 用确定性 fake embedding（按文本 hash 生成稳定向量）
    def _fake_embedding(model, input, **kw):
        data = []
        for s in input:
            h = hash(s) % 1000 / 1000.0
            vec = [h, 1 - h, 0.5]
            data.append({"embedding": vec})
        return type("R", (), {"data": data})()

    with patch("litellm.embedding", side_effect=_fake_embedding):
        from math_agent.rag.ingest import ingest_directory
        rep = ingest_directory(
            src_dir=corpus_dir, db_path=db_path,
            embedding_model="text-embedding-3-small", dim=3,
            max_chars=300, overlap=30,
        )
    assert rep.files_processed == 2, rep
    assert rep.chunks_added >= 2, rep
    assert db_path.exists() and db_path.stat().st_size > 0
    print(f"[1] ingest: files={rep.files_processed} chunks={rep.chunks_added} "
          f"db_size={db_path.stat().st_size}B  OK")


# ----- 2. RAG-enabled graph end-to-end with real DB query -----

def step_graph_with_rag(workdir: Path, db_path: Path) -> tuple[dict, list]:
    """真跑 graph：所有 LLM/LaTeX mock，但 RAG 查询走真 sqlite-vec。

    返回 (final_state, rag_search_calls)：rag_search_calls 用来证明 retrieve 真被命中。
    """
    from math_agent.state import (
        Assumption, ModelVersion, CriticReport, PaperSections,
        EvaluationReport, HumanDecision,
    )
    from math_agent.nodes.analyst import AnalystOutput
    from math_agent.nodes.coder import CoderDraft
    from math_agent.nodes.figure_pipeline import FigureCriticOut, FigureAnalysisOut
    from math_agent.nodes.sensitivity import SensitivityPlan, SensitivityCode, Interpretations

    rag_calls: list[str] = []

    # spy on retrieve.search to confirm DB was hit (let it pass through to real impl)
    from math_agent.rag import retrieve as _retrieve_mod
    real_search = _retrieve_mod.search

    def _spy_search(query, **kw):
        rag_calls.append(query[:80])
        return real_search(query, **kw)

    # fake embedding (deterministic, dim=3 matching ingest)
    def _fake_embedding(model, input, **kw):
        data = []
        for s in input:
            h = hash(s) % 1000 / 1000.0
            data.append({"embedding": [h, 1 - h, 0.5]})
        return type("R", (), {"data": data})()

    with ExitStack() as stack:
        # Patch RAG config: real DB, dim=3
        for tgt in ("math_agent.nodes.analyst", "math_agent.nodes.modeler",
                    "math_agent.nodes.writer"):
            stack.enter_context(patch(f"{tgt}.RAG_ENABLED", True))
            stack.enter_context(patch(f"{tgt}.RAG_DB_PATH", str(db_path)))
            stack.enter_context(patch(f"{tgt}.RAG_EMBEDDING_DIM", 3))
            stack.enter_context(patch(f"{tgt}.search", side_effect=_spy_search))

        stack.enter_context(patch("litellm.embedding", side_effect=_fake_embedding))

        # Mock all LLM calls
        stack.enter_context(patch(
            "math_agent.nodes.analyst.complete",
            return_value=AnalystOutput(assumptions=[
                Assumption(statement="排队服从 M/M/c", rationale="日志拟合",
                           sensitivity_relevant=True)])))
        stage_box = {"i": 0}

        def _modeler_side(*a, **k):
            stages = ["basic", "improved", "final"]
            s = stages[min(stage_box["i"], 2)]
            stage_box["i"] += 1
            return ModelVersion(stage=s, description="d" * 200,
                                equations=[r"\lambda < c\mu"])

        stack.enter_context(patch("math_agent.nodes.modeler.complete",
                                   side_effect=_modeler_side))
        stack.enter_context(patch(
            "math_agent.nodes.model_critic.complete",
            return_value=CriticReport(target="modeler", score=9, approved=True)))
        stack.enter_context(patch(
            "math_agent.nodes.coder.complete",
            return_value=CoderDraft(purpose="主结果",
                                    code="print('coder done')")))

        sens_plan = SensitivityPlan(runs=[{"parameter": "lambda",
                                           "values": [1, 2, 3, 4, 5],
                                           "metric": "y", "rationale": "r"}])
        sens_code = SensitivityCode(code=(
            "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
            "v=[1,2,3,4,5]; r=[x*2 for x in v]\n"
            "plt.plot(v,r); plt.savefig('lambda.png')\n"
            "print(f'RESULT: parameter=lambda values={v} results={r}')\n"
        ))
        sens_interp = Interpretations(interpretations=["lambda 越大 y 线性增长。"])
        stack.enter_context(patch("math_agent.nodes.sensitivity.complete",
                                   side_effect=[sens_plan, sens_code, sens_interp]))

        fc = FigureCriticOut(score=9, approved=True)
        fa = FigureAnalysisOut(analysis="趋势单调。")
        stack.enter_context(patch("math_agent.nodes.figure_pipeline.complete",
                                   side_effect=[fc, fa]))

        # Writer 输出符合 expectations 的关键词
        stack.enter_context(patch(
            "math_agent.nodes.writer.complete",
            return_value=PaperSections(
                abstract="本文围绕排队论建模与无人机调度展开。" * 5,
                problem_restatement="x" * 200, assumptions="x" * 200,
                notation="x" * 200, model_section="排队论 + 鲁棒优化 + 风扰" * 50,
                solution="x" * 200, sensitivity="x" * 200,
                conclusion="x" * 200, references="-",
            )))
        stack.enter_context(patch(
            "math_agent.nodes.paper_critic.complete",
            return_value=CriticReport(target="paper", score=9, approved=True)))
        stack.enter_context(patch(
            "math_agent.nodes.evaluation.complete",
            return_value=EvaluationReport(
                assumption_reasonableness=8, modeling_creativity=8,
                result_correctness=8, writing_clarity=8, extra_depth=8, overall=8.0)))
        stack.enter_context(patch(
            "math_agent.nodes.latex.compile_latex",
            return_value=type("R", (object,),
                              {"success": True, "pdf_path": "", "log": "",
                               "error_kind": ""})()))

        # Now: Tracer + run graph
        from math_agent.graph import build_graph
        from math_agent.tracing import Tracer, set_current, reset_current

        tracer = Tracer(thread_id="e2e", out_dir=workdir)
        tok = set_current(tracer)
        try:
            g = build_graph()  # no checkpointer, no interrupt
            final = g.invoke({
                "problem": "排队 + 无人机覆盖混合题",
                "questions": ["建模", "调度"],
                "stage_target": "basic", "iteration": 0,
                "output_dir": str(workdir),
                "human_decision": HumanDecision(approved=True).model_dump(),
            })
        finally:
            tracer.flush()
            reset_current(tok)

    return final, rag_calls


# ----- 3. Tracer & report -----

def step_report_cli(workdir: Path) -> str:
    from math_agent.cli import app
    runner = CliRunner()
    res = runner.invoke(app, ["report", "--out", str(workdir)])
    assert res.exit_code == 0, res.stdout
    return res.stdout


# ----- 4. retry path -----

def step_rate_limit_retry() -> None:
    import math_agent.llm as llm

    class RateLimitError(Exception):
        pass
    RateLimitError.__name__ = "RateLimitError"

    ok = MagicMock()
    ok.choices = [MagicMock(message=MagicMock(content="recovered"))]
    ok.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    seq = [RateLimitError("429"), RateLimitError("429"), ok]
    with patch("litellm.completion", side_effect=seq):
        out = llm.complete("hi", model="gpt-4o-mini",
                           _retry_attempts=5, _retry_base_delay=0)
    assert out == "recovered"


def step_llm_hook_into_tracer(workdir: Path) -> dict:
    """Mock 在 litellm 层（不在节点层），证 llm._do_completion 也写 Tracer。"""
    import math_agent.llm as llm
    from math_agent.tracing import Tracer, set_current, reset_current

    fake = MagicMock()
    fake.choices = [MagicMock(message=MagicMock(content="hi"))]
    fake.usage = MagicMock(prompt_tokens=42, completion_tokens=7)

    t = Tracer(thread_id="llm-hook", out_dir=workdir)
    tok = set_current(t)
    try:
        with patch("litellm.completion", return_value=fake):
            llm.complete("ping", model="gpt-4o-mini")
            llm.complete("pong", model="gpt-4o-mini")
    finally:
        t.flush()
        reset_current(tok)
    blob = json.loads((workdir / "trace.json").read_text(encoding="utf-8"))
    return blob


# ----- 5. error_kind on real subprocess -----

def step_error_kind() -> dict:
    from math_agent.tools.runner import run_python
    from math_agent.tools.latex_compile import compile_latex
    out = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out["timeout"] = run_python("import time; time.sleep(30)",
                                     workdir=td / "t", timeout=1).error_kind
        out["runtime"] = run_python("raise ValueError('x')",
                                     workdir=td / "r").error_kind
        out["ok"] = run_python("print('ok')", workdir=td / "ok").error_kind
        tex = td / "main.tex"
        tex.write_text(r"\documentclass{article}\begin{document}x\end{document}",
                       encoding="utf-8")
        with patch("shutil.which", return_value=None):
            out["latex_missing"] = compile_latex(tex).error_kind
    return out


# ----- main -----

def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        corpus = root / "corpus"
        db = root / "rag.sqlite"
        rundir = root / "run"
        rundir.mkdir()

        step_ingest(corpus, db)

        final, rag_calls = step_graph_with_rag(rundir, db)
        assert (rundir / "paper.tex").exists(), "paper.tex missing"
        assert (rundir / "paper.md").exists(), "paper.md missing"
        assert (rundir / "trace.json").exists(), "trace.json missing"
        # rag search 应当被三个节点各至少 hit 一次
        # （analyst 1 次；modeler 在 basic/improved/final 各 1 次 = 3；writer 1 次）
        # 但若 critic 通过且没有 retry，modeler 实际是 3 次，加 analyst 1 + writer 1 = 5
        assert len(rag_calls) >= 3, f"expected >=3 RAG queries, got {len(rag_calls)}"
        trace = json.loads((rundir / "trace.json").read_text(encoding="utf-8"))
        assert trace["thread_id"] == "e2e"
        # 注：本 E2E 在 *节点层* mock 了 complete()，所以 _do_completion 未被命中，
        # llm_calls 预期为 0。节点计时来自 graph._wrap，独立于 llm hook，必须有数据。
        assert len(trace["nodes"]) > 0
        node_names = {n["name"] for n in trace["nodes"]}
        for n in ("analyst", "modeler", "model_critic", "coder", "sensitivity",
                  "figure_pipeline", "writer", "paper_critic", "evaluation",
                  "human_review", "latex"):
            assert n in node_names, f"missing node in trace: {n}"
        print(f"[2] graph+RAG: rag_search_calls={len(rag_calls)} "
              f"trace_llm_calls={trace['llm_calls']} "
              f"nodes_traced={len(trace['nodes'])} paper.tex=ok paper.md=ok  OK")

        out = step_report_cli(rundir)
        assert "Run report (e2e)" in out
        assert "Per model" in out
        assert "Nodes" in out
        print(f"[3] report CLI rendered {len(out.splitlines())} lines  OK")

        step_rate_limit_retry()
        print(f"[4] rate-limit retry path  OK")

        hookdir = root / "llmhook"
        hookdir.mkdir()
        b = step_llm_hook_into_tracer(hookdir)
        assert b["llm_calls"] == 2 and b["tokens"]["prompt"] == 84, b
        print(f"[4b] llm.complete -> Tracer hook: calls={b['llm_calls']} "
              f"prompt_tokens={b['tokens']['prompt']}  OK")

        ek = step_error_kind()
        assert ek == {"timeout": "timeout", "runtime": "runtime",
                      "ok": "", "latex_missing": "missing_binary"}, ek
        print(f"[5] error_kind on subprocess: {ek}  OK")

    print("\nALL PLAN-C E2E CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
