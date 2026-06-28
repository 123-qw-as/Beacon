"""Writer：把 state 内的素材组装成论文各章节文本（Markdown）。"""

SYSTEM = (
    "你是负责撰写国赛论文的主笔。请把给定素材组织成正式论文章节。"
    "禁止编造数据；引用代码结果时使用'根据计算（见附录代码 X）'句式。"
)

def build_prompt(state):
    asum = "\n".join(f"- {a.statement}" for a in state.assumptions)
    models = "\n\n".join(
        f"### {m.stage}\n{m.description}\n方程：" + "; ".join(m.equations)
        for m in state.model_versions
    )
    code_stdout = "\n".join(a.stdout for a in state.code_artifacts if a.success)[:2000]
    sens = "\n".join(
        f"- {r.parameter}: {r.metric} 随取值 {r.values} → {r.results}；{r.interpretation}"
        for r in state.sensitivity_runs
    ) or "（暂无敏感性结果）"
    return (
        f"# 题目\n{state.problem}\n\n# 假设\n{asum}\n\n"
        f"# 模型演化\n{models}\n\n# 代码运行关键输出（截断）\n{code_stdout}\n\n"
        f"# 敏感性结果\n{sens}\n\n"
        f"请输出 JSON：{{\"abstract\":str,\"problem_restatement\":str,\"assumptions\":str,"
        f"\"notation\":str,\"model_section\":str,\"solution\":str,\"sensitivity\":str,"
        f"\"conclusion\":str,\"references\":str}}。每段不少于 150 字。"
        f"（sensitivity 段：综述 state.sensitivity_runs 的关键发现；若无 runs 则写'未执行敏感性分析'。）"
    )