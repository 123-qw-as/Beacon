from math_agent.llm import complete
from math_agent.config import MODEL_ROUTING
from math_agent.prompts.writer import SYSTEM, build_prompt
from math_agent.state import MathModelingState, PaperSections


def writer_node(state: MathModelingState) -> dict:
    out: PaperSections = complete(
        build_prompt(state),
        schema=PaperSections,
        system=SYSTEM,
        model=MODEL_ROUTING["writer"],
    )
    return {"paper": out}


def render_markdown(state: MathModelingState) -> str:
    p = state.paper
    body = (
        f"# {state.problem}\n\n"
        f"## 摘要\n{p.abstract}\n\n"
        f"## 1. 问题重述\n{p.problem_restatement}\n\n"
        f"## 2. 模型假设\n{p.assumptions}\n\n"
        f"## 3. 符号说明\n{p.notation}\n\n"
        f"## 4. 模型的建立与演化\n{p.model_section}\n\n"
        f"## 5. 模型的求解\n{p.solution}\n\n"
        f"## 6. 敏感性分析\n{p.sensitivity}\n\n"
        f"## 7. 模型评价与结论\n{p.conclusion}\n\n"
        f"## 参考文献\n{p.references}\n\n---\n\n## 附录 A. 代码与运行输出\n"
    )
    for i, a in enumerate(state.code_artifacts, 1):
        body += f"### A.{i} {a.purpose}（success={a.success}）\n```python\n{a.code}\n```\n**stdout**：\n```\n{a.stdout}\n```\n"
        if a.stderr:
            body += f"**stderr**：\n```\n{a.stderr}\n```\n"
    if state.figures:
        body += "\n## 附录 B. 图表\n"
        for f in state.figures:
            body += f"### {f.purpose}\n![]({f.path})\n\n{f.analysis}\n\n"
    return body
