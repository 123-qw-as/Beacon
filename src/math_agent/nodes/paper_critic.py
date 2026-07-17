from math_agent.config import MODEL_ROUTING
from math_agent.llm import complete
from math_agent.prompts.paper_critic import SYSTEM, build_prompt
from math_agent.state import CriticIssue, CriticReport, MathModelingState
from math_agent.tools.runner import extract_valid_result_lines, infer_entity_upper_bound


def _last_successful_stdout(state: MathModelingState) -> str:
    """汇总当前批次所有通过统一门禁的 RESULT，确保评审与 writer 同源。"""
    lines: list[str] = []
    upper_bound = infer_entity_upper_bound(state.data_files)
    for art in state.latest_code_artifacts():
        if not art.success or art.evidence_role not in {"primary", "baseline"}:
            continue
        expected = art.category.split(":", 1)[1] if art.category.startswith("baseline:") else None
        lines.extend(extract_valid_result_lines(
            art.stdout,
            stderr=art.stderr,
            expected_identifier=expected,
            max_entity_count=upper_bound,
        ))
    return "\n".join(lines)


def paper_critic_node(state: MathModelingState) -> dict:
    p = state.paper
    if not any([p.abstract, p.model_section, p.solution]):
        report = CriticReport(
            target="paper", score=0, approved=False,
            issues=[CriticIssue(section="general", problem="论文初稿为空")],
            suggestions=["重新生成全部论文章节后再评审"],
        )
        return {
            "critic_reports": [report],
            "errors": ["paper_critic: 论文初稿为空，无法进入最终环节"],
        }

    out: CriticReport = complete(
        build_prompt(p, len(state.figures), len(state.sensitivity_runs),
                     _last_successful_stdout(state)),
        schema=CriticReport, system=SYSTEM,
        model=MODEL_ROUTING["paper_critic"],
    )
    out.target = "paper"
    return {"critic_reports": [out]}
