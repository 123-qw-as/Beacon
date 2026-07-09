"""质量评估：从 final state 提取四维度分数，判定是否达到最低质量门槛。

四维度（每项 0-10）：
- BlueprintScore：blueprint 完整性（小问覆盖、目标/约束/指标/路线/验证计划齐全）
- ModelScore：模型质量（question_coverage、推导链、baseline、model_critic 通过）
- CodeScore：代码质量（成功 artifact、RESULT 行、一致性审查通过）
- PaperScore：论文质量（关键词覆盖、evaluation overall、paper_critic 通过）

门槛：每项 >= 7，overall = 算术平均 >= 7.2。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent
_RUBRICS_DIR = _BENCH_ROOT / "rubrics"


@dataclass
class QualityScore:
    blueprint: int
    model: int
    code: int
    paper: int
    overall: float
    passed: bool
    failures: list[str] = field(default_factory=list)
    question_coverage: str = ""  # 如 "3/3"
    unresolved_issues: int = 0


def load_rubric(problem_id: str) -> dict:
    path = _RUBRICS_DIR / f"{problem_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _get(state, key, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _score_blueprint(state, rubric: dict) -> tuple[int, list[str]]:
    bp = _get(state, "problem_blueprint")
    issues: list[str] = []
    if bp is None:
        return 0, ["problem_blueprint is None"]

    spec = rubric.get("blueprint", {})
    score = 10

    subqs = getattr(bp, "subquestions", []) or []
    min_sq = spec.get("min_subquestions", 0)
    if len(subqs) < min_sq:
        score -= 3
        issues.append(f"subquestions {len(subqs)} < {min_sq}")

    checks = [
        ("must_have_objectives", "objectives"),
        ("must_have_constraints", "constraints"),
        ("must_have_metrics", "metrics"),
        ("must_have_recommended_route", "recommended_route"),
        ("must_have_validation_plan", "validation_plan"),
    ]
    for flag, attr in checks:
        if spec.get(flag):
            val = getattr(bp, attr, None)
            if not val:
                score -= 2
                issues.append(f"blueprint missing {attr}")

    return max(0, score), issues


def _score_model(state, rubric: dict) -> tuple[int, list[str]]:
    issues: list[str] = []
    models = _get(state, "model_versions") or []
    if not models:
        return 0, ["no model_versions"]
    model = models[-1]
    spec = rubric.get("model", {})

    score = 10

    if spec.get("must_have_question_coverage"):
        cov = getattr(model, "question_coverage", []) or []
        if not cov:
            score -= 3
            issues.append("missing question_coverage")

    min_deriv = spec.get("must_have_derivation_steps", 0)
    steps = len(getattr(model, "derivation_steps", []) or [])
    if min_deriv and steps < min_deriv:
        score -= 2
        issues.append(f"derivation_steps {steps} < {min_deriv}")

    if spec.get("must_have_baseline"):
        notes = getattr(model, "notes", "") or ""
        desc = getattr(model, "description", "") or ""
        # ponytail: 宽松匹配 baseline 关键词，避免对 LLM 措辞过度敏感
        if "baseline" not in (notes + desc).lower():
            score -= 1
            issues.append("model missing baseline mention")

    # model_critic 是否通过
    critics = _get(state, "critic_reports") or []
    model_critic = next(
        (r for r in reversed(critics) if _get(r, "target", "") == "modeler"), None)
    if model_critic is not None and not _get(model_critic, "approved", False):
        score -= 2
        issues.append("model_critic not approved")

    return max(0, score), issues


def _score_code(state, rubric: dict) -> tuple[int, list[str]]:
    issues: list[str] = []
    artifacts = _get(state, "code_artifacts") or []
    spec = rubric.get("code", {})

    if not artifacts:
        return 0, ["no code_artifacts"]

    score = 10
    success = [a for a in artifacts if _get(a, "success", False)]
    min_success = spec.get("min_success_artifacts", 0)
    if len(success) < min_success:
        score -= 4
        issues.append(f"success artifacts {len(success)} < {min_success}")

    if spec.get("must_have_result_lines"):
        has_result = any(
            "RESULT:" in (_get(a, "stdout", "") or "")
            for a in success
        )
        if not has_result:
            score -= 4
            issues.append("no RESULT: lines in successful stdout")

    if spec.get("must_have_consistency_approved"):
        reports = _get(state, "model_code_reports") or []
        if reports:
            last = reports[-1]
            if not _get(last, "approved", False):
                score -= 4
                issues.append("model_code_consistency not approved")
        else:
            score -= 2
            issues.append("no model_code_reports")

    return max(0, score), issues


def _score_paper(state, rubric: dict) -> tuple[int, list[str]]:
    issues: list[str] = []
    paper = _get(state, "paper")
    spec = rubric.get("paper", {})

    if paper is None:
        return 0, ["no paper"]

    score = 10
    text = " ".join([
        getattr(paper, f, "") or ""
        for f in ("abstract", "model_section", "solution", "conclusion")
    ])
    for kw in spec.get("must_contain_keywords", []):
        if kw not in text:
            score -= 2
            issues.append(f"paper missing keyword: {kw}")

    evaluation = _get(state, "evaluation")
    if evaluation is not None:
        overall = float(getattr(evaluation, "overall", 0))
        min_overall = spec.get("min_overall", 0)
        if overall < min_overall:
            score -= 2
            issues.append(f"evaluation overall {overall} < {min_overall}")

    # paper_critic 是否通过
    critics = _get(state, "critic_reports") or []
    paper_critic = next(
        (r for r in reversed(critics) if _get(r, "target", "") == "paper"), None)
    if paper_critic is not None and not _get(paper_critic, "approved", False):
        score -= 2
        issues.append("paper_critic not approved")

    return max(0, score), issues


def evaluate_quality(state, problem_id: str) -> QualityScore:
    """从 final state 提取四维度分数，判定是否达到最低质量门槛。"""
    rubric = load_rubric(problem_id)
    thresholds = rubric.get("thresholds", {})

    bp_score, bp_issues = _score_blueprint(state, rubric)
    model_score, model_issues = _score_model(state, rubric)
    code_score, code_issues = _score_code(state, rubric)
    paper_score, paper_issues = _score_paper(state, rubric)

    overall = (bp_score + model_score + code_score + paper_score) / 4.0
    all_issues = bp_issues + model_issues + code_issues + paper_issues

    min_bp = thresholds.get("min_blueprint", 7)
    min_model = thresholds.get("min_model", 7)
    min_code = thresholds.get("min_code", 7)
    min_paper = thresholds.get("min_paper", 7)
    min_overall = thresholds.get("min_overall", 7.2)

    failures: list[str] = []
    if bp_score < min_bp:
        failures.append(f"BlueprintScore {bp_score} < {min_bp}")
    if model_score < min_model:
        failures.append(f"ModelScore {model_score} < {min_model}")
    if code_score < min_code:
        failures.append(f"CodeScore {code_score} < {min_code}")
    if paper_score < min_paper:
        failures.append(f"PaperScore {paper_score} < {min_paper}")
    if overall < min_overall:
        failures.append(f"overall {overall} < {min_overall}")

    # question_coverage 摘要
    models = _get(state, "model_versions") or []
    bp = _get(state, "problem_blueprint")
    total_sq = len(getattr(bp, "subquestions", []) or []) if bp else 0
    covered = len(getattr(models[-1], "question_coverage", []) or []) if models else 0
    coverage_str = f"{covered}/{total_sq}" if total_sq else f"{covered}/0"

    return QualityScore(
        blueprint=bp_score,
        model=model_score,
        code=code_score,
        paper=paper_score,
        overall=round(overall, 2),
        passed=len(failures) == 0,
        failures=failures,
        question_coverage=coverage_str,
        unresolved_issues=len(all_issues),
    )
