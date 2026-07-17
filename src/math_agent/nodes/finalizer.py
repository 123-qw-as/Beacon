"""最终收口节点：验证状态与产物，并原子提交可机读完成标记。"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from math_agent.errors import FinalizationError
from math_agent.state import ArtifactDigest, FinalizationReport, MathModelingState
from math_agent.tools.runner import extract_valid_result_lines, infer_entity_upper_bound


def _atomic_json(path: Path, payload: dict) -> None:
    """同目录临时文件 + replace，避免进程被杀后留下半截 JSON。"""
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _digest(path: Path) -> ArtifactDigest:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return ArtifactDigest(size=path.stat().st_size, sha256=h.hexdigest())


def load_verified_completion(out: str | Path) -> FinalizationReport | None:
    """读取并校验终态文件和所有已登记产物摘要；任一不一致即返回 None。"""
    out = Path(out).resolve()
    try:
        completion_blob = json.loads((out / "completion.json").read_text(encoding="utf-8"))
        final_blob = json.loads((out / "final_state.json").read_text(encoding="utf-8"))
        report = FinalizationReport.model_validate(completion_blob)
        committed = FinalizationReport.model_validate(final_blob["finalization"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None
    if report.model_dump() != committed.model_dump():
        return None
    if report.status not in {"completed", "degraded"}:
        return None
    if not set(report.artifacts).issubset({"paper.md", "paper.tex", "paper.pdf", "compile.log"}):
        return None
    if not {"paper.md", "paper.tex"}.issubset(report.artifacts):
        return None
    for name, expected in report.artifacts.items():
        path = out / name
        if not path.is_file() or _digest(path) != expected:
            return None
    return report


def _collect_invariant_issues(state: MathModelingState, out: Path) -> list[str]:
    issues: list[str] = []
    for name in ("modeler_phase", "coder_phase", "sensitivity_phase", "figure_phase"):
        value = getattr(state, name)
        if value != "done":
            issues.append(f"{name} 必须为 done，当前为 {value}")

    pending = {
        "modeler_derivation_queue": state.modeler_derivation_queue,
        "coder_work_queue": state.coder_work_queue,
        "coder_pending_draft": state.coder_pending_draft,
        "sensitivity_pending_runs": state.sensitivity_pending_runs,
        "sensitivity_pending_code": state.sensitivity_pending_code,
        "figure_work_queue": state.figure_work_queue,
        "figure_current_critic": state.figure_current_critic,
        "figure_work_results": state.figure_work_results,
        "writer_section_queue": state.writer_section_queue,
    }
    for name, value in pending.items():
        if value:
            issues.append(f"{name} 必须为空")
    if state.modeler_draft is not None:
        issues.append("modeler_draft 必须为空")

    if not any(model.stage == "final" for model in state.model_versions):
        issues.append("缺少 final 阶段模型")
    if state.evaluation is None:
        issues.append("缺少 evaluation")
    if state.human_decision is None or not state.human_decision.approved:
        issues.append("缺少已批准的 human_decision")
    for field in ("abstract", "model_section", "solution", "conclusion"):
        if not (getattr(state.paper, field, "") or "").strip():
            issues.append(f"paper.{field} 为空")

    for name in ("paper.md", "paper.tex"):
        path = out / name
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"缺少有效产物 {name}")
    return issues


def _minimum_final_score() -> float:
    try:
        return max(0.0, min(10.0, float(os.getenv("MATH_AGENT_MIN_FINAL_SCORE", "7.0"))))
    except ValueError:
        return 7.0


def _minimum_paper_body_pages() -> int:
    try:
        return max(1, int(os.getenv("MATH_AGENT_MIN_PAPER_BODY_PAGES", "20")))
    except ValueError:
        return 20


def _minimum_paper_body_chars() -> int:
    try:
        return max(0, int(os.getenv("MATH_AGENT_MIN_PAPER_BODY_CHARS", "15000")))
    except ValueError:
        return 15000


def _pdf_body_metrics(pdf_path: Path) -> tuple[int, int, int, int]:
    """返回 PDF 总页数、正文页数、非空正文页数和正文非空白字符数。

    正文在“关键算法代码”或明确的附录首页之前结束。页数与文本量同时计量，
    避免通过空白页、纯图片占位或把代码附录拉长来满足竞赛论文篇幅门禁。
    """
    texts = [(page.extract_text() or "") for page in PdfReader(str(pdf_path)).pages]
    appendix_index = len(texts)
    for index, text in enumerate(texts):
        compact = "".join(text.split())
        if (
            "关键算法代码" in compact
            or re.search(r"(?mi)^\s*(?:附录|Appendix)\b", text)
        ):
            appendix_index = index
            break
    body = texts[:appendix_index]
    nonempty_pages = sum(bool("".join(text.split())) for text in body)
    nonspace_chars = sum(len("".join(text.split())) for text in body)
    return len(texts), len(body), nonempty_pages, nonspace_chars


def _latex_artifacts_verified(out: Path) -> bool:
    """Return whether a later successful compile supersedes historical errors."""
    pdf_path = out / "paper.pdf"
    log_path = out / "compile.log"
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0 or not log_path.is_file():
        return False
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    # The compiler writes a fresh log for each invocation. A successful
    # two-pass XeLaTeX run is authoritative recovery evidence for an older
    # append-only state.errors entry.
    return "[pass 1] exit=0" in content and "[pass 2] exit=0" in content


def _paper_evidence_lineage_warnings(
    state: MathModelingState, out: Path,
) -> list[str]:
    """拒绝 append-only 历史重新进入正式论文产物。

    sensitivity_runs 会保留重试历史，但每个参数只有最后一轮属于正式证据。
    这里不试图用正则猜测所有论文数字，只检查可机械证明为旧轮次的解释文本和图路径。
    """
    bodies: list[str] = []
    for name in ("paper.md", "paper.tex"):
        path = out / name
        if path.is_file():
            try:
                bodies.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
    if not bodies:
        return []
    joined = "\n".join(bodies).replace("\\", "/")
    latest_index: dict[str, int] = {}
    for index, run in enumerate(state.sensitivity_runs):
        latest_index[run.parameter] = index

    warnings: list[str] = []
    for index, run in enumerate(state.sensitivity_runs):
        if latest_index.get(run.parameter) == index:
            continue
        interpretation = run.interpretation.strip()
        if len(interpretation) >= 16 and interpretation in joined:
            warnings.append(
                f"质量门禁：论文包含参数“{run.parameter}”的历史敏感性解释"
            )
        if run.figure_path:
            stale_path = str(run.figure_path).replace("\\", "/")
            if stale_path in joined:
                warnings.append(
                    f"质量门禁：论文包含参数“{run.parameter}”的历史敏感性图"
                )
    return list(dict.fromkeys(warnings))


def _unresolved_runtime_errors(state: MathModelingState, out: Path) -> list[str]:
    """错误列表保留追踪历史；终态只报告尚未被后续有效证据解决的错误。"""
    latest = state.latest_code_artifacts()
    has_primary = any(
        a.success and a.category == "figure" and a.evidence_role == "primary"
        for a in latest
    )
    latex_ok = _latex_artifacts_verified(out)
    unresolved: list[str] = []
    for item in state.errors:
        text = str(item)
        if text.startswith("coder:") and has_primary:
            continue
        if text.startswith("sensitivity:") and state.sensitivity_runs:
            continue
        if text.startswith("latex compile failed:") and latex_ok:
            continue
        unresolved.append(text)
    return unresolved


def _collect_quality_warnings(state: MathModelingState, out: Path) -> list[str]:
    """收集不阻断产物落盘、但禁止标记为 completed 的质量问题。"""
    warnings: list[str] = []
    threshold = _minimum_final_score()
    warnings.extend(_paper_evidence_lineage_warnings(state, out))

    pdf_path = out / "paper.pdf"
    if pdf_path.is_file() and pdf_path.stat().st_size > 0:
        try:
            total_pages, body_pages, nonempty_pages, body_chars = _pdf_body_metrics(pdf_path)
        except Exception as exc:  # pragma: no cover - 第三方 PDF 解析器异常类型不稳定
            warnings.append(f"质量门禁：paper.pdf 无法读取正文篇幅（{exc}）")
        else:
            minimum_pages = _minimum_paper_body_pages()
            minimum_chars = _minimum_paper_body_chars()
            if body_pages < minimum_pages:
                warnings.append(
                    f"质量门禁：论文正文页数 {body_pages}，至少需要 {minimum_pages} 页"
                    f"（PDF 共 {total_pages} 页）"
                )
            if nonempty_pages < body_pages:
                warnings.append(
                    f"质量门禁：正文存在 {body_pages - nonempty_pages} 个空白页"
                )
            if body_chars < minimum_chars:
                warnings.append(
                    f"质量门禁：论文正文非空白字符数 {body_chars}，至少需要 {minimum_chars}"
                )

    upper_bound = infer_entity_upper_bound(state.data_files)
    valid_main = 0
    valid_baseline_categories: set[str] = set()
    green_depth_labels = (
        "ALGORITHM_SEARCH", "ROBUSTNESS", "SERVICE_DIAGNOSTICS", "DYNAMIC_EVENTS",
    )
    green_depth_missing: set[str] = set()
    for artifact in state.latest_code_artifacts():
        if not artifact.success:
            continue
        expected = (
            artifact.category.split(":", 1)[1]
            if artifact.category.startswith("baseline:") else None
        )
        lines = extract_valid_result_lines(
            artifact.stdout,
            stderr=artifact.stderr,
            expected_identifier=expected,
            max_entity_count=upper_bound,
        )
        if not lines:
            continue
        if artifact.category == "figure" and artifact.evidence_role == "primary":
            valid_main += 1
            if "BEACON_GREEN_LOGISTICS_SAFE_SOLVER" in (artifact.code or ""):
                green_depth_missing.update(
                    label for label in green_depth_labels
                    if f"{label}:" not in (artifact.stdout or "")
                )
        elif artifact.category.startswith("baseline:") and artifact.evidence_role == "baseline":
            valid_baseline_categories.add(artifact.category)
    if valid_main == 0:
        warnings.append("质量门禁：缺少通过协议与合理性校验的主方案 RESULT")
    if green_depth_missing:
        warnings.append(
            "质量门禁：城市物流主方案缺少深度实验："
            + "、".join(sorted(green_depth_missing))
        )
    if len(valid_baseline_categories) < 2:
        warnings.append(
            f"质量门禁：有效对照方案仅 {len(valid_baseline_categories)} 个，至少需要 2 个"
        )

    if not state.sensitivity_runs:
        warnings.append("质量门禁：缺少有效敏感性分析结果")

    consistency = state.model_code_reports[-1] if state.model_code_reports else None
    if consistency is None:
        warnings.append("质量门禁：缺少模型—代码一致性报告")
    elif not consistency.approved or consistency.score < 7:
        warnings.append(
            f"质量门禁：模型—代码一致性未通过（score={consistency.score}, "
            f"approved={consistency.approved}）"
        )

    paper_critic = state.latest_critic("paper")
    if paper_critic is None:
        warnings.append("质量门禁：缺少论文评审报告")
    elif not paper_critic.approved or paper_critic.score < 7:
        warnings.append(
            f"质量门禁：论文评审未通过（score={paper_critic.score}, "
            f"approved={paper_critic.approved}）"
        )

    if state.evaluation is not None:
        if state.evaluation.overall < threshold:
            warnings.append(
                f"质量门禁：综合评分 {state.evaluation.overall:.2f} 低于 {threshold:.2f}"
            )
        weak = {
            "假设合理性": state.evaluation.assumption_reasonableness,
            "建模创新性": state.evaluation.modeling_creativity,
            "结果正确性": state.evaluation.result_correctness,
            "写作清晰度": state.evaluation.writing_clarity,
            "分析深度": state.evaluation.extra_depth,
        }
        low = [f"{name}={score}" for name, score in weak.items() if score < 6]
        if low:
            warnings.append("质量门禁：存在低于 6 分的维度（" + "，".join(low) + "）")

    compile_log = out / "compile.log"
    if compile_log.is_file():
        try:
            content = compile_log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = ""
        excesses = [
            float(value)
            for value in re.findall(r"Overfull \\hbox \((\d+(?:\.\d+)?)pt too wide\)", content)
        ]
        severe = [value for value in excesses if value > 10.0]
        if severe:
            warnings.append(
                f"质量门禁：LaTeX 存在 {len(severe)} 处严重横向溢出，最大 {max(severe):.2f}pt"
            )
    return warnings


def finalizer_node(state: MathModelingState) -> dict:
    """验证终态并写 completion.json / final_state.json。

    内部状态未收口或核心论文产物缺失时抛出异常，让 LangGraph checkpoint
    保持在 finalizer 之前；修复后执行 recover 即可再次验证。
    """
    out = Path(state.output_dir or ".").resolve()
    out.mkdir(parents=True, exist_ok=True)
    completed_at = datetime.now(timezone.utc).isoformat()
    issues = _collect_invariant_issues(state, out)
    warnings = _unresolved_runtime_errors(state, out)

    if issues:
        report = FinalizationReport(
            status="failed", issues=issues, warnings=warnings, completed_at=completed_at,
        )
        (out / "final_state.json").unlink(missing_ok=True)
        _atomic_json(out / "completion.json", report.model_dump(mode="json"))
        raise FinalizationError("；".join(issues))

    artifacts: dict[str, ArtifactDigest] = {}
    for name in ("paper.md", "paper.tex", "paper.pdf", "compile.log"):
        path = out / name
        if path.is_file() and path.stat().st_size > 0:
            artifacts[name] = _digest(path)
    if "paper.pdf" not in artifacts:
        warnings.append("未生成 paper.pdf；Markdown 与 LaTeX 源文件可用")

    warnings.extend(_collect_quality_warnings(state, out))
    warnings = list(dict.fromkeys(warnings))

    status = "degraded" if warnings else "completed"
    report = FinalizationReport(
        status=status,
        warnings=warnings,
        artifacts=artifacts,
        completed_at=completed_at,
    )
    committed = state.model_copy(deep=True, update={"finalization": report})
    _atomic_json(out / "final_state.json", committed.model_dump(mode="json"))
    _atomic_json(out / "completion.json", report.model_dump(mode="json"))
    if load_verified_completion(out) is None:
        raise FinalizationError("终态文件写入后的摘要复核失败")
    return {"finalization": report}
