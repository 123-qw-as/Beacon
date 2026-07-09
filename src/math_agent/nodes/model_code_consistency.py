from math_agent.llm import complete
from math_agent.config import MODEL_ROUTING
from math_agent.prompts.model_code_consistency import SYSTEM, build_prompt
from math_agent.state import MathModelingState, ModelCodeConsistencyReport


def model_code_consistency_node(state: MathModelingState) -> dict:
    blueprint = state.problem_blueprint
    model = state.latest_model()

    if model is None:
        report = ModelCodeConsistencyReport(
            score=0, approved=False,
            issues=["model_code_consistency: 没有 model_versions，无法审查"],
        )
        return {"model_code_reports": [report],
                "code_verify_iteration": state.code_verify_iteration + 1}

    # 只看最新批次的 artifact（batch 递增机制保证 retry 不产生脏数据）
    max_batch = max((a.batch for a in state.code_artifacts), default=0)
    main_artifacts = [
        a for a in state.code_artifacts
        if a.success and a.category == "figure" and a.batch == max_batch
    ]
    baseline_artifacts = [
        a for a in state.code_artifacts
        if a.category.startswith("baseline:") and a.batch == max_batch
    ]
    failed_artifacts = [
        a for a in state.code_artifacts
        if not a.success and a.batch == max_batch
    ]

    if not main_artifacts:
        # 没有成功主方案代码 -> 直接未通过
        report = ModelCodeConsistencyReport(
            score=0, approved=False,
            missing_variables=list(model.variables.keys()),
            issues=["model_code_consistency: 没有成功的主方案代码 artifact，无法审查一致性"],
        )
        return {"model_code_reports": [report],
                "code_verify_iteration": state.code_verify_iteration + 1}

    # 构造审查输入
    blueprint_json = blueprint.model_dump_json(indent=2) if blueprint else "（无 blueprint）"
    model_json = model.model_dump_json(indent=2)

    main_summaries = "\n---\n".join(
        f"purpose: {a.purpose}\ncode:\n{a.code[:2000]}"
        for a in main_artifacts
    )
    main_stdout = "\n---\n".join(
        f"[{a.purpose}]\n{a.stdout[:1000]}" for a in main_artifacts if a.stdout
    ) or "（无 stdout）"
    baseline_stdout = "\n---\n".join(
        f"[{a.category}]\n{a.stdout[:500]}" for a in baseline_artifacts if a.stdout
    ) or "（无 baseline stdout）"
    failed_stderr = "\n---\n".join(
        f"[{a.category or 'figure'}]\n{a.stderr[:300]}" for a in failed_artifacts if a.stderr
    ) or "（无失败 artifact）"

    prompt = build_prompt(
        blueprint_json, model_json, main_summaries, main_stdout,
        baseline_stdout, failed_stderr,
    )
    out: ModelCodeConsistencyReport = complete(
        prompt, schema=ModelCodeConsistencyReport, system=SYSTEM,
        model=MODEL_ROUTING["model_critic"],
    )
    return {
        "model_code_reports": [out],
        "code_verify_iteration": state.code_verify_iteration + 1,
    }
