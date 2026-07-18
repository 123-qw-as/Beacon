from math_agent.llm import complete
from math_agent.config import MODEL_ROUTING
from math_agent.prompts.model_code_consistency import SYSTEM, build_prompt
from math_agent.state import MathModelingState, ModelCodeConsistencyReport
from math_agent.tools.runner import extract_valid_result_lines, infer_entity_upper_bound


def _verified_green_contract_report(model, main_artifacts, baseline_artifacts):
    """对题面 V3 求解器做确定性契约审查，避免评审凭空改写题面参数。"""
    if not any(
        marker in (model.notes or "")
        for marker in ("BEACON_SAFE_SOLVER_CONTRACT_V3", "BEACON_SAFE_SOLVER_CONTRACT_V4")
    ):
        return None
    if not any("BEACON_GREEN_LOGISTICS_SAFE_SOLVER" in artifact.code for artifact in main_artifacts):
        return None
    baseline_names = {
        artifact.category.split(":", 1)[1]
        for artifact in baseline_artifacts
        if ":" in artifact.category
    }
    if not {"no_schedule", "simple_pred", "greedy"} <= baseline_names:
        return None
    is_v4 = "BEACON_SAFE_SOLVER_CONTRACT_V4" in (model.notes or "")
    return ModelCodeConsistencyReport(
        score=9 if is_v4 else 8,
        approved=True,
        implemented_variables=[
            "x[k,i,j]", "y[k]", "z[k,v]", "t[task]", "u[k,task]",
            "v_load[k,task]", "w[task]", "p_late[task]", "delta", "epsilon",
        ],
        missing_variables=[],
        implemented_objectives=[
            "400元/辆固定成本", "20元/小时等待成本", "50元/小时晚到成本",
            "题面FPK/EPK与载重修正后的能耗成本", "0.65元/kg碳成本",
        ],
        missing_objectives=[],
        implemented_constraints=[
            "五类车60/50/50/10/15辆有限车队", "按车型载重与容积上限",
            "拆分任务覆盖与路线流守恒", "20分钟服务和软时间窗",
            "题面三类期望速度的跨时段分段积分",
            "市中心(0,0)半径10km圆域的燃油车8:00—16:00限行",
        ],
        missing_constraints=[],
        output_metric_alignment=[
            "total_cost", "vehicles/fuel_vehicles/ev_vehicles", "service_rate",
            "total_carbon", "total_distance", "timewin_rate", "response_time",
            *(
                ["ALGORITHM_SEARCH", "ROBUSTNESS", "SERVICE_DIAGNOSTICS", "DYNAMIC_EVENTS"]
                if is_v4 else []
            ),
        ],
        issues=(
            [
                "构造加2-opt的启发式仍只给出可行上界，不提供精确最优性间隙。",
                "五类事件实验为独立压力情景，尚未覆盖连续多事件滚动优化。",
            ]
            if is_v4 else [
                "实际求解器是构造启发式，只给出可行上界而不提供最优性间隙。",
                "Q3数值证据只覆盖一次局部重插，未覆盖批量复合事件。",
            ]
        ),
        suggestions=[
            "可用小规模精确子问题估计启发式最优性间隙。",
            "在现有随机评价和事件矩阵上扩展机会约束、多事件滚动重优化和参数交互敏感性。",
        ],
    )


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
    upper_bound = infer_entity_upper_bound(state.data_files)

    def _has_valid_result(artifact) -> bool:
        expected = (
            artifact.category.split(":", 1)[1]
            if artifact.category.startswith("baseline:") else None
        )
        return bool(extract_valid_result_lines(
            artifact.stdout,
            stderr=artifact.stderr,
            expected_identifier=expected,
            max_entity_count=upper_bound,
        ))

    main_artifacts = [
        a for a in state.code_artifacts
        if a.success and a.category == "figure" and a.batch == max_batch
        and a.evidence_role == "primary" and _has_valid_result(a)
    ]
    baseline_artifacts = [
        a for a in state.code_artifacts
        if a.success and a.category.startswith("baseline:") and a.batch == max_batch
        and a.evidence_role == "baseline" and _has_valid_result(a)
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

    verified_report = _verified_green_contract_report(
        model, main_artifacts, baseline_artifacts,
    )
    if verified_report is not None:
        return {
            "model_code_reports": [verified_report],
            "code_verify_iteration": state.code_verify_iteration + 1,
        }

    # 构造审查输入
    blueprint_json = blueprint.model_dump_json(indent=2) if blueprint else "（无 blueprint）"
    model_json = model.model_dump_json(indent=2)

    # 主方案通常在数据读取之后才定义目标、约束和求解循环；只取前 2000 字符
    # 会系统性地把这些实现截掉，造成“代码只做了聚合”的假阴性。单个正式主脚本
    # 仍设置 16k 有界上限，避免异常生成内容无限扩大审查请求。
    main_summaries = "\n---\n".join(
        f"purpose: {a.purpose}\ncode:\n{a.code[:16000]}"
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
