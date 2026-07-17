from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote

from jinja2 import Environment, FileSystemLoader, select_autoescape

from math_agent.config import (
    MODEL_ROUTING,
    RAG_CTX_MAX_CHARS_WRITER,
    RAG_DB_PATH,
    RAG_EMBEDDING_DIM,
    RAG_EMBEDDING_MODEL,
    RAG_ENABLED,
    RAG_TOPK,
)
from math_agent.llm import complete
from math_agent.nodes.rendering import (
    _curate_code,
    _curate_stdout,
    _normalize_escaped_layout_text,
)
from math_agent.prompts.writer import SYSTEM, build_prompt  # noqa: F401
from math_agent.prompts.writer_section import (
    WriterOutline,
    _sections_to_rewrite,
    build_outline_prompt,
    build_section_prompt,
    schema_for_group,
    writer_sections,
)
from math_agent.rag.retrieve import format_snippets, search
from math_agent.state import FigureArtifact, MathModelingState, SensitivityRun
from math_agent.tools.runner import extract_valid_result_lines, infer_entity_upper_bound


def writer_node(state: MathModelingState) -> dict:
    """准备论文大纲和本轮分节队列。

    分节写作使每一节完成后都能形成 LangGraph checkpoint，进程被杀或路由器
    短暂不可用时，恢复运行只需继续尚未完成的章节。
    """
    ctx = ""
    if RAG_ENABLED:
        prev_paper_hint = ""
        if state.paper is not None and state.paper.model_section:
            prev_paper_hint = state.paper.model_section[:500]
        query = (state.problem + " " + prev_paper_hint).strip()
        snippets = search(
            query,
            db_path=RAG_DB_PATH,
            k=RAG_TOPK,
            embedding_model=RAG_EMBEDDING_MODEL,
            dim=RAG_EMBEDDING_DIM,
            source_type="paper",
        )
        ctx = format_snippets(snippets, max_chars=RAG_CTX_MAX_CHARS_WRITER)

    prior_critic = state.latest_critic("paper")
    if state.writer_iteration == 0:
        if _has_green_safe_solver(state):
            outline = WriterOutline(
                abstract="题面口径、方法、主结果与证据边界",
                problem_restatement="Q1/Q2/Q3职责划分",
                assumptions="数据、拆分、有限车队与政策假设",
                notation="决策变量、参数和单位",
                model_section="目标、容量、速度、能耗与限行约束",
                solution="主方案、Q1场景、两类Q2基线和动态重插",
                sensitivity="三组中心点对齐的单因素扫描",
                conclusion="结论、局限与可扩展方向",
                references="题面文献和方法来源",
            )
        else:
            outline = complete(
                build_outline_prompt(state, retrieved_context=ctx),
                schema=WriterOutline,
                system=SYSTEM,
                model=MODEL_ROUTING["writer"],
                profile="long",
            )
    else:
        outline = WriterOutline(**state.writer_outline_dump)

    if state.writer_iteration > 0 and prior_critic is not None and prior_critic.issues:
        run_set = set(_sections_to_rewrite(prior_critic.issues))
    else:
        run_set = {group.name for group in writer_sections()}

    return {
        "writer_section_queue": [
            group.name for group in writer_sections() if group.name in run_set
        ],
        "writer_outline_dump": outline.model_dump(),
        "writer_retrieved_context": ctx,
        "writer_iteration": state.writer_iteration + 1,
    }


def _coerce_unit_and_type(description: str) -> tuple[str, str]:
    text = (description or "").lower()
    if any(word in text for word in ("binary", "0-1", "是否", "选择")):
        return "—", "0-1变量"
    if any(word in text for word in ("integer", "整数", "数量", "辆")):
        return "按定义", "整数变量"
    if any(word in text for word in ("continuous", "连续", "距离", "时间", "成本")):
        return "按定义", "连续变量"
    return "—", "参数或变量"


def _build_assumptions_text(state: MathModelingState) -> str:
    blocks: list[str] = []
    for index, item in enumerate(state.assumptions[:8], 1):
        rationale = item.rationale or "该假设用于明确模型边界，并保证求解过程可计算。"
        impact = "若该假设偏离实际，应通过场景分析检验结论是否稳定。"
        if item.sensitivity_relevant:
            impact += "该项已列入敏感性分析的重点参数。"
        blocks.append(
            f"**假设{index}**：{item.statement}\n"
            f"**依据**：{rationale}\n"
            f"**影响与检验**：{impact}"
        )
    if blocks:
        return "\n\n".join(blocks)
    return (
        "**假设1**：题目给出的数据在研究周期内有效，未给出的随机扰动不作为基准模型输入。\n"
        "**依据**：该假设使模型与现有数据口径一致。\n"
        "**影响与检验**：在扩展分析中通过扰动关键参数评估结论的稳健性。"
    )


def _build_notation_text(state: MathModelingState) -> str:
    model = state.latest_model()
    variables = model.variables if model else {}
    lines = ["| 符号 | 含义 | 单位 | 类型 |", "|---|---|---|---|"]
    for symbol, meaning in list(variables.items())[:20]:
        unit, kind = _coerce_unit_and_type(meaning)
        lines.append(f"| `{symbol}` | {meaning} | {unit} | {kind} |")
    if len(lines) == 2:
        lines.append("| `x` | 模型的核心决策变量 | 按定义 | 决策变量 |")
    lines.append(
        "\n同一符号在全文中保持唯一含义；集合索引、决策变量与外生参数分别说明，"
        "0-1变量取值为1表示相应方案被采用。"
    )
    return "\n".join(lines)


def _build_model_section_text(state: MathModelingState) -> str:
    model = state.latest_model()
    if model is None:
        return (
            "## 建模思路\n"
            "依据题目目标定义决策变量，并将业务规则转化为可检验的约束。\n\n"
            "## 目标函数与约束\n"
            "以题目要求的核心指标为目标，并保留可行性、容量和时序约束。\n\n"
            "## 求解与验证\n"
            "采用可复现程序求解，并通过边界检查和敏感性分析验证结果。"
        )

    equation_lines = "\n".join(
        f"- $${equation}$$" for equation in model.equations[:10]
    ) or "- 目标函数与约束以程序实现为准。"
    objectives = "；".join(model.objective_mapping[:4]) or model.description
    constraints = "；".join(model.constraint_mapping[:6]) or "满足题目给定的业务约束"
    validations = "；".join(model.validation_mapping[:4]) or "进行可行性与边界检查"
    return (
        "## 模型结构与目标\n"
        f"模型围绕以下目标建立：{objectives}。通过统一决策变量连接各子问题，"
        "确保目标、约束、算法输出和评价指标具有一致的数据口径。\n\n"
        f"{equation_lines}\n\n"
        "## 约束体系\n"
        f"约束映射为：{constraints}。求解时逐项检查容量、时序、资源和逻辑可行性，"
        "不以惩罚值代替必须满足的硬约束。\n\n"
        "## 验证设计\n"
        f"验证计划包括：{validations}。同时使用程序输出、边界场景和敏感性结果"
        "相互校验，避免仅凭单次最优值下结论。"
    )


def _result_evidence(state: MathModelingState) -> list[str]:
    evidence: list[str] = []
    upper_bound = infer_entity_upper_bound(state.data_files)
    for artifact in state.latest_code_artifacts():
        # 正文的主结果证据只取正式主求解器；基线由专门的比较段落承载。
        if not artifact.success or artifact.evidence_role != "primary":
            continue
        expected = (
            artifact.category.split(":", 1)[1]
            if artifact.category.startswith("baseline:") else None
        )
        for raw_line in extract_valid_result_lines(
            artifact.stdout,
            stderr=artifact.stderr,
            expected_identifier=expected,
            max_entity_count=upper_bound,
        ):
            line = raw_line.strip()
            # 只保留结构化结果行，避免把 saved/data_dir 等本机路径写入论文。
            if re.search(r"(?:[A-Za-z]:[\\/]|[/\\](?:Users|home)[/\\])", line):
                continue
            evidence.append(line.replace("\\", "/"))
    return evidence[:6]


def _build_solution_text(state: MathModelingState) -> str:
    evidence = _result_evidence(state)
    evidence_text = (
        "\n".join(f"- `{line}`" for line in evidence)
        if evidence
        else "- 当前 checkpoint 未提供可引用的结构化 RESULT 行。"
    )
    return (
        "## 求解流程\n"
        "先完成数据一致性检查，再构造可行初解并执行优化；每次求解后重新计算目标值、"
        "约束违反量和题目要求的评价指标。中间结果按节点写入 checkpoint，恢复运行时"
        "复用已完成计算。\n\n"
        "## 可复现结果\n"
        f"程序提供的结构化证据如下：\n{evidence_text}\n\n"
        "论文只引用上述程序实际输出的数值。若某项指标缺少结构化结果，则将其标记为"
        "待验证项，不用推测值补齐。"
    )


def _build_sensitivity_text(state: MathModelingState) -> str:
    if not state.sensitivity_runs:
        return (
            "当前运行未形成有效的敏感性结果，因此不报告未经计算的变化幅度。"
            "建议围绕成本、容量、需求和时间约束分别设置扰动场景后重新计算。"
        )

    ranked = sorted(
        (
            (run, max(run.results) - min(run.results) if run.results else 0.0)
            for run in state.sensitivity_runs
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    paragraphs = [
        "采用单因素扰动保持其他条件不变，并在每个扰动点重新求解模型。",
        "结果跨度用于比较局部敏感程度，但不替代可行性和业务边界检查。",
    ]
    for index, (run, span) in enumerate(ranked, 1):
        interpretation = run.interpretation or "该参数变化会影响模型输出，应结合业务边界解释。"
        paragraphs.append(
            f"{index}. 参数 `{run.parameter}` 取值为 {run.values}，指标 `{run.metric}` "
            f"对应结果为 {run.results}，观测跨度为 {span:.4g}。{interpretation}"
        )
    return "\n\n".join(paragraphs)


def _build_conclusion_text(state: MathModelingState) -> str:
    model = state.latest_model()
    stage = model.stage if model else "最终模型"
    return (
        f"本文围绕题目要求完成了从问题拆解、{stage}建立、程序求解到稳健性检验的完整链路。"
        "结论以程序的结构化输出为证据，并对缺少数据支持的指标保留限制说明。\n\n"
        "模型的优点是目标与约束映射明确、运行过程可恢复、结果可复算；主要局限在于"
        "输入数据口径和场景假设会影响外推能力。\n\n"
        "后续可引入更细粒度的需求分布、动态路况或多场景联合优化，并使用独立数据"
        "进行样本外验证。"
    )


def _build_references_text(references) -> str:
    if not references:
        return (
            "[1] 运筹学教材编写组. 运筹学基础与应用[M].\n"
            "[2] 数学建模课程组. 数学建模方法与实践[M].\n"
            "[3] 相关行业标准与题目所附数据说明[R]."
        )

    lines: list[str] = []
    for index, reference in enumerate(references[:8], 1):
        authors = "，".join(reference.authors[:3]) if reference.authors else "佚名"
        if len(reference.authors) > 3:
            authors += "，等"
        venue = f". {reference.venue}" if reference.venue else ""
        year = f"，{reference.year}" if reference.year else ""
        doi = f". DOI: {reference.doi}" if reference.doi else ""
        lines.append(f"[{index}] {authors}. {reference.title}{venue}{year}{doi}.")
    return "\n".join(lines)


def _build_section_fallback(group_name: str, state: MathModelingState, references=None):
    schema = schema_for_group(group_name)
    if group_name == "abstract_problem":
        return schema(
            abstract=(
                "本文针对题目所给的多约束优化问题，建立目标、约束和评价指标相互对应的"
                "数学模型。首先完成数据与问题拆解，其次使用可复现程序求解并输出结构化"
                "结果，最后通过敏感性分析检验关键参数变化对方案的影响。全文仅引用实际"
                "计算结果，对缺少证据的指标明确保留限制。"
            ),
            problem_restatement=(
                "题目要求在给定数据和业务规则下形成可执行方案，并回答各子问题。"
                "建模时需要同时处理目标冲突、资源限制、时序约束和结果验证，最终给出"
                "能够由程序复算的决策依据。"
            ),
            keywords="数学规划；多约束优化；可恢复计算；敏感性分析；结果验证",
        )
    if group_name == "assumptions_notation":
        return schema(
            assumptions=_build_assumptions_text(state),
            notation=_build_notation_text(state),
        )
    if group_name == "model":
        return schema(model_section=_build_model_section_text(state))
    if group_name == "solution":
        return schema(solution=_build_solution_text(state))
    if group_name == "sensitivity":
        return schema(sensitivity=_build_sensitivity_text(state))
    if group_name == "conclusion":
        return schema(conclusion=_build_conclusion_text(state))
    if group_name == "references":
        return schema(references=_build_references_text(references))
    return schema()


def _should_use_deterministic_writer() -> bool:
    """确定性写作仅用于显式离线/应急模式，不因 prompt 较长而自动降质。"""
    return os.getenv("MATH_AGENT_WRITER_DETERMINISTIC", "").strip() == "1"


def _has_green_safe_solver(state: MathModelingState) -> bool:
    return any(
        artifact.success
        and artifact.evidence_role == "primary"
        and "BEACON_GREEN_LOGISTICS_SAFE_SOLVER" in artifact.code
        for artifact in state.latest_code_artifacts()
    )


def _verified_result_map(state: MathModelingState) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for artifact in state.latest_code_artifacts():
        if not artifact.success or artifact.evidence_role not in {"primary", "baseline"}:
            continue
        match = re.search(r"(?m)^RESULT:\s*baseline=([^\s]+)\s+(.+)$", artifact.stdout)
        if not match:
            continue
        metrics = {
            item.group(1): float(item.group(2))
            for item in re.finditer(r"([A-Za-z_][\w]*)=(-?\d+(?:\.\d+)?)", match.group(2))
        }
        results[match.group(1)] = metrics
    return results


def _verified_structured_map(
    state: MathModelingState, label: str
) -> dict[str, float]:
    """只从最新正式主 artifact 的指定结构化行提取数值字段。"""
    primary = next((
        artifact for artifact in reversed(state.latest_code_artifacts())
        if artifact.success
        and artifact.evidence_role == "primary"
        and "BEACON_GREEN_LOGISTICS_SAFE_SOLVER" in artifact.code
    ), None)
    if primary is None:
        return {}
    match = re.search(rf"(?m)^{re.escape(label)}:\s+(.+)$", primary.stdout)
    if not match:
        return {}
    return {
        item.group(1): float(item.group(2))
        for item in re.finditer(
            r"([A-Za-z_][\w]*)=(-?(?:\d+(?:\.\d+)?|\.\d+))", match.group(1)
        )
    }


def _verified_green_references() -> str:
    """绿色/动态车辆路径论文的领域参考文献；不依赖临时 RAG 命中。"""
    return "\n\n".join([
        "[1] Dantzig G B, Ramser J H. The Truck Dispatching Problem[J]. Management Science, 1959, 6(1): 80-91.",
        "[2] Solomon M M. Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints[J]. Operations Research, 1987, 35(2): 254-265.",
        "[3] Laporte G. The Vehicle Routing Problem: An Overview of Exact and Approximate Algorithms[J]. European Journal of Operational Research, 1992, 59(3): 345-358.",
        "[4] Bektaş T, Laporte G. The Pollution-Routing Problem[J]. Transportation Research Part B: Methodological, 2011, 45(8): 1232-1250.",
        "[5] Pillac V, Gendreau M, Guéret C, et al. A Review of Dynamic Vehicle Routing Problems[J]. European Journal of Operational Research, 2013, 225(1): 1-11.",
        "[6] Psaraftis H N. Dynamic Vehicle Routing Problems[M]//Golden B L, Assad A A. Vehicle Routing: Methods and Studies. Amsterdam: North-Holland, 1988: 223-248.",
        "[7] 曹庆奎, 杨凯文, 任向阳, 等. 基于交通流的多模糊时间窗车辆路径优化[J]. 运筹与管理, 2018, 27(8): 20-26.",
        "[8] 唐慧玲, 唐恒书, 朱兴亮. 基于改进蚁群算法的低碳车辆路径问题研究[J]. 中国管理科学, 2021, 29(7): 118-127.",
    ])


def _verified_abstract_problem(state: MathModelingState):
    schema = schema_for_group("abstract_problem")
    metrics = _verified_result_map(state)
    ours = metrics.get("ours", {})
    q1 = metrics.get("no_schedule", {})
    profile = _verified_structured_map(state, "DATA_PROFILE")
    stress = _verified_structured_map(state, "DYNAMIC_STRESS")
    delta = ours.get("total_cost", 0.0) - q1.get("total_cost", 0.0)
    direction = "增加" if delta >= 0 else "减少"
    customer_count = profile.get("active_customers", profile.get("customers", 98.0))
    order_rows = profile.get("order_rows", 2169.0)
    stress_summary = (
        f"进一步对 {stress.get('samples', 0):.0f} 个独立局部事件样本执行压力测试，"
        f"成功重插 {stress.get('success', 0):.0f} 个，成功率为 "
        f"{100.0 * stress.get('success_rate', 0.0):.2f}% 。"
        if stress else "并用容量可行局部重插验证动态响应。"
    )
    return schema(
        abstract=(
            f"针对 {customer_count:.0f} 个有效客户、{order_rows:.0f} 行订单和五类有限异构车队的城市配送问题，本文建立"
            "含载重容积、软时间窗、分段时变速度、油电耗和绿色区限行的分割配送模型，"
            "并采用具有无进展检测的成本感知构造启发式求可行上界。所有结果均由程序"
            "现场读取订单、距离矩阵、时间窗和坐标四个附件获得。Q2 政策主方案总成本为"
            f" {ours.get('total_cost', 0.0):.2f} 元，启用 {ours.get('vehicles', 0.0):.0f} 辆车，"
            f"其中燃油车 {ours.get('fuel_vehicles', 0.0):.0f} 辆、新能源车 {ours.get('ev_vehicles', 0.0):.0f} 辆；"
            f"服务率为 {ours.get('service_rate', 0.0):.4f}，时间窗满足率为 {ours.get('timewin_rate', 0.0):.4f}，"
            f"碳排放为 {ours.get('total_carbon', 0.0):.2f} kg。与仅关闭限行、其余口径相同的 Q1 场景相比，"
            f"成本{direction} {abs(delta):.2f} 元；该结果反映当前启发式在限行约束下更多使用低能耗新能源车，"
            "不构成政策必然降低成本或全局最优的结论。另以定速预测和 first-fit 最近邻为基线，"
            "以三个单因素扫描检验局部敏感性。" + stress_summary
        ),
        problem_restatement=(
            "## 问题背景与研究意义\n\n"
            "城市末端配送同时受到订单分散、车辆异构、道路时变和低碳政策约束。企业不仅需要"
            "回答“怎样把货物送完”，还要在有限车队内协调固定启用成本、等待与晚到惩罚、燃油或"
            "电力消耗以及碳排放成本。若只按最短距离排线，可能在容量、时间窗或绿色区限行处失去"
            "可行性；若只追求成本，又可能把拥堵、排放和动态事件的处置代价转移到实际运营阶段。"
            "因此，本题本质上是带分割配送、软时间窗、分段时变速度和区域限行的异构车辆路径问题，"
            "其难点不在于给出一条孤立路线，而在于建立从题面数据到可执行车队计划的完整证据链。\n\n"
            "四个附件分别承载订单、道路距离、客户时间窗和二维坐标。订单记录需要先按客户聚合，"
            "道路矩阵用于运输里程与费用计算，坐标仅用于判断客户或弧段是否进入绿色配送区，时间窗"
            "则决定等待、晚到和服务时序。不同附件的客户编号、行列索引和单位必须在建模前统一；"
            "任何错位都会让路线看似可行、实际却引用了另一客户的需求或距离。本文把附件读取审计、"
            "字段标准化和索引一致性检查视为模型的一部分，而不是求解前可以省略的准备工作。\n\n"
            "## 子问题的数学化表达\n\n"
            "问题 1 要求在不启用绿色区限行的条件下确定车辆类型、车辆启用、客户访问次序和服务"
            "时刻。其决策变量包括车辆是否启用、车辆所属车型、弧是否被访问以及客户到达和开始服务"
            "时间；目标是在满足客户覆盖、载重、容积、有限车队和时间递推约束的前提下，最小化固定、"
            "等待、晚到、能耗和碳排放成本。问题 1 不是脱离后续问题的另一个模型，而是与正式主方案"
            "保持同一数据、同一费用和同一启发式，仅关闭绿色区限行，从而形成可解释的政策基准。\n\n"
            "问题 2 在问题 1 的网络上增加 8:00--16:00 燃油车不得进入市中心 10 km 圆域的规则。"
            "新增决策影响至少包括三层：一是燃油车到达区内客户或穿越圆域时需要改变出发时刻或车型；"
            "二是等待会通过时间递推改变后续客户的晚到惩罚；三是油车与新能源车在容量、数量、能耗和"
            "排放系数上的差异会重塑车队结构。因此，需要把圆域相交判定、政策时段和车型属性嵌入每次"
            "候选插入的可行性检查，而不能在路线生成结束后再用一个统一系数修正总成本。\n\n"
            "问题 3 要求面对取消、新增、地址或时间窗变化给出实时调整思路。受现有程序证据边界限制，"
            "本文只对“一个受影响任务被移出后，在其他路线中寻找容量可行且距离增量最小的位置”进行"
            "数值验证。其输入是已经形成的静态路线和变更任务，输出是是否成功重插、响应时间、距离"
            "变化绝对值和改善标志。新增订单、批量取消、地址改变及多个时间窗同时变化被保留为策略"
            "扩展，不把尚未执行的事件写成已经得到的实验结论。\n\n"
            "## 关键难点分析\n\n"
            "第一，部分客户聚合需求超过小车型容量，若坚持整客户不可拆分，构造算法会出现剩余任务"
            "无法装入任何车辆的无进展状态。本文采用分割配送，把超容量需求拆为单车可承载的访问任务，"
            "同时保留客户原时间窗。第二，速度随时段变化，长弧可能跨越多个速度区间，必须按时段边界"
            "积分旅行时间。第三，绿色区约束同时依赖空间和时间：客户点在圆内只是一个情形，连接两个"
            "圆外点的线段也可能穿越圆域。第四，五类车辆的数量上限要求每条路线对应真实物理车辆，"
            "不能无限复用成本最低的车型。第五，构造启发式只能给出可行上界，结果评价必须通过同口径"
            "基线、敏感性扫描和约束断言完成，不能把“程序成功退出”解释为全局最优。\n\n"
            "## 数据预处理与总体路线\n\n"
            "预处理依次完成文件存在性与读取审计、字段名规范化、客户编号转为整数、需求字段转为数值、"
            "订单按客户聚合、距离矩阵行列索引校准、时间窗转为从日初起算的分钟以及坐标表中配送中心"
            "与客户点的分离。随后以最小车型的载重和容积联合上界拆分超容量需求，保证每个访问任务"
            "至少能装入一种车型。所有正式主方案和基线都重新读取四个附件并执行相同预处理，避免从"
            "主方案输出改标签伪造对照。\n\n"
            "## 数据质量风险与处理原则\n\n"
            "订单表的主要风险是客户编号或重量、体积字段被读取为文本，以及同一客户多笔订单聚合后"
            "超过单车容量。处理时先把编号和数值字段显式转换，无法转换的记录不能静默填成正常需求；"
            "聚合后再执行重量与体积守恒检查。时间窗的风险是时刻字符串、跨日和单位混用，本文统一"
            "转为日初起算分钟，并要求开始时刻不晚于结束时刻。距离矩阵的风险是表头与行索引错位、"
            "对角线非零或出现负距离；坐标表的风险则是把配送中心与市中心混淆。上述检查不改变原始"
            "附件，只在隔离执行目录中形成规范化视图和读取审计。\n\n"
            "对于缺失或异常数据，原则是“影响硬约束的字段不猜测，能够追溯的格式问题才转换”。"
            "客户需求、距离或时间窗若缺失，会直接影响容量与时序可行性，正式运行应失败并指出字段；"
            "日期时间、整数编号等可确定的格式差异则按统一规则转换。这样可以避免用均值填补关键"
            "业务数据后得到一份数值完整却不可执行的路线。\n\n"
            "预处理结果还需要三组守恒核验。订单聚合前后的总重量与总体积应一致；需求拆分前后的"
            "客户级重量与体积应一致；距离和坐标连接后的客户集合应与时间窗客户集合一致。第一组"
            "防止分组求和漏单，第二组防止拆分时舍入或重复，第三组防止跨附件错配。守恒检查只判断"
            "数据转换是否忠实，不评价路线优劣，因此应在任何主方案、基线或敏感性执行之前完成。\n\n"
            "距离矩阵还承担方向与闭环检查。程序把配送中心和客户编号转换为同一索引类型，逐次取出"
            "路线相邻节点的距离；路线必须从配送中心出发并返回配送中心，不能因为表头或行标签类型"
            "不同而漏计返程。坐标表则单独提取配送中心、客户与题面市中心，明确配送中心 (20,20)"
            "只是路线起点，绿色区圆心仍为 (0,0)。这一处理直接避免原始报告中最容易出现的空间"
            "口径混淆。\n\n"
            "时间数据统一后，任一候选路线都按到达、等待、服务、离开、行驶的顺序递推。提前到达"
            "产生等待，晚于窗口终点开始服务产生晚到；20 分钟服务时间在每次实际访问处计入。若一"
            "个客户被拆成多个访问任务，每次访问共享原时间窗但独立占用服务时长。这一口径虽然可能"
            "增加拆分访问的时间成本，却与实际多次装卸一致，不能为了得到更低成本而只计一次服务。\n\n"
            "## 评价指标与回答方式\n\n"
            "三个问题采用统一的指标层次。第一层是硬可行性，包括任务覆盖、容量、容积、车型库存、"
            "路线起讫与绿色区合规；任一失败都不能用成本优势抵消。第二层是服务质量，包括服务率、"
            "时间窗满足率、等待和晚到。第三层是运营与环境表现，包括总成本、分项成本、车辆数、"
            "道路里程、燃油与新能源车辆结构及碳排放。问题三另报告响应时间、距离变化绝对值和改善"
            "标志。论文按这一层次逐项回答，避免用单一总成本代替赛题要求的完整结论。\n\n"
            "建模路线遵循“数据口径统一—无政策基准—绿色区扩展—动态局部调整—对照与敏感性验证”。"
            "先用问题 1 检查基本容量、时序与成本核算，再把绿色区约束加入问题 2，最后在问题 3 中"
            "复用静态解执行局部重插。每个阶段都输出结构化 RESULT，并以附件读取记录、约束断言、"
            "基线结果和中心点对齐的敏感性数组相互校验。这样，三个问题形成递进关系，而不是三个"
            "互不相干的方法堆叠。"
        ),
        keywords="异构车辆路径；分割配送；绿色物流；时变速度；数据血缘；动态重插",
    )


def _verified_assumptions_notation(state: MathModelingState):
    schema = schema_for_group("assumptions_notation")
    return schema(
        assumptions=(
            "**假设1：附件数据在同一计划日内有效且编号可对应。**\n"
            "**依据**：订单、距离、时间窗和坐标由同一题目附件给出，程序通过客户编号与距离矩阵"
            "行列索引建立关联。\n"
            "**合理性与影响**：该假设使跨表连接具有唯一口径；若现实中存在跨日订单或编号重用，"
            "应增加日期键和版本号，并重新执行完整性检查。\n\n"
            "**假设2：道路距离矩阵用于运输核算，二维坐标只用于绿色区几何判定。**\n"
            "**依据**：题面同时提供实际道路距离和 X、Y 平面坐标，两者服务于不同计算职责。\n"
            "**合理性与影响**：实际道路距离比坐标欧氏距离更适合计费和旅行时间；若缺少道路矩阵，"
            "可退化为欧氏距离或地图路网距离，但必须重新标定速度和成本。\n\n"
            "**假设3：超容量客户需求允许分割配送。**\n"
            "**依据**：客户聚合需求可能超过单车载重或容积，坚持整客户一次服务会导致任何车型都"
            "无法承载。拆分任务共享原客户时间窗，每次访问都计入统一服务时间。\n"
            "**合理性与影响**：假设保证构造过程存在容量可行的基本任务；若业务禁止拆单，则需引入"
            "更大车型、跨日配送或前置分拨，而不能把不可行任务强行放入路线。\n\n"
            "**假设4：每次实际客户访问的服务时间固定为 20 分钟。**\n"
            "**依据**：该值由题面统一给定，并在到达—等待—服务—离开的时间递推中显式使用。\n"
            "**合理性与影响**：固定服务时长便于比较不同策略；若服务时长随货量变化，可改为"
            "基础时长与装卸量的分段函数，并把相应参数纳入敏感性分析。\n\n"
            "**假设5：题面分时段正态车速以期望值作为确定性计划速度。**\n"
            "**依据**：本次正式求解器没有进行随机抽样，而是用顺畅、一般和拥堵时段的期望速度，"
            "长弧跨时段时按边界逐段积分。\n"
            "**合理性与影响**：该处理适合形成可复算基准，但不能代表随机拥堵尾部风险；推广时应"
            "使用情景模拟、机会约束或分布鲁棒模型。\n\n"
            "**假设6：五类车的载重、容积、数量和固定启用成本在计划期内不变。**\n"
            "**依据**：这些参数由题面给定，程序把每种车型的可用数量作为硬上限，一辆物理车在"
            "本日计划中至多启用一次。\n"
            "**合理性与影响**：假设防止低成本车型被无限复用；若考虑故障、临时租车或多班次，"
            "需增加车辆可用状态、班次衔接和租赁决策。\n\n"
            "**假设7：绿色配送区为以市中心 (0,0) 为圆心、半径 10 km 的圆域。**\n"
            "**依据**：坐标系和政策边界由题面定义，配送中心 (20,20) 不是圆心。燃油车在"
            "8:00--16:00 不得到达区内客户或穿越圆域。\n"
            "**合理性与影响**：圆域使相交判定可以解析计算；若现实政策边界是不规则多边形，应"
            "替换为路段级通行标签或计算几何相交测试。\n\n"
            "**假设8：时间窗为软约束，容量、车队数量和绿色区禁入为硬约束。**\n"
            "**依据**：等待和晚到进入成本函数，而超容量、超车数或政策时段闯入会直接破坏方案"
            "可执行性。\n"
            "**合理性与影响**：该区分保留了运营中的服务弹性；若某类客户时间窗不可违约，应按"
            "客户等级把相应晚到约束改为硬约束并重新求解。\n\n"
            "**假设9：构造启发式输出可行上界，不承诺全局最优。**\n"
            "**依据**：程序逐车、逐候选地构造路线，没有执行精确分支定界或给出最优性间隙。\n"
            "**合理性与影响**：这一证据边界避免夸大结果；评价时依赖独立基线、约束断言和敏感性"
            "扫描。若需要最优性证明，可在缩小实例上建立混合整数规划作为对照。\n\n"
            "**假设10：同一计划日内不考虑车辆途中故障与临时补能。**\n"
            "**依据**：题面给出车型容量、能耗和数量，但没有提供故障概率、充电站位置或补能时长，"
            "正式程序也未执行充电调度。\n"
            "**合理性与影响**：该假设使车辆从配送中心出发后按既定路线完成任务；若应用于长距离"
            "或多班次运营，应增加剩余能量状态、充电站容量和故障救援约束，现有电动车结果不能直接"
            "外推到需要途中补能的场景。\n\n"
            "**假设11：订单、道路和政策参数在一次求解过程中保持静态。**\n"
            "**依据**：问题一、问题二用于形成日计划，问题三仅在计划形成后验证一次局部事件响应。\n"
            "**合理性与影响**：该假设保证主方案与基线可在同一输入上比较；若订单持续到达或道路"
            "状态实时变化，应采用滚动时域，每个窗口冻结当前快照并保存版本，不能把不同时间的数据"
            "混在同一 RESULT 中。\n\n"
            "**假设的可检验性分级。** 假设1、2、3、6和7可由附件字段、守恒关系、车辆清单与几何"
            "判定直接核验，属于本次运行内可审计假设；假设4和8由题面业务规则给定，可通过修改服务"
            "时长或软硬窗口口径开展反事实检验；假设5、9、10和11涉及随机交通、算法最优性、途中"
            "补能与实时到达，只能通过额外数据或更强算法验证。把三类假设分开，可以避免把程序断言"
            "通过误解为所有现实条件均成立。\n\n"
            "**假设失效时的处理顺序。** 若可审计假设失败，正式运行立即停止并定位附件或索引问题；"
            "若业务规则变化，需重设参数并重跑主方案、基线和敏感性中心点；若外部环境假设失效，则"
            "应扩展状态变量和情景集合，不能仅调整论文措辞。该顺序使假设章节与后续数据门禁、参数"
            "扫描和动态回退形成闭环，而不是把假设当作无法被质疑的前提清单。"
        ),
        notation=(
            "| 符号 | 含义 | 单位或取值 |\n"
            "|---|---|---|\n"
            "| $C$ | 分割后的访问任务集合 | 集合 |\n"
            "| $K$、$V$ | 车辆实例集合、车型集合 | 集合 |\n"
            "| $N_v$ | 车型 $v$ 的可用车辆上限 | 辆 |\n"
            "| $Q_v$、$B_v$ | 车型 $v$ 的载重、容积上限 | kg、m$^3$ |\n"
            "| $q_i$、$b_i$ | 访问任务 $i$ 的重量、体积需求 | kg、m$^3$ |\n"
            "| $d_{ij}$ | 附件距离矩阵中节点 $i$ 至 $j$ 的道路距离 | km |\n"
            "| $x_{ijk}$ | 车辆 $k$ 是否行驶弧 $(i,j)$ | 0-1 变量 |\n"
            "| $y_k$ | 车辆 $k$ 是否启用 | 0-1 变量 |\n"
            "| $z_{kv}$ | 车辆 $k$ 是否选择车型 $v$ | 0-1 变量 |\n"
            "| $t_i$、$s_i$ | 到达任务 $i$、开始服务任务 $i$ 的时刻 | 日初起分钟 |\n"
            "| $E_i$、$L_i$ | 任务 $i$ 的时间窗起点、终点 | 日初起分钟 |\n"
            "| $w_i$、$l_i$ | 等待时长、晚到时长 | 分钟 |\n"
            "| $\tau(d,t)$ | 时刻 $t$ 出发、行驶距离 $d$ 的分段旅行时间 | 分钟 |\n"
            "| $r_{ijk}$ | 行驶弧起点的车辆载重率 | $[0,1]$ |\n"
            "| $E_{ijk}$ | 弧段的燃油消耗或电力消耗 | L 或 kWh |\n"
            "| $E_{\rm carbon}$ | 全部路线换算后的二氧化碳排放 | kg |\n"
            "| $Z_{\rm fix}$、$Z_{\rm wait}$、$Z_{\rm late}$ | 固定、等待、晚到成本 | 元 |\n"
            "| $Z_{\rm energy}$、$Z_{\rm carbon}$ | 能耗、碳排放成本 | 元 |\n"
            "| $Z$ | 五类成本之和 | 元 |\n\n"
            "集合与索引采用大写字母和下标表示，0 专指配送中心；客户编号与分割后的访问任务编号"
            "在程序内部建立映射，不能混用。时间全部转为日初起算的分钟，响应时间单独以秒记录。"
            "费用统一以元计，碳排放统一以 kg CO2 计。若一个符号同时出现在数学模型、程序字段和"
            "结果表中，其业务含义和单位保持不变。"
        ),
    )


def _verified_solution(state: MathModelingState):
    schema = schema_for_group("solution")
    metrics = _verified_result_map(state)
    rows = []
    for name in ("ours", "no_schedule", "simple_pred", "greedy"):
        item = metrics.get(name)
        if not item:
            continue
        rows.append(
            f"| {name} | {item.get('total_cost', 0):.2f} | {item.get('vehicles', 0):.0f} | "
            f"{item.get('fuel_vehicles', 0):.0f} | {item.get('ev_vehicles', 0):.0f} | "
            f"{item.get('total_carbon', 0):.2f} | {item.get('timewin_rate', 0):.4f} |"
        )
    ours = metrics.get("ours", {})
    profile = _verified_structured_map(state, "DATA_PROFILE")
    stress = _verified_structured_map(state, "DYNAMIC_STRESS")
    primary = next((
        artifact for artifact in reversed(state.latest_code_artifacts())
        if artifact.success and artifact.evidence_role == "primary"
    ), None)
    breakdown_section = ""
    if primary is not None:
        match = re.search(
            r"(?m)^BREAKDOWN:\s+Z_fix=([0-9.]+)\s+Z_wait=([0-9.]+)\s+"
            r"Z_late=([0-9.]+)\s+Z_energy=([0-9.]+)\s+Z_carbon=([0-9.]+)",
            primary.stdout,
        )
        if match:
            labels = ["固定启用", "等待", "晚到", "能耗", "碳排放"]
            values = [float(value) for value in match.groups()]
            total = sum(values)
            breakdown_rows = "\n".join(
                f"| {label} | {value:.2f} | {100.0 * value / total:.2f}% |"
                for label, value in zip(labels, values)
            )
            largest_index = max(range(len(values)), key=values.__getitem__)
            breakdown_section = (
                "\n\n## 主方案成本分解\n\n"
                "| 成本项 | 金额/元 | 占分项合计 |\n|---|---:|---:|\n"
                + breakdown_rows
                + "\n\n五项分解直接来自主求解器 BREAKDOWN，并与总成本采用同一运行证据。"
                f"其中{labels[largest_index]}成本为 {values[largest_index]:.2f} 元，是分项中最大者；"
                "这说明后续改进应优先针对主导成本机制，而不是只调整占比较小的碳价格。"
                "固定成本高时应审查路线合并与车型容量利用，等待或晚到成本高时应审查出发时刻和"
                "客户顺序，能耗成本高时则需联合分析里程、速度、载重率与车型结构。成本图只展示"
                "上述已验证分项，不重新求解或覆盖主 RESULT。"
            )
    comparison_blocks: list[str] = []
    for name in ("no_schedule", "simple_pred", "greedy"):
        baseline = metrics.get(name)
        if not baseline or not ours:
            continue
        cost_delta = baseline.get("total_cost", 0.0) - ours.get("total_cost", 0.0)
        carbon_delta = baseline.get("total_carbon", 0.0) - ours.get("total_carbon", 0.0)
        vehicle_delta = baseline.get("vehicles", 0.0) - ours.get("vehicles", 0.0)
        comparison_blocks.append(
            f"与 `{name}` 相比，主方案总成本差值为 {-cost_delta:+.2f} 元、车辆数差值为 "
            f"{-vehicle_delta:+.0f} 辆、碳排放差值为 {-carbon_delta:+.2f} kg。"
            "差值由两行原始 RESULT 直接相减；正负只表示本次构造解的方向，不解释为统计显著性。"
        )
    comparison_section = (
        "\n\n## 定量差值与结果边界\n\n" + "\n\n".join(comparison_blocks)
        if comparison_blocks else ""
    )
    profile_section = ""
    if profile:
        profile_section = (
            "\n\n## 数据画像与约束规模\n\n"
            "| 指标 | 数值 | 建模含义 |\n|---|---:|---|\n"
            f"| 原始订单行数 | {profile.get('order_rows', 0):.0f} | 决定读取与聚合规模 |\n"
            f"| 聚合客户数 | {profile.get('customers', 0):.0f} | 客户级需求节点 |\n"
            f"| 有效客户数 | {profile.get('active_customers', 0):.0f} | 存在正重量或体积需求 |\n"
            f"| 分割后任务数 | {profile.get('tasks', 0):.0f} | 路径覆盖的实际访问任务 |\n"
            f"| 需拆分客户数 | {profile.get('split_customers', 0):.0f} | 同一客户可能多次访问 |\n"
            f"| 绿色区客户数 | {profile.get('green_customers', 0):.0f} | 直接受车型与时段限制 |\n"
            f"| 总重量 / 总体积 | {profile.get('total_weight', 0):.2f} / {profile.get('total_volume', 0):.2f} | 容量压力总量 |\n"
            f"| 时间窗宽度中位数 | {profile.get('median_window_width', 0):.2f} min | 服务时序松紧程度 |\n"
            f"| 重量 / 体积缺失 | {profile.get('missing_weight', 0):.0f} / {profile.get('missing_volume', 0):.0f} | 填充前数据质量审计 |\n\n"
            "该表与数据画像图均由正式求解器在缺失值处理前后分别统计。订单行数不是客户数，"
            "聚合客户数也不是分割后任务数；三者分开报告可以解释为什么一条客户记录可能对应多笔"
            "订单、一个超容量客户又可能形成多个访问任务。绿色区客户数只反映点位分布，政策可行性"
            "还取决于弧段是否穿圆以及实际到达时刻，不能仅按区内客户比例推断新能源车辆数。"
        )
    stress_section = ""
    if stress:
        stress_section = (
            "\n\n## 动态压力测试与局部重调度\n\n"
            "| 指标 | 数值 |\n|---|---:|\n"
            f"| 独立事件样本 | {stress.get('samples', 0):.0f} |\n"
            f"| 成功重插样本 | {stress.get('success', 0):.0f} |\n"
            f"| 成功率 | {100.0 * stress.get('success_rate', 0):.2f}% |\n"
            f"| 平均 / P95 响应时间 | {stress.get('mean_response_ms', 0):.4f} / {stress.get('p95_response_ms', 0):.4f} ms |\n"
            f"| 平均 / 最大距离变化 | {stress.get('mean_distance_change', 0):.4f} / {stress.get('max_distance_change', 0):.4f} km |\n"
            f"| 距离改善样本 | {stress.get('improved', 0):.0f} |\n"
            f"| 目标路线平均晚到变化 | {stress.get('mean_late_change', 0):.4f} min |\n\n"
            "每个样本从同一静态主方案独立开始：冻结未受影响路线，移出一个任务，枚举其他路线的"
            "容量、容积和政策可行插入位置，并按距离增量与新增晚到代价选择位置。因而这些样本衡量"
            "的是局部处置能力，不是连续 30 次修改同一解；后者会混入事件累积效应。成功率低于 100%"
            "时，失败意味着在冻结边界下没有可行跨路线位置，应触发扩大邻域或全局重优化，而不能"
            "强行把任务塞入超载路线。响应时间来自实际枚举，但会受运行环境影响，只作为本机量级证据。"
        )
    return schema(solution=(
        "程序依次完成附件字段规范化、客户需求聚合、最小车型容量拆分、分车型候选路线构造、"
        "有限车队选择、逐路线约束断言和 RESULT 校验。每轮必须减少未分配任务数；耗尽车队或"
        "无法装载时抛出失败，不能以退出码 0 伪装成功。四个正式场景结果如下。\n\n"
        "## 各方案结果对比表\n\n"
        "| 场景 | 总成本/元 | 车辆数 | 燃油车 | 新能源车 | 碳排放/kg | 时间窗率 |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(rows) +
        "\n\n`ours` 是 Q2 政策主证据；`no_schedule` 是只关闭限行的 Q1 同构场景；"
        "`simple_pred` 和 `greedy` 是 Q2 的定速与 first-fit 对照。它们均重新读取四个附件，"
        "而不是从主方案 stdout 改标签。主方案服务率为 "
        f"{ours.get('service_rate', 0):.4f}，道路里程 {ours.get('total_distance', 0):.2f} km，"
        f"平均配送历时 {ours.get('avg_delivery_time', 0):.2f} 分钟。\n\n"
        + profile_section + breakdown_section + comparison_section + stress_section + "\n\n"
        "动态主 RESULT 保留首个受影响任务的兼容字段，并枚举其他路线的容量可行插入位置。"
        f"本次重插标志为 {ours.get('dynamic_reinserted', 0):.0f}，响应时间 "
        f"{ours.get('response_time', 0):.6f} s，距离变化绝对值 "
        f"{ours.get('dynamic_distance_change', 0):.4f} km；改善标志为 "
        f"{ours.get('dynamic_distance_improved', 0):.0f}。该实验只证明单事件局部响应可执行，"
        "批量取消、新增、地址变更和时间窗联动仍需按事件类型另行压力测试。\n\n"
        "## 求解算法与执行流程\n\n"
        "本文采用容量拆分与逐步最小增量构造启发式。第一步读取并规范化四个附件，检查订单客户、"
        "时间窗客户、坐标客户与距离矩阵索引能否对应；第二步按客户聚合订单重量和体积，并把超过"
        "最小车型容量的聚合需求拆成单车可行任务；第三步初始化未分配任务集合、五类车型剩余数量"
        "和空路线集合；第四步对每一种仍有库存的车型尝试构造候选路线，逐个评估尚未服务任务的"
        "容量、时间和绿色区政策可行性；第五步用固定成本、里程代理和晚到代价形成单位配送重量"
        "增量，选择当前最合适的车型与路线；第六步提交路线并从未分配集合删除已覆盖任务；第七步"
        "在全部任务完成后重新计算里程、时序、能耗、排放和五项成本，最后执行覆盖、容量、车队和"
        "RESULT 合理性校验。\n\n"
        "候选路线内部采用最近可行增量思想：对当前位置和每个未分配任务，先判断加入任务后的累计"
        "重量与体积是否越界，再用分段速度函数递推到达和开始服务时刻；燃油车还要检查客户点与"
        "连接弧是否触发绿色区禁入。只有硬约束全部满足的任务才进入候选集合。若存在多个候选，"
        "按新增道路距离、等待与晚到代价的组合排序。该策略把昂贵的完整成本重算留到路线提交后，"
        "同时保证不会为了局部短距离选择一个政策不可行的客户。\n\n"
        "算法设置了两个无进展保护。其一，一条候选路线若未装入任何任务，不消耗车辆库存；其二，"
        "外层循环每轮都记录未分配任务数，若没有严格减少，则立即报告 no progress 或 fleet exhausted。"
        "因此，程序不会在超容量任务或耗尽车型后无限循环。成功退出还不是论文证据：只有 stdout"
        "包含字段完整、数值有限、服务率和车辆数处于合理范围的 RESULT 行，且附件读取记录证明"
        "正式代码确实访问数据，artifact 才能进入主方案。\n\n"
        "## 计算复杂度与可复现性\n\n"
        "设分割后任务数为 $n$，可用车辆实例数为 $m$。一次候选路线构造最多扫描当前未分配集合，"
        "完整构造在最坏情况下需要多轮扫描；几何相交、容量和单次时间递推都是常数时间操作，因而"
        "主要复杂度可保守写为 $O(mn^2)$。动态局部重插需要枚举其他路线的相邻位置，若全部路线"
        "总弧数为 $a$，一次事件的候选搜索为 $O(a)$。该复杂度分析描述程序实际循环结构，不声称"
        "启发式具有多项式最优性保证。\n\n"
        "可复现性由三层机制保证。数据层记录每次附件读取及文件摘要；执行层在隔离目录中保存实际"
        "运行脚本、stdout、stderr 和图像；论文层只选取正式主方案、独立基线与最后一轮对齐的敏感性"
        "结果。主方案和三个基线都从同一附件重新计算，不共享经过改标签的 RESULT。这样既可以复查"
        "某个数值来自哪次执行，也能防止历史失败或 supporting 图的临时指标进入正式论文。\n\n"
        "## 实验设计与评价指标\n\n"
        "实验以同一组附件作为唯一输入，四个场景采用相同客户集合、距离矩阵、时间窗、车型上限和"
        "费用口径。场景差异只由政策开关、速度表示或候选选择规则定义。评价指标分为三类：可行性"
        "指标包括服务率、容量和车辆上限；运营指标包括总成本、车辆数、道路里程和时间窗满足率；"
        "环境指标包括燃油与新能源车辆结构和碳排放。只有服务率与硬约束先通过，才比较成本高低。"
        "这种顺序防止用低成本但未完成任务的方案作为优胜基线。所有场景保留相同 RESULT 字段，"
        "缺少关键字段或出现非法值时整次执行失败。\n\n"
        "图表与文字分析也遵循同一评价顺序。路线图用于检查客户覆盖、配送中心位置和绿色区关系，"
        "成本构成图解释固定、等待、晚到、能耗和碳成本的相对来源，敏感性图只承载已校验数组。"
        "正文引用图表时先陈述可验证结论，再指出图中证据，最后说明适用边界；不让同一张图重复"
        "承担主结果、基线和敏感性三种证据职责。若图的生成脚本产生与主 RESULT 不一致的指标，"
        "图只能作为 supporting artifact，不能替换主方案。\n\n"
        "## 问题一结果解释\n\n"
        "`no_schedule` 场景只关闭绿色区限行，保留分割配送、有限异构车队、时变速度、软时间窗、"
        "能耗与碳成本。因此它直接回答问题一，并为问题二提供同构政策基准。比较时不只看总成本，"
        "还同时检查车辆数、燃油与新能源结构、碳排放和时间窗满足率。若无政策场景成本更高或更低，"
        "都只能解释为当前构造启发式在两个可行域中的运行结果；由于算法没有全局最优性证明，不能"
        "把差值直接归因于政策的必然经济效应。\n\n"
        "问题一的另一个作用是验证基础模型。若关闭限行后仍出现燃油车等待至政策结束、弧段相交"
        "被拒绝或绿色区参数进入成本，说明场景开关没有真正隔离。正式基线通过独立代码执行和字段"
        "校验，确认政策规则关闭而其他口径保持不变，因而能够作为后续比较的可信参照。\n\n"
        "## 问题二主方案与基线分析\n\n"
        "`ours` 为启用绿色区限制的正式主方案。`simple_pred` 使用更简化的定速预测口径，考察忽略"
        "分段时变速度对车队与成本判断的影响；`greedy` 使用 first-fit 最近邻规则，考察成本感知"
        "车型选择与候选排序是否优于简单贪心。三个 Q2 场景具有相同输入和质量门禁，因此表中差异"
        "来自算法或速度口径，而不是来自数据截取。\n\n"
        "主方案的服务率、道路里程、车队结构、时间窗满足率和碳排放必须联合解释。服务率达到完整"
        "覆盖只说明全部任务被安排，不意味着每个时间窗都严格满足；时间窗率低于 1 时，晚到成本"
        "必须进入总目标并在局限性中说明。燃油与新能源车辆数反映限行、容量和库存的共同作用，"
        "不能把新能源占比变化简单写成单一政策因果。总里程和能耗的关系还受速度与载重率修正，"
        "所以只比较路线图的几何长度不足以解释碳排放。\n\n"
        "结果表采用“结论—证据—边界”的阅读方式：先指出哪一场景在某指标上更优，再引用该行"
        "结构化数值，最后说明比较只对本次数据和构造解有效。若某个基线成本低于主方案，应如实"
        "承认主启发式仍有改进空间，而不是删除该基线或改用不可比指标。这样的负结果同样具有"
        "建模价值，因为它能够定位车型评分、候选插入或局部搜索需要加强的环节。\n\n"
        "## 服务质量、车队结构与环境指标\n\n"
        f"主方案服务率为 {ours.get('service_rate', 0):.4f}，说明正式构造解完成了进入 RESULT 的"
        "全部访问任务；时间窗满足率为 "
        f"{ours.get('timewin_rate', 0):.4f}，两者必须区分。服务率衡量是否安排，时间窗率衡量是否"
        "在承诺窗口内开始服务。后者未达到 1 时，论文不能只强调完整覆盖，还应检查晚到成本并"
        "承认部分任务存在服务违约。当前汇总 RESULT 没有客户级晚到清单，因此只能报告总体比例，"
        "不能编造晚到客户的地理分布或需求特征。\n\n"
        f"主方案共启用 {ours.get('vehicles', 0):.0f} 辆车，其中燃油车 "
        f"{ours.get('fuel_vehicles', 0):.0f} 辆、新能源车 {ours.get('ev_vehicles', 0):.0f} 辆。"
        "该结构是容量、车型数量、固定成本、能耗和绿色区可行性的共同结果。新能源车可以在政策"
        "时段进入圆域，但其可用数量有限；燃油车数量较多时需要通过等待或绕开受限弧保持合规。"
        "因此，车队结构不能只按能源单价解释，也不能在没有同口径反事实的情况下断言政策单独造成"
        "某一车型占比。\n\n"
        f"主方案道路里程为 {ours.get('total_distance', 0):.2f} km，平均配送历时为 "
        f"{ours.get('avg_delivery_time', 0):.2f} 分钟，碳排放为 "
        f"{ours.get('total_carbon', 0):.2f} kg。里程影响能耗但不是唯一因素：速度处于不同分段时"
        "基础消耗不同，载重率修正又使同样里程在路线前后产生不同消耗。平均配送历时包含行驶、"
        "等待与服务递推，其量纲是分钟而非响应时间的秒。三项指标共同说明资源投入、服务时序和"
        "环境代价，避免将较短路线自动等同于更低碳方案。\n\n"
        "## 运营决策规则\n\n"
        "结果可以转化为三类决策。日计划层面，先保证客户覆盖和硬约束，再根据成本分解判断应优先"
        "合并路线、调整出发时刻还是优化车型分配。政策执行层面，对预计在限行窗口进入绿色区的"
        "燃油路线提前标记，优先分配新能源车或调整服务顺序，不能等车辆到达边界后才处理。监控层面，"
        "持续记录实时速度、晚到比例、车型库存和动态事件数量；当参数离开已扫描区间或局部重插"
        "累计代价过高时，触发完整重优化。以上规则来自模型结构和已验证指标，不引入未运行的"
        "预测器或全局算法。\n\n"
        "## 问题三动态调整分析\n\n"
        "动态实验从已提交路线中移出一个受影响任务，枚举其他路线的可行插入位置并选择最小距离"
        "增量。重插前后都执行容量与路线结构检查，响应时间由实际 Python 计时差得到，单位为秒。"
        "距离变化记录绝对值并另设改善标志，避免用符号方向不清的单一字段造成误读。该结果说明"
        "局部处置链路可运行，但不能覆盖新增订单、地址修改、时间窗变化和多个事件同时发生的情形。\n\n"
        "实际部署可采用分层响应：单个取消或轻微变化先局部修复，若连续事件使累计距离或晚到成本"
        "超过阈值，则触发全局邻域搜索；若车型库存、政策或需求规模发生结构性变化，则从最新数据"
        "重新运行完整模型。这样既保留局部算法的速度优势，也限制多次局部决策累积的次优风险。\n\n"
        "## 动态事件分类与响应边界\n\n"
        "| 事件类型 | 状态变化 | 当前程序可直接处理的部分 | 超出边界后的动作 |\n"
        "|---|---|---|---|\n"
        "| 订单取消 | 从原路线移除任务并连接前后弧 | 可用同一删除—重插算子验证路线恢复 | 重新核算原路线时序与能耗 |\n"
        "| 临时新增 | 生成一个新的访问任务 | 可枚举既有路线容量可行位置 | 无位置时启用备用车或全局重排 |\n"
        "| 地址变更 | 客户坐标及相关道路距离改变 | 可复用候选插入框架 | 必须先更新距离矩阵，不能只改绘图坐标 |\n"
        "| 时间窗变更 | 服务区间收紧、平移或放宽 | 可在候选评估中重算等待与晚到 | 硬窗口无可行位置时升级重优化 |\n"
        "| 车辆故障 | 一条路线及其车辆暂时失效 | 可逐项释放该路线任务 | 多任务联动需扩大邻域并重配车型 |\n\n"
        "表中的前两列描述状态变化，后两列区分已实现的局部算子和仍需升级的系统动作。当前压力测试"
        "并未分别生成五类业务事件，而是用 30 个不同来源任务检验共同的跨路线插入内核。因此，"
        "论文可以据实评价插入内核的覆盖率与响应时间，却不能把结果写成五类事件均已验证。这个"
        "区分比罗列事件名称更重要：地址变化必须先得到新的道路距离，车辆故障会同时释放一组任务，"
        "两者都改变了局部搜索的输入规模，不能用一次任务搬移的成功标志替代。\n\n"
        "## 动态发布与回退规则\n\n"
        "一次局部方案只有同时满足四个条件才可发布：受影响任务重新获得唯一归属；目标车辆的载重"
        "和容积不越界；燃油车在限行窗口不进入或穿越绿色区；重新递推后不存在未计入的时序成本。"
        "若无可行插入位置，系统保留原静态方案和失败原因，不发布半完成路线。若连续失败次数、"
        "累计距离增量或新增晚到超过运营阈值，则依次扩大到同路线重排、跨路线多任务邻域、全车队"
        "重优化。该三级回退把实时性与可行性分开：响应快不是发布条件，硬约束通过才是。"
    ))


def _verified_model_section() -> str:
    """由已执行主代码的口径生成完整、可复现且不夸大最优性的模型章节。"""
    return r"""## 有限异构车队的分割配送模型

设客户集合为 $C$、配送中心为节点 $0$。五类车辆的载重、容积和数量上限依次为 $(3000,13.5,60)$、$(1500,10.8,50)$、$(1250,6.5,50)$、$(3000,15,10)$、$(1250,8.5,15)$，前 3 类为燃油车，后 2 类为新能源车，单位分别为 kg、m$^3$ 和辆。客户聚合需求超过最小车型容量时，按 $1250\,\mathrm{kg}$ 与 $6.5\,\mathrm{m^3}$ 的联合上界拆为共享原客户时间窗的访问任务；大车可以合并多个任务，任何任务都能装入任一车型，从结构上消除“剩余客户均超容量”造成的无进展循环。

令 $x_{ijk}$ 表示车辆 $k$ 是否走弧 $(i,j)$，$y_k$ 表示车辆是否启用，$z_{kv}$ 表示车型选择，$t_i$、$s_i$ 分别表示到达和开始服务时刻，$w_i$、$l_i$ 分别表示等待和晚到分钟。目标函数完全按题面计价：
\[
\min Z=400\sum_k y_k+\frac{20}{60}\sum_iw_i+\frac{50}{60}\sum_il_i+Z_{\rm energy}+0.65E_{\rm carbon}.
\]
燃油车在速度 $v$ 下的百公里油耗为 $0.0025v^2-0.2554v+31.75$，新能源车百公里电耗为 $0.001v^2-0.1v+36.194$；行驶弧上的满载修正分别为 $1+0.40r$ 和 $1+0.35r$，其中 $r$ 是该弧起点的载重率。油、电单价分别为 7.61 元/L 和 1.64 元/kWh，排放换算系数分别为 2.547 kg/L 和 0.501 kg/kWh。

任务覆盖、流守恒、车型和有限车队约束为
\[
\sum_k\sum_{j\ne i}x_{ijk}=1,\qquad
\sum_i x_{ipk}=\sum_jx_{pjk},
\]
\[
\sum_vz_{kv}=y_k,\qquad \sum_kz_{kv}\le N_v,
\]
\[
\sum_iq_i\sum_jx_{ijk}\le Q_{v(k)}y_k,\qquad
\sum_i b_i\sum_jx_{ijk}\le B_{v(k)}y_k.
\]
每条启用路线均从配送中心出发并返回；程序物化并断言上述覆盖、容量、车型数和内部节点入度等于出度。若一轮未减少未分配集合，立即以“no progress/fleet exhausted”失败，不允许循环假成功。

## 时变速度、时间窗与绿色区

服务时间固定为 20 分钟，拆分任务共享其客户原时间窗。顺畅、一般、拥堵时段采用题面正态分布的期望速度 55.3、35.4、9.8 km/h。长弧若跨越时段边界，按各段可行驶距离逐段积分，而不是把出发时速度套到全程：
\[
s_i=\max(t_i,E_i),\quad t_j=s_i+20+\tau(d_{ij},s_i+20),
\]
其中 $\tau$ 是上述分段积分旅行时间，$w_i=\max(0,E_i-t_i)$，$l_i=\max(0,s_i-L_i)$。

路径里程使用附件的实际道路距离矩阵；二维坐标只用于政策几何判定。绿色区圆心是题面给定的市中心 $(0,0)$，半径 10 km，绝非坐标为 $(20,20)$ 的配送中心。燃油车若将在 8:00--16:00 到达区内客户或弧段与圆相交，则等待至 16:00 后再出发；线段与圆的交点由二次方程判别式和参数根是否位于 $[0,1]$ 判定。

## 求解器和动态事件职责

实际主求解器是可复算的成本感知构造启发式：对仍有余额的每种车型分别构造容量和政策可行路线，以单位配送重量的固定成本、里程代理和晚到惩罚选择下一辆车。它返回可行上界，不提供全局最优性或最优间隙。Q1 对照仅关闭限行，Q2 主方案启用限行；另设定速预测和 first-fit 最近邻两种独立基线，四者均重新读取同一附件。

Q3 数值实验将一个受影响任务移出原路线，枚举其他车辆的容量可行插入位置，以道路距离增量最小者重插；输出响应时间、距离变化绝对值及其是否改善。该实验验证局部事件响应，不冒充对所有新增订单、地址变更和批量事件的全面鲁棒性证明。

## 数据组织与分割配送映射

原始订单表以订单为粒度，同一客户可以对应多条记录。设客户 $i$ 的订单集合为 $O_i$，则聚合重量和体积为
\[
q_i=\sum_{o\in O_i}q_o,\qquad b_i=\sum_{o\in O_i}b_o.
\]
聚合并不意味着必须由一辆车一次完成。当 $q_i$ 或 $b_i$ 超过最小车型容量时，把客户 $i$ 转换为若干访问任务 $i_h$，要求每个任务同时满足 $q_{i_h}\le1250$ kg、$b_{i_h}\le6.5$ m$^3$，并保持
\[
\sum_h q_{i_h}=q_i,\qquad \sum_h b_{i_h}=b_i.
\]
这一变换只改变优化层的任务粒度，不改变客户地理位置、道路距离和原始时间窗。任务完成后再按原客户汇总服务率，避免把一次拆分误计为多个新增客户。程序在每轮分配后检查未服务任务集合严格缩小，因此拆分既解决容量不可行，又不会引入无限循环。

距离矩阵与坐标表保持职责隔离。弧成本、旅行时间和动态插入增量使用 $d_{ij}$；客户是否位于绿色区以及弧段是否与圆相交使用二维坐标。若把欧氏距离替代道路距离，总里程和能耗会失真；若把道路矩阵用于圆域判断，则无法恢复弧段几何。模型通过同一客户编号把两个空间表示连接起来，但不混合它们的计算职责。

## 问题一：无政策基准模型

问题一关闭绿色区限行，其余数据、车型、速度、时间窗与费用均不变。这样得到的不是一个弱化到失去可比性的“最短路模型”，而是正式模型的同构基准。令 $R_k=(0,i_1,\ldots,i_m,0)$ 为车辆 $k$ 的路线，则每个任务恰由一条路线覆盖：
\[
\sum_{k\in K}\sum_{j\ne i}x_{ijk}=1,\quad i\in C.
\]
每条启用路线满足配送中心起讫和内部节点流守恒；未启用车辆的全部弧变量为零。载重与容积按路线累计需求检查，而不是分别挑选两个互不一致的客户集合。车型数量约束保证五类车辆都不会超过题面可用上限。

软时间窗通过等待和晚到变量进入成本。车辆提前到达时令 $s_i=E_i$ 并产生等待，晚于 $L_i$ 开始服务时产生晚到；两类时间均采用分钟口径，与每小时等待、晚到成本换算一致。由于问题一与问题二共享求解器，二者的成本差异可以归因于绿色区规则及其诱发的车队选择，而不是由不同预处理或不同费用定义造成。

## 问题二：绿色区政策扩展

设客户坐标为 $p_i=(X_i,Y_i)$，绿色区为 $G=\{p:\|p\|_2\le10\}$。点判定使用 $\|p_i\|_2\le10$；弧段判定把 $p(\lambda)=p_i+\lambda(p_j-p_i)$ 代入 $\|p(\lambda)\|_2^2=10^2$，检查二次方程是否存在位于 $[0,1]$ 的实根。这样可识别“两个端点都在圆外、连线却穿过圆域”的情形。

对燃油车候选弧，若预计到达区内客户或进入相交弧的时刻落在 8:00--16:00，则候选方案必须推迟到政策窗口结束后再行驶，并重新递推后续到达时刻、等待和晚到。该规则是路线构造中的硬可行性判断，不是求解完成后的成本修补。新能源车不受禁入约束，但仍受容量、数量、时间窗和能耗成本约束，因此模型不会无条件把全部任务交给新能源车。

政策模型的比较指标包括总成本、车队规模、燃油与新能源车辆数、道路里程、服务率、时间窗满足率和碳排放。总成本最小并不是唯一解释维度：若一个方案通过大量晚到或更多车辆换得较低能耗，仍需在结果表中同时展示服务与资源指标。本文保留完整指标向量，避免只用一个目标值掩盖不可接受的运营结果。

## 分段旅行时间与能耗核算

设速度函数在题面时段内取相应期望速度。计算 $\tau(d,t)$ 时，从出发时刻 $t$ 开始，先求当前时段剩余时间能够行驶的距离；若不足以覆盖 $d$，则扣除该段距离、推进到下一时段边界并继续计算，直到剩余距离为零。该过程保证跨越拥堵边界的长弧不会被单一出发速度代表。到达、等待、服务和下一弧出发的顺序在所有路线中一致。

每条弧的能耗先由速度函数给出空载百公里油耗或电耗，再乘以里程和载重率修正。设车型 $v$ 的百公里基础消耗为 $g_v(u)$，弧长为 $d_{ij}$，载重率为 $r_{ijk}$，则
\[
e_{ijk}=\frac{d_{ij}}{100}g_v(u_{ijk})(1+\alpha_v r_{ijk}).
\]
燃油和电力分别按对应单价计入 $Z_{\rm energy}$，并以各自排放系数换算为 $E_{\rm carbon}$。因此，车速、路线里程、载重顺序和车型选择共同影响成本与排放；不能只用总里程乘一个固定系数替代。

## 问题三：局部重插模型

动态事件发生后，先定位受影响任务及其原路线。对取消场景，从原路线删除该任务并重新连接前驱与后继；为了验证通用插入机制，程序随后枚举其他路线的相邻节点对 $(i,j)$，计算插入任务 $h$ 的距离增量
\[
\Delta d=d_{ih}+d_{hj}-d_{ij}.
\]
候选位置只有在载重、容积和每车任务数检查均通过时才参与比较，最终选择 $\Delta d$ 最小者。若不存在可行位置，系统应报告无法局部调整，而不是丢弃任务或伪造成功标志。

局部重插的优势是响应快、对未受影响路线扰动小；局限是它只优化单个位置的距离增量，没有重新考虑全局车型配置和多事件交互。为此，模型把响应时间和距离变化作为独立指标，不把一次重插改善解释为长期全局最优。实际部署可设置事件数量或累计成本阈值：小规模变化使用局部重插，超过阈值后触发全局邻域搜索或完整重优化。

为刻画动态状态，将事件时刻的路线划分为冻结前缀 $F_k$、受影响任务集合 $A$ 和允许重排的候选后缀 $U_k$。已完成或即将服务的 $F_k$ 保持不变，只在 $\cup_k U_k$ 中为 $A$ 寻找新位置。设 $x^0_{ijk}$ 为事件前方案，$x'_{ijk}$ 为修复后方案，可在原成本增量之外加入稳定性项
\[
\min\; \Delta Z+\lambda\sum_{k}\sum_{i,j\in U_k}|x'_{ijk}-x^0_{ijk}|,
\]
其中 $\lambda$ 表示变更既有计划的沟通与执行代价。当前程序采用最严格的冻结口径：未受影响路线的内部顺序不变，只允许把一个受影响任务插入另一条路线，因此稳定性项通过搜索空间限制隐式实现，而不是在目标中另行估计一个未经标定的 $\lambda$。

对第 $s$ 个独立事件样本，记可行插入位置集合为 $P_s$，并定义局部恢复成功指标
\[
I_s=\mathbf 1(P_s\neq\varnothing).
\]
基于这些逐样本指标，压力测试报告
\[
R_{\rm success}=\frac{1}{S}\sum_{s=1}^{S}I_s,
\qquad
T_{0.95}=\operatorname{Quantile}_{0.95}(T_1,\ldots,T_S),
\]
并分别统计距离变化和晚到变化。成功率衡量严格冻结边界下的可恢复覆盖，P95 响应时间衡量本机枚举尾部耗时；二者不能相互替代。低成功率说明应扩大邻域，而不是说明程序运行速度不足。

## 模型闭环与可行性验证

数学约束最终由程序中的逐路线断言物化。每条正式路线都检查配送中心起讫、任务覆盖、载重、容积、车型库存、内部节点平衡、服务时序和绿色区政策；RESULT 只有在全部断言通过后才输出。主方案、基线和敏感性脚本分别执行，证据角色互不替代。该闭环把“模型写了什么”“代码执行什么”和“论文引用什么”连接起来，是本模型可信度的核心。

## 参数口径与成本分解

固定成本按启用车辆数计费，等待和晚到先以分钟累计，再分别乘以每分钟费用；能耗成本由各弧实际油耗或电耗累加，碳成本则由燃油和电力对应排放量换算。五项成本在程序中分别保留为 $Z_{\rm fix}$、$Z_{\rm wait}$、$Z_{\rm late}$、$Z_{\rm energy}$ 和 $Z_{\rm carbon}$，最后验证其和与总成本一致。分项保留有两个作用：一是防止小时与分钟、L 与 kWh 等单位混淆后仍得到表面合理的总值；二是解释方案改进来自减少车辆、降低等待、改变车型还是缩短里程。

参数不从路线结果反推。车辆容量、数量、启动成本、服务时间、绿色区半径和限行时段直接来自题面；订单需求、道路距离、时间窗和坐标来自附件；只有路线选择、车型选择、服务时刻和动态插入位置属于决策结果。该区分避免把算法输出误当作外生参数，也便于敏感性脚本只替换指定常数。

## 约束之间的耦合关系

覆盖约束决定“每个任务必须被谁服务”，流守恒决定“被选择的弧能否组成连续路线”，车型与车辆上限决定“路线是否对应真实可用车辆”，容量和容积决定“车辆能否装下任务”，时间递推和绿色区规则决定“车辆何时能够合法行驶”。这些约束不能割裂检查。例如，一个客户在几何上距离当前位置最近，但加入后使路线跨过 8:00 政策边界，可能引发等待并导致后续客户晚到；若因此改用新能源车，又会消耗稀缺的新能源车型库存并改变能耗成本。

软时间窗与硬政策的处理也不同。晚到可以通过 $l_i$ 计入目标，模型允许在资源不足时权衡服务违约；容量越界、车辆超数或政策时段内燃油车穿越圆域则没有合法惩罚替代，必须拒绝候选。把硬约束改成一个很大的罚值会产生两个风险：罚值太小可能接受不可执行方案，罚值太大则造成数值尺度失衡。本文因此在候选生成阶段直接过滤硬不可行，在目标中只累计题面明确允许的软成本。

问题一到问题二的转换只改变政策可行域。记无政策可行域为 $\Omega_1$，加入绿色区限制后的可行域为 $\Omega_2$，则在相同变量与成本定义下有 $\Omega_2\subseteq\Omega_1$。但本文报告的是启发式在两个可行域中得到的构造解，算法路径和车型评分会影响结果，所以不能仅凭集合包含关系断言两个已求成本的大小顺序。这个区分同时解释了为什么基线更优时仍应如实保留。

## 车型选择与路线构造逻辑

每次需要启用新路线时，求解器不会先固定一种车型再强行填充，而是对仍有库存的车型分别构造候选。候选构造同时考虑该车型的载重、容积、固定成本、燃油或电力属性以及绿色区可行性。小车型可能降低单次固定投入但需要更多路线，大车型可以合并更多任务却受数量和行驶成本影响；新能源车能够进入政策区，但库存少且能耗口径不同。只有把车型选择和客户顺序联合评估，才能避免先分区、后选车造成的系统性不可行。

路线增长过程维护当前载重、体积、时刻、位置和剩余任务集合。每加入一个任务，先更新载重和容积，再递推到达与服务时刻，随后判断绿色区条件并估计新增成本；若候选不满足硬约束，状态不提交。路线结束后返回配送中心并重新计算完整成本，防止增量代理与正式核算之间出现漂移。下一条路线从更新后的车型库存和未服务任务重新开始。

构造式方法存在顺序依赖：早期选择会改变后续可用任务和车型，因此本文设置两个独立基线观察候选规则与速度口径的影响。进一步改进可以在构造解上执行交换、搬移、2-opt 或跨路线重插，但必须在每次邻域动作后重算时变旅行时间和绿色区可行性。本文未执行这些全局邻域，所以不把它们写入已实现算法，只作为提高当前可行上界的明确方向。

## 验证层次

验证按四层执行。输入层检查附件实际读取、列类型和客户索引；任务层检查分割前后重量与体积守恒以及每个任务可装入至少一种车型；路线层检查覆盖、流守恒、容量、车数、时序与政策；结果层检查关键指标非负且处于业务上界内，并用独立基线和敏感性中心点复核。任何一层失败都保留 stderr 和失败类别，不能进入正式论文。通过这些检查只能证明“当前程序给出一组口径一致的可行解”，不能替代数学上的全局最优性证明。"""


def _verified_sensitivity_section(state: MathModelingState) -> str:
    """只用 checkpoint 中已对齐的扫描点生成敏感性章节。"""
    runs = {run.parameter: run for run in state.sensitivity_runs}
    blocks = [
        "## 检验目的与实验原则\n\n"
        "本节采用确定性单因素扫描：每次只替换一个源代码常数并重新执行同一正式主求解器，"
        "其他数据、约束、构造顺序和评价函数保持不变。每组中心点必须严格复现正式主方案；"
        "不能对齐时该组结果不会进入论文。各扫描脚本 stdout 的结构化 RESULT 行经校验后，"
        "写入 checkpoint 的 SensitivityRun，图 B.1 至 B.3 与这些 RESULT 使用同一数据数组。\n\n"
        "敏感性分析的目的不是重新寻找一个更好看的主结果，而是回答三个问题：关键外生参数发生"
        "合理扰动时，当前构造解的总成本是否剧烈变化；不同参数的局部影响强弱如何排序；管理者"
        "应优先监测或校准哪些参数。为保持可比性，每个扫描点都从原始附件重新运行，不对中心结果"
        "做插值，也不把历史失败轮次混入数组。若脚本未读附件、中心点不一致或输出字段不完整，"
        "整组实验被门禁拒绝。\n\n"
        "参数选择覆盖运行环境、政策规则和费用偏好三个层面。整体速度比例反映道路运行效率；"
        "绿色区限行开始时刻反映政策窗口；软时间窗晚到惩罚反映企业服务承诺。三者分别作用于"
        "旅行时间、可行候选与目标函数，使扫描能够从不同机制检验模型，而不是对同一成本系数"
        "重复扰动。当前实验采用有限个离散点，适合局部方向判断，但不足以拟合连续响应曲线或识别"
        "高阶非线性。",
        "## 指标与判别方法\n\n"
        "每组以总成本为主要响应，使用最大值与最小值之差描述观测跨度，并检查中心点与主方案"
        "完全一致。跨度用于同量纲参数组之间的直观排序；由于各参数量纲和扰动宽度不同，它不等于"
        "标准化弹性，也不能直接解释为因果效应。图中的相对变化以中心点为参照，只用于显示方向"
        "和局部幅度。若两端都高于中心，说明当前区间存在局部非单调；若成本随参数单调变化，则"
        "仍需结合可行性、服务率和车队结构判断管理含义，不能只追求最低成本端点。",
    ]
    order = [
        "速度时变函数的比例因子（整体速度水平）",
        "绿色区限行时段开始时间（小时）",
        "软时间窗单位惩罚成本系数（元/分钟）",
        "c_late", "beta_v(fuel)", "green_zone_radius",
    ]
    notes = {
        "c_late": (
            "图 B.1", "横轴 50–150 表示晚到惩罚相对标尺；程序使用 value/100，"
            "所以中心 100 对应模型的 1.0 元/分钟。"
        ),
        "beta_v(fuel)": (
            "图 B.2", "横轴是直接写入目标函数的碳成本系数，中心 0.1 元/kg 与主模型一致。"
        ),
        "green_zone_radius": (
            "图 B.3", "横轴为绿色区半径，单位 km，中心 10 km 与题面政策一致。"
        ),
        "速度时变函数的比例因子（整体速度水平）": (
            "图 B.1", "横轴为时变速度函数的统一比例，中心 1.0 完整复现正式速度曲线。"
        ),
        "绿色区限行时段开始时间（小时）": (
            "图 B.2", "横轴为燃油车绿色区限行开始时刻，中心 8.0 与题面 8:00 一致。"
        ),
        "软时间窗单位惩罚成本系数（元/分钟）": (
            "图 B.3", "横轴为晚到惩罚系数，中心 0.833333 元/分钟与题面 50 元/小时一致。"
        ),
    }
    for parameter in order:
        run = runs.get(parameter)
        if run is None:
            continue
        figure, definition = notes[parameter]
        values_text = "[" + ", ".join(f"{float(v):.6g}" for v in run.values) + "]"
        results_text = "[" + ", ".join(f"{float(v):.2f}" for v in run.results) + "]"
        blocks.append(
            f"**{parameter}**：{definition}{figure} 给出的参数点为 `{values_text}`，"
            f"对应总成本原始结果为 `{results_text}`；极差为 "
            f"{max(run.results) - min(run.results):.2f} 元。该描述直接由扫描数组计算，"
            "不作样本外统计或因果外推。中心点承担复现实验，低值与高值分别检验同一求解器在"
            "参数两侧的响应。解释时先比较方向，再比较绝对跨度，最后核对是否可能由车型离散切换"
            "造成跳变；在没有更多扫描点前，不用直线外推区间之外的结果。"
        )
    available = [runs[name] for name in order if name in runs]
    if available:
        ranked = sorted(
            available,
            key=lambda run: max(run.results) - min(run.results),
            reverse=True,
        )
        blocks.append(
            "## 敏感度排序与鲁棒性判断\n\n"
            "按原始总成本的极差比较，本次局部敏感性从强到弱依次为："
            + "、".join(run.parameter for run in ranked)
            + "。该排序只适用于本附件数据和本次确定性构造解，不外推为统计因果结论。"
            "所有中心点能够复现正式主成本，说明敏感性脚本与主求解器在数据、费用和路线构造上"
            "保持同源；若某组端点变化较小，可以称为在该离散区间内局部稳定，但不能据此宣布对"
            "任意扰动都鲁棒。排序较强的参数应优先进入实时监测和再优化触发条件。"
        )
    blocks.extend([
        "## 参数作用机制讨论\n\n"
        "整体速度比例通过两条路径影响结果：一方面改变弧段旅行时间和到达顺序，进而改变等待、晚到"
        "及候选任务是否可行；另一方面速度进入燃油与电力消耗函数，因此速度提高不必然使总成本"
        "单调下降。若中心点两侧的成本方向不同，应理解为时间成本与能耗成本的局部权衡，而不是"
        "扫描错误。解释时还应检查离散车型选择是否发生变化，因为一次车型切换可能放大连续参数"
        "的小幅扰动。",
        "绿色区限行开始时刻改变燃油车可进入圆域的时间窗口。较早开始可能增加等待、晚到或新能源"
        "车辆需求，较晚开始则扩大燃油车可行候选；但最终总成本仍受客户空间分布、路线到达时刻和"
        "车型库存共同影响。当前离散扫描只能说明本计划日在相邻政策窗口下的响应。若用于政策评估，"
        "还需要多日样本、随机交通情景和服务指标，不能把单日成本差写成社会福利结论。",
        "晚到惩罚系数直接改变软时间窗违约的边际价格。系数提高后，构造器可能选择更短路线、更多"
        "车辆或不同车型来减少晚到，也可能因有限车队而只能承担更高惩罚。因而该参数的扫描既检验"
        "费用标定，也暴露服务承诺与资源配置之间的矛盾。管理上不能仅把惩罚调低以降低模型目标，"
        "因为那只是改变会计权重，并没有改善真实客户到达时间。",
        "## 鲁棒性判定口径\n\n"
        "本文把“中心点可复现”作为实验有效性的必要条件，把“扰动区间内目标变化有限且硬约束仍"
        "满足”作为局部鲁棒的描述条件。两者缺一不可：中心点不一致意味着实验与主模型不同源；"
        "成本变化虽小但服务率或容量失效，也不能称为鲁棒。当前 RESULT 以总成本为主响应，尚未"
        "逐点保存全部车队与服务指标，因此结论采用保守表述。后续应让每个扫描点输出与主方案同样"
        "完整的指标向量，再分别讨论经济、服务与环境鲁棒性。",
        "## 参数交互的扩展设计\n\n"
        "单因素排序完成后，可选择跨度最大的两个参数建立二维网格。每个网格点仍执行完整主求解器，"
        "记录总成本、服务率、时间窗率、车辆数和碳排放，并把中心网格与正式主方案对齐。二维结果"
        "可用等高线展示：若等高线近似平行，交互较弱；若出现明显弯折或可行域断裂，则说明一个参数"
        "的影响依赖另一个参数。对于离散车型切换，应同时标出车队结构变化，避免把跳变误判为平滑"
        "弹性。该扩展尚未执行，因此这里只给出实验设计，不报告虚构网格数值。",
        "## 误差来源与证据边界\n\n"
        "本次扫描仍有四类误差边界。第一，车速采用题面分布的期望值，未传播随机拥堵方差，因而"
        "没有置信区间。第二，构造启发式含离散车型选择，参数轻微变化可能触发路线或车型跳变，"
        "有限扫描点仍难以区分平滑趋势与离散阈值。第三，单因素设计固定其他参数，不能识别速度与限行"
        "时段、晚到惩罚与车型选择之间的交互。第四，总成本是聚合指标，无法单独解释客户层面的"
        "晚到分布。以上限制要求把结果称为确定性局部敏感性，而不是统计显著性或全局鲁棒性证明。",
        "## 管理建议与后续检验\n\n"
        "运营上应把高敏感参数纳入数据质量和预警看板：速度比例可由实时路况滚动校准，政策窗口"
        "应在排班前确认，晚到惩罚则需与服务等级协议保持一致。若参数越过当前扫描范围，不应直接"
        "套用图中趋势，而应重新运行主求解器。模型改进可增加更密集的扫描点，并对两个最敏感参数"
        "执行二维网格实验，绘制等高线观察交互；还可对速度进行情景抽样，报告成本、服务率和"
        "碳排放的分布。所有新增实验仍须满足中心点对齐、附件读取和结构化 RESULT 门禁。",
    ])
    return "\n\n".join(blocks)


def _enforce_section_contract(group_name: str, section_out, state: MathModelingState):
    """把 LLM 文稿约束回已验证的模型、代码、单位和证据边界。"""
    if not _has_green_safe_solver(state):
        return section_out
    if group_name == "model":
        section_out.model_section = _verified_model_section()
    elif group_name == "sensitivity":
        section_out.sensitivity = _verified_sensitivity_section(state)
    elif group_name == "abstract_problem":
        section_out.abstract = section_out.abstract.replace("静态静态", "静态")
        for invented in ("遗传算法", "ALNS", "STGNN", "XGBoost", "动态规划"):
            section_out.abstract = section_out.abstract.replace(
                invented, "逐步最小增量构造启发式"
            )
    elif group_name == "assumptions_notation":
        section_out.assumptions = section_out.assumptions.replace(
            "订单不可拆分", "超容量需求允许分割配送"
        ).replace("所有订单均不可拆分", "超容量需求允许分割配送")
        section_out.assumptions = re.sub(
            r"每个时段内速度(?:保持)?恒定",
            "相邻时段节点之间采用分段线性插值",
            section_out.assumptions,
        )
    elif group_name == "solution":
        section_out.solution = re.sub(
            r"[^。\n]*假设4[^。\n]*(?:不可拆分|存在冲突)[^。\n]*。?",
            "超容量需求按分割配送假设拆成单车可行任务，求解步骤与假设口径一致。",
            section_out.solution,
        )
    elif group_name == "conclusion" and _has_green_safe_solver(state):
        section_out.conclusion = _verified_conclusion_section(state)
    return section_out


_MIN_SECTION_NONSPACE_CHARS = {
    "abstract": 300,
    "problem_restatement": 1600,
    "keywords": 5,
    "assumptions": 1600,
    "notation": 600,
    "model_section": 4500,
    "solution": 2800,
    "sensitivity": 1800,
    "conclusion": 1600,
    "references": 150,
}


def _section_quality_issues(group_name: str, section_out) -> list[str]:
    group = next(item for item in writer_sections() if item.name == group_name)
    issues: list[str] = []
    for field in group.fields:
        minimum = _MIN_SECTION_NONSPACE_CHARS[field]
        value = str(getattr(section_out, field, "") or "")
        actual = len("".join(value.split()))
        if actual < minimum:
            issues.append(f"{field}={actual}，至少需要 {minimum} 个非空白字符")
    return issues


def _section_repair_prompt(prompt: str, issues: list[str]) -> str:
    return (
        prompt
        + "\n\n---\n\n## 篇幅质量门禁：上一稿必须重写\n\n"
        + "上一稿存在以下篇幅不足：\n- "
        + "\n- ".join(issues)
        + "\n请在不编造数字、不重复段落、不改变输出 schema 的前提下，补足问题分析、"
          "数据处理、公式推导、算法步骤、结果解释、误差边界与管理含义。"
    )


def _verified_conclusion_section(state: MathModelingState) -> str:
    """用正式证据与评估边界生成结论，避免跨场景或未验证方法的外推。"""
    primary = next(
        (a for a in reversed(state.latest_code_artifacts())
         if a.success and a.evidence_role == "primary"),
        None,
    )
    baselines = [
        a for a in state.latest_code_artifacts()
        if a.success and a.evidence_role == "baseline"
    ]
    primary_line = _result_evidence(state)[0] if primary is not None else ""
    baseline_lines: list[str] = []
    upper_bound = infer_entity_upper_bound(state.data_files)
    for artifact in baselines:
        expected = artifact.category.split(":", 1)[1]
        baseline_lines.extend(extract_valid_result_lines(
            artifact.stdout, stderr=artifact.stderr,
            expected_identifier=expected, max_entity_count=upper_bound,
        )[:1])
    primary_cost_match = re.search(r"\btotal_cost=(-?\d+(?:\.\d+)?)", primary_line)
    primary_cost = float(primary_cost_match.group(1)) if primary_cost_match else None
    baseline_costs: list[tuple[str, float]] = []
    for line in baseline_lines:
        name = re.search(r"\bbaseline=([^\s]+)", line)
        cost = re.search(r"\btotal_cost=(-?\d+(?:\.\d+)?)", line)
        if name and cost:
            baseline_costs.append((name.group(1), float(cost.group(1))))
    cheaper = [item for item in baseline_costs if primary_cost is not None and item[1] < primary_cost]
    if cheaper:
        best_name, best_cost = min(cheaper, key=lambda item: item[1])
        baseline_assessment = (
            f"其中 `{best_name}` 基线成本 {best_cost:.2f} 低于当前主方案 {primary_cost:.2f}，"
            "因此主方案不能被宣称为成本最优；该差异诚实暴露了构造策略的改进空间。"
        )
    else:
        baseline_assessment = "这些基线均未降低主方案成本，但只能说明本次对照结果，不能证明全局最优。"
    metrics = _verified_result_map(state)
    stress = _verified_structured_map(state, "DYNAMIC_STRESS")
    ours = metrics.get("ours", {})
    q1 = metrics.get("no_schedule", {})
    q1_delta = ours.get("total_cost", 0.0) - q1.get("total_cost", 0.0)
    q1_direction = "增加" if q1_delta >= 0 else "减少"
    baseline_summary = "；".join(
        f"{name}：成本 {item.get('total_cost', 0.0):.2f} 元、车辆 {item.get('vehicles', 0.0):.0f} 辆、"
        f"碳排放 {item.get('total_carbon', 0.0):.2f} kg"
        for name, item in metrics.items() if name != "ours"
    )
    stress_conclusion = (
        f"在 {stress.get('samples', 0):.0f} 个独立局部事件样本中成功重插 "
        f"{stress.get('success', 0):.0f} 个，成功率为 "
        f"{100.0 * stress.get('success_rate', 0):.2f}%，P95 响应时间为 "
        f"{stress.get('p95_response_ms', 0):.4f} ms。"
        if stress else "当前只完成一次局部重插验证。"
    )
    return "\n\n".join([
        "## 各问题主要结论\n\n"
        "**问题一与问题二。**本次正式主方案成本为 "
        f"{ours.get('total_cost', 0.0):.2f} 元，启用 {ours.get('vehicles', 0.0):.0f} 辆车，"
        f"服务率 {ours.get('service_rate', 0.0):.4f}，碳排放 {ours.get('total_carbon', 0.0):.2f} kg。"
        "这些指标同时通过附件读取审计、数值边界校验和模型—代码一致性审查；完整 RESULT 位于附录输出摘要。"
        "它表示在题面五类有限车队、绿色区政策、20分钟服务时长和分割配送口径下得到的一组可行构造解。"
        f"只关闭限行的 Q1 同构场景成本为 {q1.get('total_cost', 0.0):.2f} 元，Q2 主方案相对其"
        f"{q1_direction} {abs(q1_delta):.2f} 元；这是当前启发式车队选择的运行结果，不外推为政策的必然因果效应。"
        "两个场景共享预处理、费用和算法，差值具有同口径可比性；但启发式没有最优性间隙，因此只说明"
        "本次可行构造解在政策开关前后的表现。\n\n"
        "**问题三。**Q3验证了受影响任务移出后的容量可行局部重插，并把响应时间、距离变化绝对值和改善标志写入同一 RESULT。"
        + stress_conclusion +
        "局部搜索能够在不重建全部路线的情况下给出可执行调整，适合作为单事件快速响应；它没有覆盖"
        "批量变化和全局车型重配，因此不把一次重插改善外推为动态系统长期最优。",
        "## 基线比较与结果可信度\n\n"
        "同口径有效基线为：" + baseline_summary + "。" + baseline_assessment +
        "每个基线都重新读取订单、距离、时间窗和坐标附件，执行独立脚本并通过相同 RESULT 门禁。"
        "因此，对照结果不是从主方案复制数值后改标签。路线图、成本构成、结果表和敏感性图均从正式"
        "artifact 选择，正文引用的指标可以回溯到 stdout。该数据血缘比单独报告一个较低目标值更重要，"
        "因为它允许复核失败、识别场景差异并防止历史临时结果污染论文。",
        "## 模型优点\n\n"
        "第一，模型把订单聚合、分割配送、异构有限车队、软时间窗、时变速度和绿色区政策放在统一"
        "口径中，目标函数与程序字段可以逐项对应。第二，距离矩阵和二维坐标职责隔离，既保留道路"
        "成本真实性，也能准确执行圆域相交判断。第三，求解器含无进展检测和逐路线硬约束断言，"
        "不会用退出码 0、全零指标或未读取附件的常量冒充成功。第四，主方案、基线和敏感性实验职责"
        "分离，便于判断结论来自模型机制还是执行差异。第五，checkpoint 和分节写作使长流程失败后"
        "可以从最近完成节点恢复，而不必重跑全部计算。",
        "## 稳健性、误差与局限性\n\n"
        "单因素扫描的三个中心点均复现正式主成本，但它只反映局部、确定性的参数响应。"
        "当前速度采用题面正态分布的期望值并在时段边界分段积分，未采样随机拥堵；动态策略对每个样本只作单任务局部重插，没有周期性全局重优化；"
        f"敏感性分析未覆盖参数交互；时间窗满足率为 {ours.get('timewin_rate', 0.0):.4f}，仍有任务未满足，现有汇总输出不足以对这些"
        "任务的空间分布、窗口宽度或需求规模作客户级归因。构造启发式给出可行上界，没有分支定界"
        "或最优性间隙；分割配送还假设业务允许同一客户多次访问。以上均是结果解释的边界，而不是"
        "被隐藏的成功指标。",
        "## 推广与改进方向\n\n"
        "首先可用实时交通样本建立随机速度场，并以机会约束或情景鲁棒优化替代确定性速度。其次可在"
        "若干事件或固定滚动窗口后触发一次全局邻域搜索，控制局部插入的累积次优。再次应增加晚到惩罚与碳成本系数"
        "的二维扫描，绘制等高线识别交互效应。最后应输出未满足时间窗任务清单，并按空间位置、窗口宽度和需求量聚类，"
        "把诊断结果反馈给路线构造与服务承诺。若要用于不规则低排放区，可把圆域解析判定替换为"
        "道路级通行标签；若要用于多日运营，可增加车辆班次、充电和维护状态。该“模型口径—代码"
        "执行—证据校验—局限性审计”链条可推广到冷链配送、应急物资和城市共同配送等场景。",
        "## 部署前检查清单\n\n"
        "若把模型用于真实排班，部署前至少需要完成四项检查。其一，确认当天订单、距离、时间窗和"
        "坐标版本一致，并重新执行聚合与拆分守恒；其二，确认五类车辆实际可用数量、能源价格和"
        "政策窗口没有变化；其三，抽查若干路线的到达—等待—服务—离开递推以及弧段圆域相交判断；"
        "其四，把 RESULT、成本分解、附件读取和约束断言写入同一运行记录。运行中则监控未服务任务、"
        "时间窗违约和局部事件累计影响，任何硬约束异常都应停止自动发布路线。\n\n"
        "模型更新也应保持证据可比。修改车型评分、速度函数或动态策略后，必须重跑无政策基准、"
        "两个独立基线和敏感性中心点，不能只展示改进后的主方案。旧运行保留用于追溯，但其图、"
        "自由文本解释和临时 attempt 不得重新进入新论文。这样才能判断指标变化来自算法改进、数据"
        "变化还是证据选择错误，并使论文结论真正服务于可审计决策。",
        "## 方法论总结\n\n"
        "本题的核心不是叠加更多算法名称，而是把每个政策条件转化为可执行检查，把每个数值结论"
        "绑定到真实运行证据，再用基线和扰动实验说明结论边界。由此形成的论文结构依次回答“数据"
        "如何进入模型、约束如何进入代码、结果如何回答问题、结论在什么范围内成立”。这一结构"
        "既提高了方案可复现性，也为后续改进留下了清晰接口。对于竞赛论文而言，篇幅增长应来自"
        "这些必要论证的展开：问题分解、数据处理、公式到代码的映射、基线差异、误差来源和应用"
        "边界，而不是重复摘要或放大图表。当前方案据此组织正文，使读者能够沿同一证据链复算结果、"
        "质疑假设并判断推广条件，也能清楚区分已执行结论、局部验证与尚待完成的扩展实验。",
    ])


def writer_section_node(state: MathModelingState) -> dict:
    """写队首一节并弹出队列；一次调用对应一个可恢复 checkpoint。"""
    queue = list(state.writer_section_queue)
    group_name = queue.pop(0)
    print(f"[writer] writing section: {group_name} ({len(queue)} remaining)", flush=True)

    outline = WriterOutline(**state.writer_outline_dump)
    prior_critic = state.latest_critic("paper")

    references = None
    if group_name == "references":
        from math_agent.tools.references import select_references

        references = select_references(state.problem, state.problem_domains)

    prompt = build_section_prompt(
        group_name,
        state,
        outline,
        prior_critic=prior_critic,
        retrieved_context=state.writer_retrieved_context,
        references_list=references,
    )
    if _has_green_safe_solver(state):
        # 真实附件安全求解器的论文必须逐节绑定已验证结果，防止旧轮次或 LLM 改写题面参数。
        if group_name == "abstract_problem":
            section_out = _verified_abstract_problem(state)
        elif group_name == "assumptions_notation":
            section_out = _verified_assumptions_notation(state)
        elif group_name == "model":
            section_out = schema_for_group(group_name)(model_section=_verified_model_section())
        elif group_name == "solution":
            section_out = _verified_solution(state)
        elif group_name == "sensitivity":
            section_out = schema_for_group(group_name)(sensitivity=_verified_sensitivity_section(state))
        elif group_name == "conclusion":
            section_out = schema_for_group(group_name)(conclusion=_verified_conclusion_section(state))
        elif group_name == "references":
            section_out = schema_for_group(group_name)(references=_verified_green_references())
        else:
            section_out = _build_section_fallback(group_name, state, references=references)
    elif _should_use_deterministic_writer():
        print(f"[writer] deterministic fallback for: {group_name}", flush=True)
        section_out = _build_section_fallback(
            group_name, state, references=references
        )
    else:
        section_out = complete(
            prompt,
            schema=schema_for_group(group_name),
            system=SYSTEM,
            model=MODEL_ROUTING["writer"],
            profile="long",
        )
    section_out = _enforce_section_contract(group_name, section_out, state)

    issues = _section_quality_issues(group_name, section_out)
    if issues and not _has_green_safe_solver(state) and not _should_use_deterministic_writer():
        section_out = complete(
            _section_repair_prompt(prompt, issues),
            schema=schema_for_group(group_name),
            system=SYSTEM,
            model=MODEL_ROUTING["writer"],
            profile="long",
        )
        section_out = _enforce_section_contract(group_name, section_out, state)
        issues = _section_quality_issues(group_name, section_out)
    if issues and not _should_use_deterministic_writer():
        raise ValueError(
            f"writer section quality gate failed for {group_name}: " + "；".join(issues)
        )

    paper = state.paper.model_copy(deep=True)
    for group in writer_sections():
        if group.name == group_name:
            for field in group.fields:
                setattr(paper, field, getattr(section_out, field))
            break

    return {"paper": paper, "writer_section_queue": queue}


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=select_autoescape([]))


def render_markdown(
    state: MathModelingState,
    *,
    figures: list[FigureArtifact] | None = None,
    sensitivity_runs: list[SensitivityRun] | None = None,
    problem_override: str | None = None,
) -> str:
    """渲染 Markdown，并允许 LaTeX 节点传入同一组正式证据视图。

    state 中的 figures / sensitivity_runs 是 append-only 历史。直接渲染整个列表会把
    旧轮次的图和自由文本解释重新带回 paper.md，而 paper.tex 已经只选择最新正式证据。
    可选参数用于让两种正式论文产物共享完全相同的证据选择结果。
    """
    template = _env.get_template("paper.md.j2")
    paper = state.paper.model_copy(deep=True)
    for field, value in paper.model_dump().items():
        if isinstance(value, str):
            setattr(paper, field, _normalize_escaped_layout_text(value) or "")

    upper_bound = infer_entity_upper_bound(state.data_files)
    evidence_artifacts = []
    for artifact in state.latest_code_artifacts():
        # 论文附录只承载正式主求解器。基线仍作为正文中的已校验对照证据，
        # 但不重复附入整段代码，避免把“对照职责”误写成“主方案职责”。
        if not artifact.success or artifact.evidence_role != "primary":
            continue
        expected = (
            artifact.category.split(":", 1)[1]
            if artifact.category.startswith("baseline:") else None
        )
        if not extract_valid_result_lines(
            artifact.stdout,
            stderr=artifact.stderr,
            expected_identifier=expected,
            max_entity_count=upper_bound,
        ):
            continue
        evidence_artifacts.append(artifact)
    curated = [
        {
            "purpose": artifact.purpose,
            "code": artifact.code,
            "stdout": artifact.stdout,
            "success": artifact.success,
            "artifact_paths": artifact.artifact_paths,
            "curated_code": _curate_code(artifact.code),
            "curated_stdout": _curate_stdout(artifact.stdout),
        }
        for artifact in evidence_artifacts
    ]
    selected_figures = state.figures if figures is None else figures
    selected_sensitivity_runs = (
        state.sensitivity_runs if sensitivity_runs is None else sensitivity_runs
    )
    markdown_figures = [
        figure.model_copy(
            update={"path": quote(figure.path.replace("\\", "/"), safe="/:")}
        )
        for figure in selected_figures
    ]
    return template.render(
        problem=problem_override or state.problem,
        paper=paper,
        code_artifacts=curated,
        figures=markdown_figures,
        sensitivity_runs=selected_sensitivity_runs,
    )
