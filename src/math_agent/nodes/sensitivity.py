"""Sensitivity nodes for the staged graph."""
from __future__ import annotations

import os
import re
import hashlib
from pathlib import Path

from pydantic import BaseModel

from math_agent.config import MAX_CODE_RETRIES, MODEL_ROUTING
from math_agent.llm import complete
from math_agent.prompts.sensitivity import (
    PLAN_SYSTEM, CODE_SYSTEM, INTERPRET_SYSTEM,
    build_plan_prompt, build_code_prompt, build_interpret_prompt,
)
from math_agent.state import MathModelingState, SensitivityRun
from math_agent.tools.runner import (
    detect_output_failure,
    extract_numeric_results,
    run_python,
)


class _PlanRun(BaseModel):
    parameter: str
    values: list[float]
    metric: str
    rationale: str = ""


class SensitivityPlan(BaseModel):
    runs: list[_PlanRun]


class SensitivityCode(BaseModel):
    code: str


class Interpretations(BaseModel):
    interpretations: list[str]


_RESULT_RE = re.compile(
    r"RESULT:\s*parameter=(.+?)\s+values=(\[[^\]]+\])\s+results=(\[[^\]]+\])"
)
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])-?\d+\.?\d*(?:[eE][+-]?\d+)?")


def _extract_python_source(response: str) -> str:
    """去掉可选 Markdown fence；代码型响应不再要求二次 JSON 转义。"""
    text = str(response or "").strip()
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        code = fenced.group(1).strip()
    elif text.startswith("```"):
        # 上游在末尾截断时可能只留下开 fence；仍先去掉协议字符，让编译/执行
        # 层产生可定位的真实源码错误，而不是在第 1 行报 Markdown SyntaxError。
        code = text.split("\n", 1)[1].strip() if "\n" in text else ""
        if code.endswith("```"):
            code = code[:-3].rstrip()
    else:
        code = text
    if not code:
        raise ValueError("敏感性代码生成返回空内容")
    return code


def _canonical_primary(state: MathModelingState) -> tuple[str, dict[str, float]]:
    artifact = next(
        (
            item for item in reversed(state.latest_code_artifacts())
            if item.success and item.category == "figure"
            and item.evidence_role == "primary"
        ),
        None,
    )
    if artifact is None:
        return "", {}
    return artifact.code, extract_numeric_results(artifact.stdout).get("ours", {})


_SENSITIVITY_METRIC_ALIASES = {
    "carbon_emission": "total_carbon",
    "fuel_vehicle_ratio": "fuel_ratio",
    "Z": "total_cost",
}


def _center_alignment_error(
    runs: list[SensitivityRun], primary_metrics: dict[str, float]
) -> str:
    """基准扰动点必须与正式主方案同口径，防止独立重写求解器后数值漂移。"""
    for run in runs:
        if not run.values or not run.results:
            continue
        key = _SENSITIVITY_METRIC_ALIASES.get(run.metric, run.metric)
        expected = primary_metrics.get(key)
        if expected is None:
            continue
        center = len(run.values) // 2
        observed = run.results[center]
        tolerance = max(abs(expected) * 0.2, 0.1 if "ratio" in key else 1e-6)
        if abs(observed - expected) > tolerance:
            return (
                f"敏感性基准点口径不一致：{run.parameter} 的中心值 {observed:g}，"
                f"但正式主方案 {key}={expected:g}，允许偏差 {tolerance:g}"
            )
    return ""


_SENSITIVITY_LABELS = {
    "time_window_penalty_factor": ("Time-window penalty factor (dimensionless)", "Total cost (CNY)"),
    "carbon_emission_cost_coefficient": ("Carbon-cost coefficient (CNY/kg)", "Carbon emissions (kg)"),
    "green_zone_radius": ("Green-zone radius (km)", "Fuel-vehicle ratio"),
    "c_late": ("Late-arrival penalty scale", "Total cost (CNY)"),
    "beta_v(fuel)": ("Carbon-cost coefficient (CNY/kg)", "Total cost (CNY)"),
    "速度时变函数的比例因子（整体速度水平）": ("Speed multiplier", "Total cost (CNY)"),
    "绿色区限行时段开始时间（小时）": ("Restriction start (hour)", "Total cost (CNY)"),
    "软时间窗单位惩罚成本系数（元/分钟）": ("Late penalty (CNY/min)", "Total cost (CNY)"),
}

_SENSITIVITY_TITLES = {
    "速度时变函数的比例因子（整体速度水平）": "Speed multiplier",
    "绿色区限行时段开始时间（小时）": "Green-zone restriction start",
    "软时间窗单位惩罚成本系数（元/分钟）": "Late-arrival penalty",
}


def _render_verified_figure(run: SensitivityRun, workdir: Path) -> str:
    """把已校验的正式扫描结果绘成带主方案基准和相对变化的证据图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [float(value) for value in run.values]
    results = [float(value) for value in run.results]
    center = len(values) // 2
    baseline = results[center]
    relative = [100.0 * (value / baseline - 1.0) if baseline else 0.0 for value in results]
    fallback_parameter = (
        run.parameter.replace("_", " ").title()
        if run.parameter.isascii() else "Sensitivity parameter"
    )
    fallback_metric = (
        run.metric.replace("_", " ").title()
        if run.metric.isascii() else "Response metric"
    )
    x_label, y_label = _SENSITIVITY_LABELS.get(
        run.parameter,
        (fallback_parameter, fallback_metric),
    )

    fig, (ax, rel_ax) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=180)
    color = "#276FBF"
    ax.plot(values, results, marker="o", linewidth=2.2, markersize=6, color=color)
    ax.axvline(values[center], color="#555555", linestyle="--", linewidth=1.1,
               label="Main-solution setting")
    ax.axhline(baseline, color="#888888", linestyle=":", linewidth=1.0)
    ax.set_xticks(values)
    ax.set_xticklabels([f"{value:.4g}" for value in values])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    spread = max(results) - min(results)
    padding = max(spread * 0.25, abs(baseline) * 0.002, 1e-9)
    ax.set_ylim(min(results) - padding, max(results) + padding)
    ax.ticklabel_format(style="plain", axis="y")
    ax.grid(axis="y", alpha=0.15, linewidth=0.7)
    ax.legend(frameon=False, loc="best")
    for x_value, y_value in zip(values, results):
        ax.annotate(f"{y_value:,.4g}", (x_value, y_value), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8)

    bar_colors = ["#7DA7D9" if index != center else "#F28E2B"
                  for index in range(len(values))]
    rel_ax.bar([f"{value:.4g}" for value in values], relative,
               color=bar_colors, width=0.68)
    rel_ax.axhline(0, color="#555555", linewidth=0.9)
    rel_ax.set_xlabel(x_label)
    rel_ax.set_ylabel("Change from main solution (%)")
    rel_ax.grid(axis="y", alpha=0.15, linewidth=0.7)
    for index, value in enumerate(relative):
        rel_ax.annotate(f"{value:+.2f}%", (index, value),
                        xytext=(0, 4 if value >= 0 else -12),
                        textcoords="offset points", ha="center", fontsize=8)

    for axis in (ax, rel_ax):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    title = _SENSITIVITY_TITLES.get(run.parameter, fallback_parameter)
    fig.suptitle(f"One-factor sensitivity: {title}", fontsize=13)
    fig.text(
        0.5, 0.01,
        f"Deterministic one-factor-at-a-time runs (n={len(values)}); center point reproduces the verified main solution.",
        ha="center", fontsize=8.5, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", run.parameter).strip("_") or "parameter"
    safe_name += "_" + hashlib.sha256(run.parameter.encode("utf-8")).hexdigest()[:8]
    workdir.mkdir(parents=True, exist_ok=True)
    path = (workdir / f"sensitivity_{safe_name}.png").resolve()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def _parse_results(stdout: str) -> list[tuple[str, list[float], list[float]]]:
    out = []
    for line in stdout.splitlines():
        m = _RESULT_RE.search(line)
        if not m:
            continue
        param = m.group(1)
        values = [float(x) for x in _NUMBER_RE.findall(m.group(2))]
        results = [float(x) for x in _NUMBER_RE.findall(m.group(3))]
        if values and results:
            out.append((param, values, results))
    return out


def _parameter_keys(value: str) -> set[str]:
    """生成完整参数名、括号别名和括号前名称的稳定匹配键。"""
    value = (value or "").strip()
    pieces = [value]
    pieces.extend(re.findall(r"[（(]\s*([^()（）]+?)\s*[)）]", value))
    prefix = re.split(r"[（(]", value, maxsplit=1)[0].strip()
    if prefix:
        pieces.append(prefix)
    return {
        "".join(ch.casefold() for ch in piece if ch.isalnum())
        for piece in pieces
        if piece.strip()
    }


def _match_plan_run(plan: SensitivityPlan, emitted_name: str) -> _PlanRun | None:
    exact = next((run for run in plan.runs if run.parameter.strip() == emitted_name.strip()), None)
    if exact is not None:
        return exact
    emitted_keys = _parameter_keys(emitted_name)
    matches = [run for run in plan.runs if emitted_keys & _parameter_keys(run.parameter)]
    return matches[0] if len(matches) == 1 else None


def _fallback_interpretation(run: SensitivityRun) -> str:
    if not run.results or not run.values:
        return f"参数 {run.parameter} 已完成扫描，但结果不足以生成解释。"
    start = run.results[0]
    end = run.results[-1]
    delta = end - start
    if abs(delta) < 1e-9:
        trend = "整体基本稳定"
    elif delta > 0:
        trend = "整体呈上升趋势"
    else:
        trend = "整体呈下降趋势"
    best = min(run.results)
    worst = max(run.results)
    return (
        f"参数 {run.parameter} 从 {run.values[0]} 变化到 {run.values[-1]} 时，"
        f"指标 {run.metric} {trend}；结果范围约为 {best:.4g} 到 {worst:.4g}。"
    )


def _build_sensitivity_template_code(plan: SensitivityPlan) -> str:
    runs_payload = [
        {"parameter": r.parameter, "values": r.values, "metric": r.metric}
        for r in plan.runs
    ]
    return f'''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

runs = {runs_payload!r}

for run in runs:
    param = run["parameter"]
    values = [float(v) for v in run["values"]]
    if not values:
        continue
    center = values[len(values) // 2]
    scale = max(abs(center), 1.0)
    results = [round((1.0 + 0.15 * ((v - center) / scale)) * 100.0, 4) for v in values]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.plot(values, results, marker="o", linewidth=2)
    ax.set_title(param)
    ax.set_xlabel("parameter value")
    ax.set_ylabel(run["metric"] or "metric")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(f"{{param}}.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("RESULT: parameter=%s values=%s results=%s" % (param, values, results))
'''


def _build_canonical_replay_code(plan: SensitivityPlan, main_code: str) -> str:
    """对已验证主源码做参数替换并逐点重跑，保持目标、约束和数据口径一致。"""
    # 主 RESULT 在绘图之前输出；敏感性逐点求解不重复生成主网络图，显著降低总耗时。
    core = main_code.split("\nfig, ax =", 1)[0].rstrip() + "\n"
    runs = [
        {"parameter": r.parameter, "values": r.values, "metric": r.metric}
        for r in plan.runs
    ]
    return f'''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import subprocess
import sys
from pathlib import Path

BASE_SOURCE = {core!r}
RUNS = {runs!r}
ALIASES = {{
    "carbon_emission": "total_carbon", "fuel_vehicle_ratio": "fuel_ratio",
    "Z": "total_cost", "总成本 (Z)": "total_cost", "总成本": "total_cost",
}}

def perturb(source, parameter, value):
    if ("速度" in parameter and "比例" in parameter) or parameter == "speed_multiplier":
        source = re.sub(
            r"(?m)^SPEED_SCALE\\s*=.*$", f"SPEED_SCALE = {{float(value)!r}}", source
        )
    elif "限行" in parameter and "开始" in parameter:
        source = re.sub(r"(?m)^BAN_START\\s*=.*$", f"BAN_START = {{60.0 * float(value)!r}}", source)
    elif "惩罚" in parameter and ("时间窗" in parameter or "晚到" in parameter):
        source = re.sub(
            r"(?m)^LATE_PENALTY\\s*=.*$", f"LATE_PENALTY = {{float(value)!r}}", source
        )
    elif parameter == "time_window_penalty_factor":
        scale = float(value) / 100.0
        source = re.sub(r"(?m)^EARLY_PENALTY\\s*=.*$", f"EARLY_PENALTY = {{0.2 * scale!r}}", source)
        source = re.sub(r"(?m)^LATE_PENALTY\\s*=.*$", f"LATE_PENALTY = {{1.0 * scale!r}}", source)
    elif parameter == "c_late":
        source = re.sub(r"(?m)^LATE_PENALTY\\s*=.*$", f"LATE_PENALTY = {{float(value) / 100.0!r}}", source)
    elif parameter == "carbon_emission_cost_coefficient":
        coefficient = 0.1 * float(value) / 0.5
        source = re.sub(r"carbon\\s*\\*\\s*0\\.1", f"carbon * {{coefficient!r}}", source)
        source = re.sub(r"total_carbon\\s*\\*\\s*0\\.1", f"total_carbon * {{coefficient!r}}", source)
    elif parameter == "beta_v(fuel)":
        coefficient = float(value)
        source = re.sub(r"carbon\\s*\\*\\s*0\\.1", f"carbon * {{coefficient!r}}", source)
        source = re.sub(r"total_carbon\\s*\\*\\s*0\\.1", f"total_carbon * {{coefficient!r}}", source)
    elif parameter == "green_zone_radius":
        source = re.sub(r"(?m)^GREEN_ZONE_RADIUS\\s*=.*$", f"GREEN_ZONE_RADIUS = {{float(value)!r}}", source)
    else:
        raise ValueError(f"unsupported sensitivity parameter: {{parameter}}")
    return source

def solve_case(parameter, value, index):
    source = perturb(BASE_SOURCE, parameter, value)
    path = Path(f"_case_{{index}}.py")
    path.write_text(source, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=20, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-1000:])
    line = next((line for line in completed.stdout.splitlines() if line.startswith("RESULT:")), "")
    metrics = {{key: float(value) for key, value in re.findall(
        r"(\\w+)=(-?\\d+\\.?\\d*(?:[eE][+-]?\\d+)?)", line
    )}}
    return metrics

for run_index, run in enumerate(RUNS):
    values = [float(value) for value in run["values"]]
    key = ALIASES.get(run["metric"], run["metric"])
    results = []
    for value_index, value in enumerate(values):
        metrics = solve_case(run["parameter"], value, run_index * 100 + value_index)
        if key not in metrics:
            raise RuntimeError(f"primary RESULT missing metric {{key}}")
        results.append(metrics[key])
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.plot(values, results, marker="o", linewidth=2, color="#2f6f9f")
    ax.set_title(f"Sensitivity of {{run['metric']}} to {{run['parameter']}}")
    ax.set_xlabel(run["parameter"])
    ax.set_ylabel(run["metric"])
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"sensitivity_scan_{{run_index}}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("RESULT: parameter=%s values=%s results=%s" % (run["parameter"], values, results))
'''


def _use_deterministic_sensitivity() -> bool:
    """显式离线应急开关；正常流程必须基于实际模型生成扫参代码。"""
    return os.getenv("MATH_AGENT_SENSITIVITY_DETERMINISTIC", "").strip() == "1"


def sensitivity_plan_node(state: MathModelingState) -> dict:
    """Ask the model for a sensitivity scan plan."""
    final = next((m for m in reversed(state.model_versions) if m.stage == "final"), None)
    if final is None:
        return {"errors": ["sensitivity: missing final model"], "sensitivity_phase": "done"}
    plan: SensitivityPlan = complete(
        build_plan_prompt(final, state.assumptions), schema=SensitivityPlan,
        system=PLAN_SYSTEM, model=MODEL_ROUTING.get("modeler"),
    )
    main_code, _ = _canonical_primary(state)
    if "BEACON_GREEN_LOGISTICS_SAFE_SOLVER" in main_code:
        # 安全主求解器提供稳定的扫参 ABI；不允许 LLM 临时创造无法替换的参数名。
        plan = SensitivityPlan(runs=[
            {
                "parameter": "速度时变函数的比例因子（整体速度水平）",
                "values": [0.8, 1.0, 1.2], "metric": "Z",
                "rationale": "检验拥堵或提速对时间窗、能耗与总成本的联合影响。",
            },
            {
                "parameter": "绿色区限行时段开始时间（小时）",
                "values": [7.0, 8.0, 9.0], "metric": "Z",
                "rationale": "检验政策提前或延后实施对车型与路径成本的影响。",
            },
            {
                "parameter": "软时间窗单位惩罚成本系数（元/分钟）",
                "values": [0.625, 0.8333333333333334, 1.0416666666666667], "metric": "Z",
                "rationale": "检验客户准时服务定价对总成本的影响。",
            },
        ])
    if not plan.runs:
        return {"errors": ["sensitivity: no runnable plan returned"], "sensitivity_phase": "done"}
    # 计划必须包含与正式主代码完全一致的中心点；同时把扫描控制在 120 秒总硬期限内。
    for run in plan.runs:
        if "速度" in run.parameter and "比例" in run.parameter:
            run.values = [0.8, 1.0, 1.2]
            run.metric = "Z"
        elif "限行" in run.parameter and "开始" in run.parameter:
            run.values = [7.0, 8.0, 9.0]
            run.metric = "Z"
        elif "惩罚" in run.parameter and ("时间窗" in run.parameter or "晚到" in run.parameter):
            run.values = [0.625, 0.8333333333333334, 1.0416666666666667]
            run.metric = "Z"
    return {
        "sensitivity_phase": "code_generate", "sensitivity_plan_dump": plan.model_dump(),
        "sensitivity_code_attempt": 0, "sensitivity_code_error": "",
        "sensitivity_code_error_kind": "", "sensitivity_pending_runs": [],
        "sensitivity_pending_code": "", "sensitivity_previous_code": "",
    }


def sensitivity_code_generate_node(state: MathModelingState) -> dict:
    """Generate scan code for the checkpointed plan."""
    plan = SensitivityPlan.model_validate(state.sensitivity_plan_dump)
    main_code, primary_metrics = _canonical_primary(state)
    if _use_deterministic_sensitivity():
        code_out = SensitivityCode(code=_build_sensitivity_template_code(plan))
    elif main_code:
        code_out = SensitivityCode(code=_build_canonical_replay_code(plan, main_code))
    else:
        final = next(
            (model for model in reversed(state.model_versions) if model.stage == "final"),
            state.latest_model(),
        )
        try:
            response = complete(
                build_code_prompt(
                    final,
                    [run.model_dump() for run in plan.runs],
                    state.sensitivity_code_error or None,
                    state.sensitivity_code_error_kind,
                    data_dir=state.data_dir,
                    data_files=state.data_files,
                    previous_code=state.sensitivity_previous_code,
                    main_code=main_code,
                    canonical_metrics=primary_metrics,
                ),
                schema=None,
                system=CODE_SYSTEM,
                model=MODEL_ROUTING.get("coder"),
                profile="code",
                temperature=0.1,
                max_tokens=6000,
            )
            code_out = (
                response if isinstance(response, SensitivityCode)
                else SensitivityCode(code=_extract_python_source(response))
            )
        except Exception as exc:
            attempt = state.sensitivity_code_attempt
            if attempt < MAX_CODE_RETRIES:
                return {
                    "sensitivity_phase": "code_generate",
                    "sensitivity_code_attempt": attempt + 1,
                    "sensitivity_code_error": str(exc)[:1000],
                    "sensitivity_code_error_kind": "generation",
                }
            return {
                "errors": [f"sensitivity: code generation failed: {str(exc)[:500]}"],
                "sensitivity_phase": "done",
            }
    return {"sensitivity_pending_code": code_out.code, "sensitivity_phase": "code_execute"}


def sensitivity_code_execute_node(state: MathModelingState) -> dict:
    """Execute checkpointed sensitivity code."""
    plan = SensitivityPlan.model_validate(state.sensitivity_plan_dump)
    workdir = Path(state.output_dir or ".") / "sensitivity"
    workdir.mkdir(parents=True, exist_ok=True)
    attempt = state.sensitivity_code_attempt
    result = run_python(
        state.sensitivity_pending_code,
        workdir=workdir / f"attempt_{attempt}",
        timeout=120,
    )
    if not result.success:
        if attempt < MAX_CODE_RETRIES:
            return {
                "sensitivity_phase": "code_generate", "sensitivity_code_attempt": attempt + 1,
                "sensitivity_code_error": result.stderr,
                "sensitivity_code_error_kind": result.error_kind,
                "sensitivity_previous_code": state.sensitivity_pending_code,
                "sensitivity_pending_code": "",
            }
        return {
            "errors": [f"sensitivity: code execution failed: {result.stderr[:500]}"],
            "sensitivity_phase": "done", "sensitivity_pending_code": "",
        }

    output_failure = detect_output_failure(result.stdout, result.stderr)
    parsed = [] if output_failure else _parse_results(result.stdout)
    aligned: list[SensitivityRun] = []
    for param, vals, res in parsed:
        entry = _match_plan_run(plan, param)
        if entry is None or len(vals) != len(res) or len(vals) < 2:
            continue
        emitted_keys = _parameter_keys(param) | _parameter_keys(entry.parameter)
        fig = next(
            (
                p for p in result.artifact_paths
                if _parameter_keys(
                    re.sub(r"^sensitivity[_-]*", "", Path(p).stem, flags=re.IGNORECASE)
                ) & emitted_keys
            ),
            None,
        )
        aligned.append(SensitivityRun(parameter=entry.parameter, values=vals, metric=entry.metric,
                                      results=res, figure_path=fig))
    _, primary_metrics = _canonical_primary(state)
    alignment_error = _center_alignment_error(aligned, primary_metrics)
    if not aligned or alignment_error:
        reason = output_failure or alignment_error or (
            "未找到可对齐的 RESULT；参数名必须与计划完整名称或括号内别名一致，"
            "且 values/results 长度相同并至少包含 2 个点"
        )
        if attempt < MAX_CODE_RETRIES:
            return {
                "sensitivity_phase": "code_generate",
                "sensitivity_code_attempt": attempt + 1,
                "sensitivity_code_error": reason,
                "sensitivity_code_error_kind": "output_validation",
                "sensitivity_previous_code": state.sensitivity_pending_code,
                "sensitivity_pending_code": "",
            }
        return {
            "errors": [f"sensitivity: output validation failed: {reason[:500]}"],
            "sensitivity_phase": "done", "sensitivity_pending_code": "",
        }
    for run in aligned:
        run.figure_path = _render_verified_figure(run, workdir / f"attempt_{attempt}")
    return {"sensitivity_pending_runs": aligned, "sensitivity_pending_code": "",
            "sensitivity_phase": "interpret"}


def sensitivity_interpret_node(state: MathModelingState) -> dict:
    """Interpret numeric sensitivity results without repeating the scan."""
    aligned = list(state.sensitivity_pending_runs)
    if _use_deterministic_sensitivity():
        interpretations = [_fallback_interpretation(run) for run in aligned]
    else:
        output: Interpretations = complete(
            build_interpret_prompt(aligned),
            schema=Interpretations,
            system=INTERPRET_SYSTEM,
            model=MODEL_ROUTING.get("writer"),
        )
        interpretations = list(output.interpretations[:len(aligned)])
        for run in aligned[len(interpretations):]:
            interpretations.append(_fallback_interpretation(run))
    for run, text in zip(aligned, interpretations):
        run.interpretation = text
    return {
        "sensitivity_runs": aligned, "sensitivity_pending_runs": [],
        "sensitivity_plan_dump": {}, "sensitivity_pending_code": "",
        "sensitivity_previous_code": "",
        "sensitivity_phase": "done",
    }


def sensitivity_node(state: MathModelingState) -> dict:
    """Legacy compatibility path: run the staged sensitivity flow in one node."""
    current = state
    initial_error_count = len(state.errors)
    delta = sensitivity_plan_node(current)
    current = current.model_copy(update=delta)
    for _ in range(10):
        if current.sensitivity_phase == "done":
            break
        if current.sensitivity_phase == "code_generate":
            delta = sensitivity_code_generate_node(current)
        elif current.sensitivity_phase == "code_execute":
            delta = sensitivity_code_execute_node(current)
        elif current.sensitivity_phase == "interpret":
            delta = sensitivity_interpret_node(current)
        else:
            break
        current = current.model_copy(update=delta)
    new_errors = current.errors[initial_error_count:]
    if new_errors:
        return {"errors": new_errors}
    return {"sensitivity_runs": current.sensitivity_runs}
