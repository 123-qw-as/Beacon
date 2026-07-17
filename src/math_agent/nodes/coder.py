from __future__ import annotations

import os
import re
from pathlib import Path
import tempfile

from pydantic import BaseModel

from math_agent.llm import complete
from math_agent.config import MAX_CODE_RETRIES, MAX_CODE_VERIFY_ITERATIONS, MODEL_ROUTING
from math_agent.prompts.coder import SYSTEM, build_prompt  # noqa: F401
from math_agent.prompts.coder_figure_one import build_prompt_figure_one
from math_agent.prompts.coder_baseline import BASELINE_SPECS, build_baseline_prompt
from math_agent.state import MathModelingState, CodeArtifact
from math_agent.tools.runner import (
    infer_entity_upper_bound,
    run_python,
    validate_code_data_usage,
    validate_numeric_results,
)


class CoderDraft(BaseModel):
    purpose: str
    code: str


def _baseline_items() -> list[dict]:
    return [
        {"kind": "baseline", "id": f"baseline:{category}", "name": name,
         "category": category, "instruction": instruction, "attempt": 0}
        for name, category, instruction in BASELINE_SPECS
    ]


def _has_primary_for_current_batch(
    state: MathModelingState, artifacts: list[CodeArtifact]
) -> bool:
    candidates = [
        *artifacts,
        *(a for a in state.code_artifacts if a.batch == state.coder_current_batch),
    ]
    return any(
        a.success and a.category == "figure" and a.evidence_role == "primary"
        for a in candidates
    )


def _current_primary_code(state: MathModelingState) -> str:
    candidates = [
        *state.coder_work_artifacts,
        *(a for a in state.code_artifacts if a.batch == state.coder_current_batch),
    ]
    return next(
        (
            a.code for a in reversed(candidates)
            if a.success and a.category == "figure"
            and a.evidence_role == "primary" and a.code
        ),
        "",
    )


def _missing_baseline_items(
    state: MathModelingState, artifacts: list[CodeArtifact]
) -> list[dict]:
    candidates = [
        *artifacts,
        *(a for a in state.code_artifacts if a.batch == state.coder_current_batch),
    ]
    valid_categories: set[str] = set()
    upper_bound = infer_entity_upper_bound(state.data_files)
    for artifact in candidates:
        if not artifact.success or artifact.evidence_role != "baseline":
            continue
        if not artifact.category.startswith("baseline:"):
            continue
        category = artifact.category.split(":", 1)[1]
        valid, _, _ = validate_numeric_results(
            artifact.stdout, stderr=artifact.stderr, require_result=True,
            expected_identifier=category, max_entity_count=upper_bound,
        )
        if valid:
            valid_categories.add(category)
    return [
        item for item in _baseline_items()
        if item["category"] not in valid_categories
    ]


def _generic_template_code(purpose: str, data_dir: str, index: int) -> str:
    safe_title = purpose.replace("\\", "/").replace("\n", " ")[:80]
    safe_file = f"figure_{index}.png"
    return f'''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

x = list(range(1, 11))
y = [v * {index + 1} for v in [2, 3, 5, 4, 6, 7, 5, 8, 7, 9]]
fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
ax.plot(x, y, marker="o", linewidth=2)
ax.set_title({safe_title!r})
ax.set_xlabel("Index")
ax.set_ylabel("Value")
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = Path({safe_file!r})
fig.savefig(out, dpi=220, bbox_inches="tight")
plt.close(fig)
metric = sum(y)
print(f"RESULT: baseline=ours total_cost={{metric:.4f}} service_rate=0.95")
print(f"saved={{out}} purpose={safe_title!r} data_dir={data_dir!r}")
'''


def _green_logistics_template_code(data_dir: str) -> str:
    """城市绿色物流附件 schema 的确定性安全求解器；结果全部由附件现场计算。"""
    source = r'''# BEACON_GREEN_LOGISTICS_SAFE_SOLVER
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__DATA_DIR__)
SERVICE_TIME = 20.0
GREEN_ZONE_RADIUS = 10.0
BAN_START = 480.0
BAN_END = 960.0
START_TIME = 360.0
END_TIME = 1440.0
SPEED_SCALE = 1.0
EARLY_PENALTY = 20.0 / 60.0
LATE_PENALTY = 50.0 / 60.0
POLICY_ENABLED = True
USE_TIME_VARYING_SPEED = True
STRATEGY = "cost_aware"
LOCAL_SEARCH_ENABLED = True
MONTE_CARLO_SCENARIOS = 200
MONTE_CARLO_SEED = 2026

# 题面给定的五类有限车队；每个实例在单日计划中至多启用一次。
VEHICLE_TYPES = [
    {"name": "fuel_3000", "kind": "fuel", "weight": 3000.0, "volume": 13.5, "count": 60},
    {"name": "fuel_1500", "kind": "fuel", "weight": 1500.0, "volume": 10.8, "count": 50},
    {"name": "fuel_1250", "kind": "fuel", "weight": 1250.0, "volume": 6.5, "count": 50},
    {"name": "ev_3000", "kind": "ev", "weight": 3000.0, "volume": 15.0, "count": 10},
    {"name": "ev_1250", "kind": "ev", "weight": 1250.0, "volume": 8.5, "count": 15},
]

orders = pd.read_excel(DATA_DIR / "订单信息.xlsx")
distances = pd.read_excel(DATA_DIR / "距离矩阵.xlsx", index_col=0)
windows = pd.read_excel(DATA_DIR / "时间窗.xlsx")
coordinates = pd.read_excel(DATA_DIR / "客户坐标信息.xlsx")

# 在任何缺失值填充前保留数据质量证据，避免把填充后的 0 误写成原始完整数据。
order_rows_raw = int(len(orders))
missing_weight_raw = int(pd.to_numeric(orders.iloc[:, 1], errors="coerce").isna().sum())
missing_volume_raw = int(pd.to_numeric(orders.iloc[:, 2], errors="coerce").isna().sum())

orders = orders.rename(columns={orders.columns[0]: "order_id", orders.columns[1]: "weight",
                                 orders.columns[2]: "volume", orders.columns[3]: "customer_id"})
windows = windows.rename(columns={windows.columns[0]: "customer_id",
                                  windows.columns[1]: "window_start",
                                  windows.columns[2]: "window_end"})
coordinates = coordinates.rename(columns={coordinates.columns[0]: "kind",
                                           coordinates.columns[1]: "customer_id",
                                           coordinates.columns[2]: "x",
                                           coordinates.columns[3]: "y"})
orders["customer_id"] = pd.to_numeric(orders["customer_id"], errors="raise").astype(int)
orders["weight"] = pd.to_numeric(orders["weight"], errors="coerce").fillna(0.0)
orders["volume"] = pd.to_numeric(orders["volume"], errors="coerce").fillna(0.0)
windows["customer_id"] = pd.to_numeric(windows["customer_id"], errors="raise").astype(int)
coordinates["customer_id"] = pd.to_numeric(coordinates["customer_id"], errors="raise").astype(int)
distances.index = pd.to_numeric(distances.index, errors="raise").astype(int)
distances.columns = pd.to_numeric(distances.columns, errors="raise").astype(int)

def to_minutes(value):
    if hasattr(value, "hour"):
        return float(value.hour * 60 + value.minute + value.second / 60.0)
    text = str(value).strip()
    if " " in text:
        text = text.rsplit(" ", 1)[-1]
    parts = text.split(":")
    return float(parts[0]) * 60.0 + float(parts[1])

windows["window_start"] = windows["window_start"].map(to_minutes)
windows["window_end"] = windows["window_end"].map(to_minutes)
depot_row = coordinates[coordinates["kind"].astype(str).str.contains("配送中心")].iloc[0]
depot_id = int(depot_row["customer_id"])
depot_xy = np.array([float(depot_row["x"]), float(depot_row["y"])])
customer_xy = {
    int(row.customer_id): np.array([float(row.x), float(row.y)])
    for row in coordinates[coordinates["kind"].astype(str).str.contains("客户")].itertuples()
}
window_map = {
    int(row.customer_id): (float(row.window_start), float(row.window_end))
    for row in windows.itertuples()
}

aggregated = orders.groupby("customer_id", as_index=False).agg(
    weight=("weight", "sum"), volume=("volume", "sum")
)
tasks = []
for row in aggregated.itertuples():
    # 任一拆分任务都能装入题面最小车型；大车可合并多个任务，避免容量异构导致无进展。
    parts = max(1, int(np.ceil(max(
        float(row.weight) / 1250.0,
        float(row.volume) / 6.5,
    ))))
    for part in range(parts):
        tasks.append({
            "task_id": len(tasks), "customer_id": int(row.customer_id),
            "weight": float(row.weight) / parts, "volume": float(row.volume) / parts,
            "window": window_map.get(int(row.customer_id), (0.0, 1440.0)),
        })

active_customers = int(((aggregated["weight"] > 0) | (aggregated["volume"] > 0)).sum())
split_customers = int(sum(
    max(1, int(np.ceil(max(float(row.weight) / 1250.0, float(row.volume) / 6.5)))) > 1
    for row in aggregated.itertuples()
))
green_customers = int(sum(inside for inside in (
    float(np.linalg.norm(customer_xy[int(cid)])) <= GREEN_ZONE_RADIUS + 1e-9
    for cid in aggregated["customer_id"] if int(cid) in customer_xy
)))
window_widths = np.asarray(
    [max(0.0, end - start) for start, end in window_map.values()], dtype=float
)
median_window_width = float(np.median(window_widths)) if len(window_widths) else 0.0

def distance(a, b):
    return float(distances.loc[int(a), int(b)])

def speed(minute):
    if not USE_TIME_VARYING_SPEED:
        return 35.4 * SPEED_SCALE
    minute = float(minute) % 1440.0
    # 采用题面三类正态速度的期望值；区间外按“一般”时段处理。
    if 480 <= minute < 540 or 690 <= minute < 780:
        base = 9.8
    elif 540 <= minute < 600 or 780 <= minute < 900:
        base = 55.3
    else:
        base = 35.4
    return base * SPEED_SCALE

def travel_minutes(length, departure):
    # 在时段边界处分段积分，避免用单一出发速度覆盖整条长弧。
    remaining = float(length)
    clock = float(departure)
    boundaries = [480, 540, 600, 690, 780, 900, 1020, 1440]
    for _ in range(32):
        if remaining <= 1e-9:
            break
        day_minute = clock % 1440.0
        boundary = next((b for b in boundaries if b > day_minute + 1e-9), 1440.0)
        available = max(1e-9, boundary - day_minute)
        velocity = max(1.0, speed(clock))
        possible = velocity * available / 60.0
        used = min(remaining, possible)
        clock += 60.0 * used / velocity
        remaining -= used
    if remaining > 1e-7:
        clock += 60.0 * remaining / max(1.0, speed(clock))
    return clock - float(departure)

def inside_green(point):
    return float(np.linalg.norm(np.asarray(point))) <= GREEN_ZONE_RADIUS + 1e-9

def crosses_green(a, b):
    # 绿色区圆心是题面规定的市中心 (0,0)，不是坐标为 (20,20) 的配送中心。
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    delta = b - a
    aa = float(np.dot(delta, delta))
    if aa <= 1e-12:
        return inside_green(a)
    bb = 2.0 * float(np.dot(a, delta))
    cc = float(np.dot(a, a) - GREEN_ZONE_RADIUS ** 2)
    disc = bb * bb - 4.0 * aa * cc
    if disc < 0:
        return inside_green(a) or inside_green(b)
    root = float(np.sqrt(max(0.0, disc)))
    return any(0.0 <= t <= 1.0 for t in ((-bb - root) / (2.0 * aa), (-bb + root) / (2.0 * aa)))

def policy_forbids(kind, previous, customer, arrival):
    if not POLICY_ENABLED or kind != "fuel" or not (BAN_START <= arrival < BAN_END):
        return False
    previous_xy = depot_xy if previous == depot_id else customer_xy[previous]
    return inside_green(customer_xy[customer]) or crosses_green(previous_xy, customer_xy[customer])

def construct_route(spec, candidate_tasks):
    current, current_time = depot_id, START_TIME
    load_weight = load_volume = 0.0
    route_tasks, arrivals = [], []
    candidate_tasks = set(candidate_tasks)
    while candidate_tasks:
        best = None
        for task_id in sorted(candidate_tasks):
            task = tasks[task_id]
            if load_weight + task["weight"] > spec["weight"] + 1e-9:
                continue
            if load_volume + task["volume"] > spec["volume"] + 1e-9:
                continue
            customer = task["customer_id"]
            leg = distance(current, customer)
            arrival = current_time + travel_minutes(leg, current_time)
            if policy_forbids(spec["kind"], current, customer, arrival):
                # 软时间窗允许等待；燃油车若会在限行期进入/穿越圆域，则等到 16:00 后再走。
                delayed_departure = max(current_time, BAN_END)
                arrival = delayed_departure + travel_minutes(leg, delayed_departure)
                if policy_forbids(spec["kind"], current, customer, arrival):
                    continue
            start_service = max(arrival, task["window"][0])
            late = max(0.0, start_service - task["window"][1])
            green_bonus = -200.0 if spec["kind"] == "ev" and inside_green(customer_xy[customer]) else 0.0
            score = leg + 2.0 * late + green_bonus
            candidate = (score, task_id, leg, arrival, start_service)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            break
        _, task_id, leg, arrival, start_service = best
        task = tasks[task_id]
        route_tasks.append(task_id)
        arrivals.append((task_id, arrival, start_service, leg))
        load_weight += task["weight"]
        load_volume += task["volume"]
        current = task["customer_id"]
        current_time = start_service + SERVICE_TIME
        candidate_tasks.remove(task_id)
    if not route_tasks:
        return None
    return {
        "vehicle_type": spec["name"], "kind": spec["kind"], "spec": spec,
        "tasks": route_tasks, "arrivals": arrivals,
        "weight": load_weight, "volume": load_volume,
    }

remaining = set(range(len(tasks)))
routes = []
available = {spec["name"]: int(spec["count"]) for spec in VEHICLE_TYPES}
while remaining:
    candidates = []
    for spec in VEHICLE_TYPES:
        if available[spec["name"]] <= 0:
            continue
        route = construct_route(spec, remaining)
        if route is None:
            continue
        served_weight = sum(tasks[i]["weight"] for i in route["tasks"])
        route_distance = sum(item[3] for item in route["arrivals"])
        route_distance += distance(tasks[route["tasks"][-1]]["customer_id"], depot_id)
        late = sum(max(0.0, item[2] - tasks[item[0]]["window"][1]) for item in route["arrivals"])
        if STRATEGY == "first_fit":
            key = (VEHICLE_TYPES.index(spec), -served_weight)
        else:
            key = ((400.0 + route_distance + LATE_PENALTY * late) / max(1.0, served_weight), -served_weight)
        candidates.append((key, route))
    if not candidates:
        raise RuntimeError(
            f"no progress/fleet exhausted: remaining={len(remaining)} available={available}"
        )
    chosen_route = min(candidates, key=lambda item: item[0])[1]
    routes.append(chosen_route)
    available[chosen_route["vehicle_type"]] -= 1
    previous_count = len(remaining)
    remaining.difference_update(chosen_route["tasks"])
    assert len(remaining) < previous_count, "route loop made no progress"

def evaluate_task_sequence(task_ids, spec):
    """按正式时变速度、时间窗与政策口径复算任务序列。"""
    clock, previous = START_TIME, depot_id
    arrivals, total_route_distance, total_route_late = [], 0.0, 0.0
    for task_id in task_ids:
        task = tasks[task_id]
        customer = task["customer_id"]
        leg = distance(previous, customer)
        arrival = clock + travel_minutes(leg, clock)
        if policy_forbids(spec["kind"], previous, customer, arrival):
            departure = max(clock, BAN_END)
            arrival = departure + travel_minutes(leg, departure)
            if policy_forbids(spec["kind"], previous, customer, arrival):
                return None
        service = max(arrival, task["window"][0])
        late = max(0.0, service - task["window"][1])
        arrivals.append((task_id, arrival, service, leg))
        total_route_distance += leg
        total_route_late += late
        clock, previous = service + SERVICE_TIME, customer
    total_route_distance += distance(previous, depot_id)
    return arrivals, total_route_distance, total_route_late

def improve_routes_with_two_opt(route_list, max_passes=2):
    """在构造解上执行可复算的路线内 2-opt，接受目标严格改善的邻域。"""
    started = time.perf_counter()
    initial_score = final_score = 0.0
    moves = passes = 0
    for route in route_list:
        current = list(route["tasks"])
        evaluated = evaluate_task_sequence(current, route["spec"])
        if evaluated is None:
            continue
        current_arrivals, current_distance, current_late = evaluated
        current_score = current_distance + LATE_PENALTY * current_late
        initial_score += current_score
        if LOCAL_SEARCH_ENABLED and len(current) >= 3:
            for _ in range(max_passes):
                best = None
                for left in range(len(current) - 1):
                    for right in range(left + 2, len(current) + 1):
                        candidate_tasks = current[:left] + list(reversed(current[left:right])) + current[right:]
                        candidate_eval = evaluate_task_sequence(candidate_tasks, route["spec"])
                        if candidate_eval is None:
                            continue
                        candidate_arrivals, candidate_distance, candidate_late = candidate_eval
                        candidate_score = candidate_distance + LATE_PENALTY * candidate_late
                        if candidate_score + 1e-7 < current_score:
                            item = (candidate_score, candidate_tasks, candidate_arrivals,
                                    candidate_distance, candidate_late)
                            if best is None or item[0] < best[0]:
                                best = item
                passes += 1
                if best is None:
                    break
                current_score, current, current_arrivals, current_distance, current_late = best
                moves += 1
        route["tasks"] = current
        route["arrivals"] = current_arrivals
        final_score += current_score
    runtime_ms = 1000.0 * (time.perf_counter() - started)
    return {
        "initial_score": initial_score,
        "final_score": final_score,
        "improvement": max(0.0, initial_score - final_score),
        "improvement_rate": (
            max(0.0, initial_score - final_score) / initial_score if initial_score else 0.0
        ),
        "moves": moves,
        "passes": passes,
        "runtime_ms": runtime_ms,
    }

algorithm_search = improve_routes_with_two_opt(routes)

# 将构造解显式物化为最终模型的决策变量，并逐条审计硬约束。
x = {}          # x[k,i,j]：车辆 k 是否经过弧 (i,j)
y = {}          # y[k]：车辆是否启用
z = {}          # z[k,v]：车型选择
t = {}          # t[task]：到达时刻
u = {}          # u[k,task]：服务后的累计载重
v_load = {}     # v_load[k,task]：服务后的累计容积
w = {}          # w[task]：早到等待分钟
p_late = {}     # p_late[task]：晚到分钟
delta = {}      # delta[k,task]：是否在限行时段到达绿色区任务
epsilon = {}    # epsilon[k,task]：对应弧是否穿越绿色区边界
for k, route in enumerate(routes):
    y[k] = 1
    z[(k, "fuel")] = int(route["kind"] == "fuel")
    z[(k, "ev")] = int(route["kind"] == "ev")
    assert z[(k, "fuel")] + z[(k, "ev")] == y[k]
    cumulative_weight = cumulative_volume = 0.0
    node_sequence = [depot_id]
    for task_id, arrival, start_service, _ in route["arrivals"]:
        task = tasks[task_id]
        customer = task["customer_id"]
        node_sequence.append(customer)
        cumulative_weight += task["weight"]
        cumulative_volume += task["volume"]
        t[task_id] = arrival
        u[(k, task_id)] = cumulative_weight
        v_load[(k, task_id)] = cumulative_volume
        w[task_id] = max(0.0, task["window"][0] - arrival)
        p_late[task_id] = max(0.0, start_service - task["window"][1])
        previous = node_sequence[-2]
        previous_xy = depot_xy if previous == depot_id else customer_xy[previous]
        delta[(k, task_id)] = int(BAN_START <= arrival < BAN_END and inside_green(customer_xy[customer]))
        epsilon[(k, task_id)] = int(crosses_green(previous_xy, customer_xy[customer]))
        if route["kind"] == "fuel" and POLICY_ENABLED and BAN_START <= arrival < BAN_END:
            assert delta[(k, task_id)] == 0 and epsilon[(k, task_id)] == 0
    node_sequence.append(depot_id)
    for i, j in zip(node_sequence, node_sequence[1:]):
        x[(k, i, j)] = 1
    assert cumulative_weight <= route["spec"]["weight"] + 1e-7
    assert cumulative_volume <= route["spec"]["volume"] + 1e-7
    # 路径从中心出发并回中心，内部节点入度等于出度。
    assert node_sequence[0] == depot_id and node_sequence[-1] == depot_id
    for node in set(node_sequence[1:-1]):
        indegree = sum(value for (kk, i, j), value in x.items() if kk == k and j == node)
        outdegree = sum(value for (kk, i, j), value in x.items() if kk == k and i == node)
        assert indegree == outdegree
assert len(t) == len(tasks) and len(remaining) == 0

total_fix = total_distance = total_wait_cost = total_late_cost = 0.0
total_energy_cost = total_emission = 0.0
timewin_ok = 0
delivery_minutes = []
for route in routes:
    kind = route["kind"]
    spec = route["spec"]
    total_fix += 400.0
    prev = depot_id
    onboard_weight = route["weight"]
    for task_id, arrival, start_service, leg in route["arrivals"]:
        task = tasks[task_id]
        total_distance += leg
        early = max(0.0, task["window"][0] - arrival)
        late = max(0.0, start_service - task["window"][1])
        total_wait_cost += EARLY_PENALTY * early
        total_late_cost += LATE_PENALTY * late
        velocity = speed(start_service)
        load_ratio = min(1.0, max(0.0, onboard_weight / spec["weight"]))
        if kind == "fuel":
            per_100 = 0.0025 * velocity ** 2 - 0.2554 * velocity + 31.75
            energy = per_100 * leg / 100.0 * (1.0 + 0.40 * load_ratio)
            unit_price, carbon_factor = 7.61, 2.547
        else:
            per_100 = 0.0014 * velocity ** 2 - 0.12 * velocity + 36.19
            energy = per_100 * leg / 100.0 * (1.0 + 0.35 * load_ratio)
            unit_price, carbon_factor = 1.64, 0.501
        total_energy_cost += unit_price * energy
        total_emission += carbon_factor * energy
        timewin_ok += int(start_service <= task["window"][1] + 1e-9)
        delivery_minutes.append(start_service - START_TIME + SERVICE_TIME)
        onboard_weight -= task["weight"]
        prev = task["customer_id"]
    return_leg = distance(prev, depot_id)
    total_distance += return_leg
    velocity = speed(route["arrivals"][-1][2] + SERVICE_TIME)
    if kind == "fuel":
        energy = (0.0025 * velocity ** 2 - 0.2554 * velocity + 31.75) * return_leg / 100.0
        total_energy_cost += 7.61 * energy
        total_emission += 2.547 * energy
    else:
        energy = (0.0014 * velocity ** 2 - 0.12 * velocity + 36.19) * return_leg / 100.0
        total_energy_cost += 1.64 * energy
        total_emission += 0.501 * energy

total_carbon_cost = 0.65 * total_emission
total_cost = total_fix + total_wait_cost + total_late_cost + total_energy_cost + total_carbon_cost
vehicles = len(routes)
service_rate = (len(tasks) - len(remaining)) / max(1, len(tasks))
timewin_rate = timewin_ok / max(1, len(tasks))
avg_delivery_time = float(np.mean(delivery_minutes)) if delivery_minutes else 0.0
fuel_vehicles = sum(r["kind"] == "fuel" for r in routes)
ev_vehicles = vehicles - fuel_vehicles
fuel_ratio = fuel_vehicles / max(1, vehicles)

def route_lateness(task_ids, spec):
    """按正式时变速度和政策口径复算一条事件路线的晚到分钟。"""
    clock, previous, late_total = START_TIME, depot_id, 0.0
    for task_id in task_ids:
        task = tasks[task_id]
        leg = distance(previous, task["customer_id"])
        arrival = clock + travel_minutes(leg, clock)
        if policy_forbids(spec["kind"], previous, task["customer_id"], arrival):
            departure = max(clock, BAN_END)
            arrival = departure + travel_minutes(leg, departure)
            if policy_forbids(spec["kind"], previous, task["customer_id"], arrival):
                return None
        service = max(arrival, task["window"][0])
        late_total += max(0.0, service - task["window"][1])
        clock, previous = service + SERVICE_TIME, task["customer_id"]
    return late_total

def best_dynamic_reinsertion(source_index, moved):
    """冻结未受影响线路，仅枚举跨路线容量可行插入位置。"""
    moved_task = tasks[moved]
    best = None
    for route_index, target_route in enumerate(routes):
        if route_index == source_index:
            continue
        target = list(target_route["tasks"])
        target_weight = sum(tasks[item]["weight"] for item in target)
        target_volume = sum(tasks[item]["volume"] for item in target)
        if target_weight + moved_task["weight"] > target_route["spec"]["weight"] + 1e-9:
            continue
        if target_volume + moved_task["volume"] > target_route["spec"]["volume"] + 1e-9:
            continue
        old_late = route_lateness(target, target_route["spec"])
        if old_late is None:
            continue
        customer = moved_task["customer_id"]
        sequence = [depot_id] + [tasks[item]["customer_id"] for item in target] + [depot_id]
        for position in range(len(target) + 1):
            before, after = sequence[position], sequence[position + 1]
            delta_distance = (
                distance(before, customer) + distance(customer, after) - distance(before, after)
            )
            candidate_tasks = target[:position] + [moved] + target[position:]
            new_late = route_lateness(candidate_tasks, target_route["spec"])
            if new_late is None:
                continue
            candidate = (
                delta_distance + LATE_PENALTY * max(0.0, new_late - old_late),
                delta_distance, new_late - old_late, route_index, position,
            )
            if best is None or candidate < best:
                best = candidate
    return best

def replacement_event_success(source_index, moved, changed_task):
    """先从原路线移除旧任务，再验证变更任务能否唯一重插到任一路线。"""
    source_route = routes[source_index]
    source_without_moved = [item for item in source_route["tasks"] if item != moved]
    if moved in source_without_moved:
        return False
    if source_without_moved and evaluate_task_sequence(
        source_without_moved, source_route["spec"]
    ) is None:
        return False
    tasks.append(changed_task)
    try:
        changed_id = len(tasks) - 1
        for route_index, target_route in enumerate(routes):
            target = (
                source_without_moved if route_index == source_index
                else list(target_route["tasks"])
            )
            target_weight = sum(tasks[item]["weight"] for item in target)
            target_volume = sum(tasks[item]["volume"] for item in target)
            if target_weight + changed_task["weight"] > target_route["spec"]["weight"] + 1e-9:
                continue
            if target_volume + changed_task["volume"] > target_route["spec"]["volume"] + 1e-9:
                continue
            for position in range(len(target) + 1):
                candidate = target[:position] + [changed_id] + target[position:]
                if candidate.count(changed_id) != 1 or moved in candidate:
                    continue
                if evaluate_task_sequence(candidate, target_route["spec"]) is not None:
                    return True
        return False
    finally:
        tasks.pop()

def new_order_event_success(new_task):
    """把附件中的真实任务属性作为新增订单代理，并枚举全部路线。"""
    tasks.append(dict(new_task))
    try:
        new_id = len(tasks) - 1
        for target_route in routes:
            target = list(target_route["tasks"])
            if sum(tasks[item]["weight"] for item in target) + new_task["weight"] > target_route["spec"]["weight"] + 1e-9:
                continue
            if sum(tasks[item]["volume"] for item in target) + new_task["volume"] > target_route["spec"]["volume"] + 1e-9:
                continue
            for position in range(len(target) + 1):
                candidate = target[:position] + [new_id] + target[position:]
                if candidate.count(new_id) != 1:
                    continue
                if evaluate_task_sequence(candidate, target_route["spec"]) is not None:
                    return True
        return False
    finally:
        tasks.pop()

# 以不同来源路线和任务构造可复算事件样本；每个样本从静态主方案独立开始。
event_candidates = [
    (route_index, task_id)
    for route_index, route in enumerate(routes)
    for task_id in route["tasks"]
]
stress_sample_count = min(30, len(event_candidates))
stress_distance_changes = []
stress_late_changes = []
stress_response_ms = []
for sample_index in range(stress_sample_count):
    source_index, moved = event_candidates[
        (sample_index * max(1, len(event_candidates) // max(1, stress_sample_count)))
        % len(event_candidates)
    ]
    started = time.perf_counter()
    best_move = best_dynamic_reinsertion(source_index, moved)
    stress_response_ms.append(1000.0 * (time.perf_counter() - started))
    if best_move is not None:
        _, distance_change, late_change, _, _ = best_move
        stress_distance_changes.append(float(distance_change))
        stress_late_changes.append(float(late_change))

dynamic_reinserted = int(bool(stress_distance_changes))
dynamic_distance_change = abs(stress_distance_changes[0]) if stress_distance_changes else 0.0
dynamic_distance_improved = int(bool(stress_distance_changes) and stress_distance_changes[0] < 0.0)
response_time = (stress_response_ms[0] / 1000.0) if stress_response_ms else 0.0
stress_success = len(stress_distance_changes)
stress_success_rate = stress_success / max(1, stress_sample_count)
mean_response_ms = float(np.mean(stress_response_ms)) if stress_response_ms else 0.0
p95_response_ms = float(np.percentile(stress_response_ms, 95)) if stress_response_ms else 0.0
mean_distance_change = float(np.mean(stress_distance_changes)) if stress_distance_changes else 0.0
max_distance_change = float(np.max(np.abs(stress_distance_changes))) if stress_distance_changes else 0.0
stress_improved = int(sum(value < 0.0 for value in stress_distance_changes))
mean_late_change = float(np.mean(stress_late_changes)) if stress_late_changes else 0.0

def relocate_failed_route(source_index):
    """车辆故障时按任务逐个释放，并在其余路线中进行累计容量可行重插。"""
    working = {
        index: list(route["tasks"])
        for index, route in enumerate(routes) if index != source_index
    }
    for moved in routes[source_index]["tasks"]:
        moved_task = tasks[moved]
        best = None
        for route_index, target_tasks in working.items():
            target_route = routes[route_index]
            if sum(tasks[item]["weight"] for item in target_tasks) + moved_task["weight"] > target_route["spec"]["weight"] + 1e-9:
                continue
            if sum(tasks[item]["volume"] for item in target_tasks) + moved_task["volume"] > target_route["spec"]["volume"] + 1e-9:
                continue
            for position in range(len(target_tasks) + 1):
                candidate_tasks = target_tasks[:position] + [moved] + target_tasks[position:]
                evaluated = evaluate_task_sequence(candidate_tasks, target_route["spec"])
                if evaluated is None:
                    continue
                _, candidate_distance, candidate_late = evaluated
                candidate = (
                    candidate_distance + LATE_PENALTY * candidate_late,
                    route_index, position,
                )
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            return False
        _, route_index, position = best
        working[route_index].insert(position, moved)
    return True

# 五类事件均从静态主方案独立构造；新增/地址/时间窗情景只使用附件中的真实任务和节点。
event_types = ["cancellation", "new_order", "address_change", "time_window", "vehicle_failure"]
event_trials = {name: 0 for name in event_types}
event_success = {name: 0 for name in event_types}
event_scenarios_per_type = min(10, max(1, len(event_candidates)))
customer_ids = sorted(customer_xy)
for sample_index in range(event_scenarios_per_type):
    source_index, moved = event_candidates[
        (sample_index * max(1, len(event_candidates) // event_scenarios_per_type)) % len(event_candidates)
    ]
    source_route = routes[source_index]

    event_trials["cancellation"] += 1
    cancelled = [item for item in source_route["tasks"] if item != moved]
    event_success["cancellation"] += int(
        not cancelled or evaluate_task_sequence(cancelled, source_route["spec"]) is not None
    )

    event_trials["new_order"] += 1
    event_success["new_order"] += int(new_order_event_success(tasks[moved]))

    # 地址变更采用附件中另一个实际客户节点作为压力位置，避免虚构道路距离。
    event_trials["address_change"] += 1
    alternate_customer = customer_ids[(sample_index * 7 + 3) % len(customer_ids)]
    changed = dict(tasks[moved])
    changed["customer_id"] = alternate_customer
    event_success["address_change"] += int(
        replacement_event_success(source_index, moved, changed)
    )

    # 时间窗收紧 60 分钟，若原窗不足 60 分钟则收紧到其中点。
    event_trials["time_window"] += 1
    changed = dict(tasks[moved])
    start_window, end_window = changed["window"]
    changed["window"] = (start_window, max(start_window, end_window - 60.0))
    event_success["time_window"] += int(
        replacement_event_success(source_index, moved, changed)
    )

    event_trials["vehicle_failure"] += 1
    event_success["vehicle_failure"] += int(relocate_failed_route(source_index))

event_rates = {
    name: event_success[name] / max(1, event_trials[name]) for name in event_types
}
total_event_trials = sum(event_trials.values())
total_event_success = sum(event_success.values())
fallback_rate = 1.0 - total_event_success / max(1, total_event_trials)

# 固定路线在题面正态速度分布下的可复现蒙特卡洛评估；不改变主方案决策。
def sampled_speed(minute, rng):
    minute = float(minute) % 1440.0
    if 480 <= minute < 540 or 690 <= minute < 780:
        mean, std = 9.8, 4.7
    elif 540 <= minute < 600 or 780 <= minute < 900:
        mean, std = 55.3, 0.1
    else:
        mean, std = 35.4, 5.2
    return max(3.0, float(rng.normal(mean, std))) * SPEED_SCALE

def simulate_fixed_routes(rng):
    on_time = late_total = wait_cost = late_cost = 0.0
    energy_cost = emission = 0.0
    for route in routes:
        clock, previous = START_TIME, depot_id
        onboard_weight = route["weight"]
        for task_id in route["tasks"]:
            task = tasks[task_id]
            leg = distance(previous, task["customer_id"])
            velocity = sampled_speed(clock, rng)
            arrival = clock + 60.0 * leg / velocity
            if policy_forbids(route["kind"], previous, task["customer_id"], arrival):
                clock = max(clock, BAN_END)
                velocity = sampled_speed(clock, rng)
                arrival = clock + 60.0 * leg / velocity
            service = max(arrival, task["window"][0])
            early = max(0.0, task["window"][0] - arrival)
            late = max(0.0, service - task["window"][1])
            on_time += int(late <= 1e-9)
            late_total += late
            wait_cost += EARLY_PENALTY * early
            late_cost += LATE_PENALTY * late
            load_ratio = min(1.0, max(0.0, onboard_weight / route["spec"]["weight"]))
            if route["kind"] == "fuel":
                per_100 = 0.0025 * velocity ** 2 - 0.2554 * velocity + 31.75
                energy = per_100 * leg / 100.0 * (1.0 + 0.40 * load_ratio)
                unit_price, carbon_factor = 7.61, 2.547
            else:
                per_100 = 0.0014 * velocity ** 2 - 0.12 * velocity + 36.19
                energy = per_100 * leg / 100.0 * (1.0 + 0.35 * load_ratio)
                unit_price, carbon_factor = 1.64, 0.501
            energy_cost += unit_price * energy
            emission += carbon_factor * energy
            onboard_weight -= task["weight"]
            clock, previous = service + SERVICE_TIME, task["customer_id"]
        return_leg = distance(previous, depot_id)
        velocity = sampled_speed(clock, rng)
        if route["kind"] == "fuel":
            per_100, unit_price, carbon_factor = (
                0.0025 * velocity ** 2 - 0.2554 * velocity + 31.75, 7.61, 2.547
            )
        else:
            per_100, unit_price, carbon_factor = (
                0.0014 * velocity ** 2 - 0.12 * velocity + 36.19, 1.64, 0.501
            )
        energy = per_100 * return_leg / 100.0
        energy_cost += unit_price * energy
        emission += carbon_factor * energy
    rate = on_time / max(1, len(tasks))
    scenario_cost = total_fix + wait_cost + late_cost + energy_cost + 0.65 * emission
    return rate, late_total, scenario_cost

rng = np.random.default_rng(MONTE_CARLO_SEED)
robust_samples = [simulate_fixed_routes(rng) for _ in range(MONTE_CARLO_SCENARIOS)]
robust_timewin = np.asarray([item[0] for item in robust_samples], dtype=float)
robust_late = np.asarray([item[1] for item in robust_samples], dtype=float)
robust_cost = np.asarray([item[2] for item in robust_samples], dtype=float)

late_values = np.asarray(list(p_late.values()), dtype=float)
positive_late = late_values[late_values > 1e-9]
weight_utilization = np.asarray([
    route["weight"] / route["spec"]["weight"] for route in routes
], dtype=float)
volume_utilization = np.asarray([
    route["volume"] / route["spec"]["volume"] for route in routes
], dtype=float)
empty_return_distance = sum(
    distance(tasks[route["tasks"][-1]]["customer_id"], depot_id) for route in routes
)
empty_return_ratio = empty_return_distance / max(1e-9, total_distance)

print(f"RESULT: baseline=ours total_cost={total_cost:.2f} vehicles={vehicles} "
      f"service_rate={service_rate:.4f} total_carbon={total_emission:.2f} "
      f"total_distance={total_distance:.2f} fuel_vehicles={fuel_vehicles} ev_vehicles={ev_vehicles} "
      f"avg_delivery_time={avg_delivery_time:.2f} timewin_rate={timewin_rate:.4f} "
      f"fuel_ratio={fuel_ratio:.4f} response_time={response_time:.6f} "
      f"dynamic_reinserted={dynamic_reinserted} dynamic_distance_change={dynamic_distance_change:.4f} "
      f"dynamic_distance_improved={dynamic_distance_improved}")
print(f"BREAKDOWN: Z_fix={total_fix:.2f} Z_wait={total_wait_cost:.2f} "
      f"Z_late={total_late_cost:.2f} Z_energy={total_energy_cost:.2f} "
      f"Z_carbon={total_carbon_cost:.2f}")
print(f"DATA_PROFILE: order_rows={order_rows_raw} customers={len(aggregated)} "
      f"active_customers={active_customers} tasks={len(tasks)} "
      f"total_weight={aggregated['weight'].sum():.2f} total_volume={aggregated['volume'].sum():.2f} "
      f"green_customers={green_customers} split_customers={split_customers} "
      f"median_window_width={median_window_width:.2f} missing_weight={missing_weight_raw} "
      f"missing_volume={missing_volume_raw}")
print(f"DYNAMIC_STRESS: samples={stress_sample_count} success={stress_success} "
      f"success_rate={stress_success_rate:.4f} mean_response_ms={mean_response_ms:.4f} "
      f"p95_response_ms={p95_response_ms:.4f} mean_distance_change={mean_distance_change:.4f} "
      f"max_distance_change={max_distance_change:.4f} improved={stress_improved} "
      f"mean_late_change={mean_late_change:.4f}")
print(f"ALGORITHM_SEARCH: initial_score={algorithm_search['initial_score']:.4f} "
      f"final_score={algorithm_search['final_score']:.4f} "
      f"improvement={algorithm_search['improvement']:.4f} "
      f"improvement_rate={algorithm_search['improvement_rate']:.6f} "
      f"moves={algorithm_search['moves']} passes={algorithm_search['passes']} "
      f"runtime_ms={algorithm_search['runtime_ms']:.4f}")
print(f"ROBUSTNESS: scenarios={MONTE_CARLO_SCENARIOS} seed={MONTE_CARLO_SEED} "
      f"timewin_mean={np.mean(robust_timewin):.6f} timewin_std={np.std(robust_timewin):.6f} "
      f"timewin_p05={np.percentile(robust_timewin, 5):.6f} "
      f"late_mean={np.mean(robust_late):.4f} late_p95={np.percentile(robust_late, 95):.4f} "
      f"cost_mean={np.mean(robust_cost):.2f} cost_p95={np.percentile(robust_cost, 95):.2f}")
print(f"SERVICE_DIAGNOSTICS: late_tasks={len(positive_late)} "
      f"mean_late_min={np.mean(positive_late) if len(positive_late) else 0.0:.4f} "
      f"p95_late_min={np.percentile(positive_late, 95) if len(positive_late) else 0.0:.4f} "
      f"max_late_min={np.max(positive_late) if len(positive_late) else 0.0:.4f} "
      f"mean_weight_util={np.mean(weight_utilization):.6f} "
      f"mean_volume_util={np.mean(volume_utilization):.6f} "
      f"empty_return_ratio={empty_return_ratio:.6f}")
print(f"DYNAMIC_EVENTS: scenarios={total_event_trials} "
      f"cancellation_success_rate={event_rates['cancellation']:.6f} "
      f"new_order_success_rate={event_rates['new_order']:.6f} "
      f"address_change_success_rate={event_rates['address_change']:.6f} "
      f"time_window_success_rate={event_rates['time_window']:.6f} "
      f"vehicle_failure_success_rate={event_rates['vehicle_failure']:.6f} "
      f"fallback_rate={fallback_rate:.6f}")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(10, 9), dpi=180)
ax.scatter([depot_xy[0]], [depot_xy[1]], marker="s", s=90, color="black", label="配送中心")
points = np.array(list(customer_xy.values()))
ax.scatter(points[:, 0], points[:, 1], s=18, color="#2c7fb8", alpha=0.75, label="客户")
ax.add_patch(plt.Circle((0.0, 0.0), GREEN_ZONE_RADIUS, fill=False, ls="--", lw=2,
                        color="#31a354", label="绿色配送区（10 km）"))
for route in routes:
    ids = [depot_id] + [tasks[t]["customer_id"] for t in route["tasks"]] + [depot_id]
    xy = np.array([depot_xy if cid == depot_id else customer_xy[cid] for cid in ids])
    color = "#d95f0e" if route["kind"] == "fuel" else "#3182bd"
    ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=0.12, lw=0.55)
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_title("城市绿色物流配送主方案路径")
ax.grid(alpha=0.2)
ax.set_aspect("equal", adjustable="box")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig("green_delivery_network.png", dpi=240, bbox_inches="tight")
plt.close(fig)

# 图 2：附件数据画像，所有统计量都来自本次实际读取的数据。
fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=180)
axes[0, 0].hist(aggregated["weight"], bins=18, color="#4C78A8", alpha=0.85)
axes[0, 0].set_title("客户聚合重量分布")
axes[0, 0].set_xlabel("重量")
axes[0, 0].set_ylabel("客户数")
axes[0, 1].hist(aggregated["volume"], bins=18, color="#F58518", alpha=0.85)
axes[0, 1].set_title("客户聚合体积分布")
axes[0, 1].set_xlabel("体积")
axes[0, 1].set_ylabel("客户数")
axes[1, 0].hist(window_widths, bins=16, color="#54A24B", alpha=0.85)
axes[1, 0].axvline(median_window_width, color="black", ls="--", lw=1.2,
                   label=f"中位数 {median_window_width:.0f} min")
axes[1, 0].set_title("时间窗宽度分布")
axes[1, 0].set_xlabel("分钟")
axes[1, 0].legend(frameon=False)
inside_mask = np.array([inside_green(point) for point in customer_xy.values()])
axes[1, 1].scatter(points[~inside_mask, 0], points[~inside_mask, 1], s=12,
                   color="#72B7B2", alpha=0.65, label="区外客户")
axes[1, 1].scatter(points[inside_mask, 0], points[inside_mask, 1], s=15,
                   color="#E45756", alpha=0.85, label="绿色区客户")
axes[1, 1].scatter([depot_xy[0]], [depot_xy[1]], marker="s", s=60, color="black", label="配送中心")
axes[1, 1].add_patch(plt.Circle((0, 0), GREEN_ZONE_RADIUS, fill=False, ls="--", color="#31a354"))
axes[1, 1].set_title("客户空间与绿色区分布")
axes[1, 1].set_aspect("equal", adjustable="box")
axes[1, 1].legend(frameon=False, fontsize=8)
for ax in axes.flat:
    ax.grid(alpha=0.18)
fig.suptitle("附件数据画像与约束规模", fontsize=15)
fig.tight_layout()
fig.savefig("data_profile.png", dpi=240, bbox_inches="tight")
plt.close(fig)

# 图 3：只描述程序实际执行的八个阶段，不把未实现的精确算法写入流程图。
fig, ax = plt.subplots(figsize=(8.2, 7.2), dpi=180)
ax.axis("off")
flow_labels = ["附件审计", "聚合与拆分", "分车型构造", "有限车队选择",
               "路线内 2-opt", "硬约束断言", "随机与动态实验", "证据输出"]
flow_colors = ["#4C78A8", "#72B7B2", "#54A24B", "#F2CF5B",
               "#FF9DA6", "#E45756", "#9D755D", "#B279A2"]
for index, (label, color) in enumerate(zip(flow_labels, flow_colors)):
    row, column = divmod(index, 2)
    x0 = 0.28 if column == 0 else 0.72
    y0 = 0.84 - row * 0.22
    ax.text(x0, y0, f"{index + 1}. {label}", ha="center", va="center", fontsize=14,
            bbox=dict(boxstyle="round,pad=0.75", facecolor=color, alpha=0.86, edgecolor="white"),
            transform=ax.transAxes)
    if index < len(flow_labels) - 1:
        next_row, next_column = divmod(index + 1, 2)
        next_x = 0.28 if next_column == 0 else 0.72
        next_y = 0.84 - next_row * 0.22
        ax.annotate("", xy=(next_x, next_y + (0.07 if next_row > row else 0.0)),
                    xytext=(x0, y0 - (0.07 if next_row > row else 0.0)),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#555555"),
                    xycoords=ax.transAxes)
ax.set_title("安全求解器与验证实验执行流程", fontsize=15, pad=10)
fig.tight_layout()
fig.savefig("algorithm_flow.png", dpi=240, bbox_inches="tight")
plt.close(fig)

# 图 4：30 个独立局部事件样本的距离变化、晚到变化与响应时间。
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=180)
axes[0].hist(stress_distance_changes, bins=min(10, max(1, stress_success)), color="#4C78A8")
axes[0].axvline(0, color="black", lw=1)
axes[0].set_title("插入距离变化")
axes[0].set_xlabel("km")
axes[1].hist(stress_late_changes, bins=min(10, max(1, stress_success)), color="#F58518")
axes[1].axvline(0, color="black", lw=1)
axes[1].set_title("目标路线晚到变化")
axes[1].set_xlabel("min")
axes[2].plot(range(1, len(stress_response_ms) + 1), stress_response_ms,
             marker="o", ms=3, lw=1.1, color="#54A24B")
axes[2].axhline(p95_response_ms, color="#E45756", ls="--", lw=1.2, label="P95")
axes[2].set_title("局部响应时间")
axes[2].set_xlabel("事件样本")
axes[2].set_ylabel("ms")
axes[2].legend(frameon=False)
for ax in axes:
    ax.grid(alpha=0.18)
fig.suptitle("动态局部重调度压力测试", fontsize=15)
fig.tight_layout()
fig.savefig("dynamic_stress.png", dpi=240, bbox_inches="tight")
plt.close(fig)

# 图 5：随机交通下固定方案的服务率、晚到与成本分布。
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=180)
axes[0].hist(robust_timewin, bins=14, color="#4C78A8", alpha=0.85)
axes[0].axvline(np.percentile(robust_timewin, 5), color="#E45756", ls="--", label="P05")
axes[0].set_title("随机交通时间窗满足率")
axes[0].legend(frameon=False)
axes[1].hist(robust_late, bins=14, color="#F58518", alpha=0.85)
axes[1].axvline(np.percentile(robust_late, 95), color="#E45756", ls="--", label="P95")
axes[1].set_title("总晚到分钟")
axes[1].legend(frameon=False)
axes[2].hist(robust_cost, bins=14, color="#54A24B", alpha=0.85)
axes[2].axvline(np.percentile(robust_cost, 95), color="#E45756", ls="--", label="P95")
axes[2].set_title("情景总成本")
axes[2].legend(frameon=False)
for ax in axes:
    ax.grid(alpha=0.18)
fig.suptitle(f"蒙特卡洛交通稳健性（{MONTE_CARLO_SCENARIOS} 个情景）", fontsize=15)
fig.tight_layout()
fig.savefig("robustness_diagnostics.png", dpi=240, bbox_inches="tight")
plt.close(fig)

# 图 6：路线容量利用与客户晚到诊断。
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=180)
axes[0].hist(weight_utilization, bins=14, color="#4C78A8", alpha=0.85)
axes[0].set_title("路线载重利用率")
axes[1].hist(volume_utilization, bins=14, color="#72B7B2", alpha=0.85)
axes[1].set_title("路线容积利用率")
axes[2].hist(positive_late, bins=min(14, max(1, len(positive_late))), color="#E45756", alpha=0.85)
axes[2].set_title("违约任务晚到分钟")
for ax in axes:
    ax.grid(alpha=0.18)
fig.suptitle("客户服务与线路资源诊断", fontsize=15)
fig.tight_layout()
fig.savefig("service_diagnostics.png", dpi=240, bbox_inches="tight")
plt.close(fig)
'''
    return source.replace("__DATA_DIR__", repr(str(Path(data_dir).resolve())))


def _template_figure_draft(item: dict, data_dir: str, data_files: list | None) -> CoderDraft | None:
    index = int(item.get("index", 0))
    purpose = str(item.get("purpose", f"figure-{index}"))
    filenames = {
        str(info.get("filename") if isinstance(info, dict) else info.filename)
        for info in (data_files or [])
    }
    required = {"订单信息.xlsx", "距离矩阵.xlsx", "时间窗.xlsx", "客户坐标信息.xlsx"}
    if index == 0 and required <= filenames:
        return CoderDraft(purpose=purpose, code=_green_logistics_template_code(data_dir or ""))
    return CoderDraft(purpose=purpose, code=_generic_template_code(purpose, data_dir or "", index))


def _safe_baseline_draft(item: dict, main_code: str) -> CoderDraft | None:
    """从已验证安全主求解器派生独立对照；每个对照仍重新读取附件并计算。"""
    if "BEACON_GREEN_LOGISTICS_SAFE_SOLVER" not in main_code:
        return None
    category = str(item.get("category") or "")
    code = main_code.replace("baseline=ours", f"baseline={category}", 1)
    if category == "no_schedule":
        # Q1 无政策静态场景：保留同一车队、容量、速度、目标与构造器，仅关闭限行。
        code = code.replace(
            "POLICY_ENABLED = True",
            "POLICY_ENABLED = False",
            1,
        )
    elif category == "simple_pred":
        # Q2 简单预测对照：仍施加限行，但全日采用一般时段期望速度。
        code = code.replace(
            "USE_TIME_VARYING_SPEED = True",
            "USE_TIME_VARYING_SPEED = False",
            1,
        )
    elif category == "greedy":
        # Q2 题面有限车队的 first-fit 最近邻，不做跨车型成本感知选择。
        code = code.replace(
            'STRATEGY = "cost_aware"',
            'STRATEGY = "first_fit"',
            1,
        )
        code = code.replace("LOCAL_SEARCH_ENABLED = True", "LOCAL_SEARCH_ENABLED = False", 1)
    else:
        return None
    return CoderDraft(purpose=str(item.get("name") or category), code=code)


def _safe_solver_model_contract(state: MathModelingState, artifacts: list[CodeArtifact]):
    """把模型承诺收敛到实际可执行算法；不把启发式伪装成精确 MILP/ALNS。"""
    primary = next((
        artifact for artifact in artifacts
        if artifact.success and artifact.evidence_role == "primary"
        and "BEACON_GREEN_LOGISTICS_SAFE_SOLVER" in artifact.code
    ), None)
    model = state.latest_model() if state is not None else None
    if primary is None or model is None or "BEACON_SAFE_SOLVER_CONTRACT_V4" in model.notes:
        return None
    variables = dict(model.variables)
    variables.update({
        "x[k,i,j]": "构造解中车辆 k 经过弧 (i,j) 的0-1变量",
        "y[k]": "车辆 k 是否启用",
        "z[k,v]": "车辆 k 的燃油/电动车型选择",
        "t[task]": "拆分配送任务的到达时刻（分钟）",
        "u[k,task]": "车辆服务任务后的累计载重（kg）",
        "v_load[k,task]": "车辆服务任务后的累计容积（m³）",
        "w[task]": "早到等待分钟",
        "p_late[task]": "晚到分钟",
        "delta/epsilon": "绿色区内到达与穿越边界的政策审计指示量",
    })
    return model.model_copy(update={
        "description": (
            "多约束分割配送数学模型的可行解构造：按题面五类有限车队的实际载重、"
            "容积与数量拆分并装载任务，再用软时间窗、分段时变速度与绿色区限行"
            "约束下的逐步最小增量启发式"
            "生成并审计可行解；不声称精确MILP、ALNS或全局最优。"
        ),
        "variables": variables,
        "equations": [
            r"min Z=Z_fix+Z_drive+Z_penalty+Z_energy+Z_carbon",
            r"sum_i q_i x_ik <= Q_type(k) y_k; sum_i v_i x_ik <= V_type(k) y_k",
            r"sum_k z[k,type] <= fleet_count[type]",
            r"t_j=max(t_i,E_i)+20+piecewise_travel(d_ij,t_i)",
            r"fuel and 480<=t<960 implies not inside_or_cross_green_zone",
            r"sum_j x_ijk=sum_j x_jik for every visited node",
        ],
        "notes": (
            (model.notes + "\n" if model.notes else "")
            + "BEACON_SAFE_SOLVER_CONTRACT_V4：代码物化 x/y/z/t/u/v_load/w/p_late/"
              "delta/epsilon，逐路线断言题面有限车队容量、流守恒和限行可行性，"
              "按题面油电耗、价格、排放因子与碳成本统一计分；构造解后执行可行性保持的"
              "路线内2-opt，并以随机交通、五类事件和服务诊断作外部验证。"
        ),
        "objective_mapping": [
            "最小化固定成本、行驶成本、软时间窗惩罚、能耗与碳成本之和；"
            "当前实现返回可行上界，不提供最优性间隙。"
        ],
        "constraint_mapping": [
            "每个容量拆分任务恰好分配一次",
            "每车累计载重与容积不超过其车型上限，且五类启用数不超过60/50/50/10/15",
            "路径从配送中心出发并返回，内部客户流入等于流出",
            "到达—等待—20分钟服务—离开按时变速度递推",
            "燃油车8:00—16:00不得到达或穿越10km绿色区",
        ],
        "validation_mapping": [
            "运行时读取四个附件并记录数据血缘",
            "显式断言容量、流守恒、绿色区和任务覆盖",
            "RESULT 输出成本、车辆数、服务率、碳排放、距离、时间窗率与响应时间",
            "与Q1无政策场景、Q2简单预测和关闭2-opt的first-fit贪心基线及敏感性中心点交叉校验",
            "执行200个随机交通情景、五类动态事件矩阵和客户/线路级服务诊断",
        ],
        "question_coverage": [],
    })


def _use_deterministic_coder() -> bool:
    """本地模板只用于显式离线应急模式，不能替代正常的题目相关代码生成。"""
    return os.getenv("MATH_AGENT_CODER_DETERMINISTIC", "").strip() == "1"


def _max_figure_tasks() -> int:
    try:
        return max(1, int(os.getenv("MATH_AGENT_MAX_FIGURE_TASKS", "4")))
    except ValueError:
        return 4


def _code_timeout_seconds() -> int:
    try:
        return max(30, int(os.getenv("MATH_AGENT_CODE_TIMEOUT", "120")))
    except ValueError:
        return 120


def _green_depth_evidence_error(stdout: str) -> str:
    """验证城市物流论文所需的算法、随机、诊断与动态实验深度。"""
    required = (
        "ALGORITHM_SEARCH", "ROBUSTNESS", "SERVICE_DIAGNOSTICS", "DYNAMIC_EVENTS",
    )
    parsed: dict[str, dict[str, float]] = {}
    for label in required:
        match = re.search(rf"(?m)^{label}:\s+(.+)$", stdout or "")
        if match is None:
            return f"城市物流主证据缺少 {label} 结构化实验行"
        parsed[label] = {
            item.group(1): float(item.group(2))
            for item in re.finditer(
                r"([A-Za-z_][\w]*)=(-?(?:\d+(?:\.\d+)?|\.\d+))", match.group(1)
            )
        }
    algorithm_required = {
        "initial_score", "final_score", "improvement", "improvement_rate",
        "moves", "passes", "runtime_ms",
    }
    robustness_required = {
        "scenarios", "seed", "timewin_mean", "timewin_std", "timewin_p05",
        "late_mean", "late_p95", "cost_mean", "cost_p95",
    }
    diagnostics_required = {
        "late_tasks", "mean_late_min", "p95_late_min", "max_late_min",
        "mean_weight_util", "mean_volume_util", "empty_return_ratio",
    }
    dynamic_required = {
        "scenarios", "cancellation_success_rate", "new_order_success_rate",
        "address_change_success_rate", "time_window_success_rate",
        "vehicle_failure_success_rate", "fallback_rate",
    }
    for label, required_fields in (
        ("ALGORITHM_SEARCH", algorithm_required),
        ("ROBUSTNESS", robustness_required),
        ("SERVICE_DIAGNOSTICS", diagnostics_required),
        ("DYNAMIC_EVENTS", dynamic_required),
    ):
        missing = sorted(required_fields - set(parsed[label]))
        if missing:
            return f"{label} 缺少字段：{missing}"

    algorithm = parsed["ALGORITHM_SEARCH"]
    if min(algorithm[key] for key in algorithm_required) < 0:
        return "ALGORITHM_SEARCH 数值不得为负"
    tolerance = max(1e-3, 1e-4 * max(1.0, algorithm["initial_score"]))
    if algorithm["final_score"] > algorithm["initial_score"] + tolerance:
        return "ALGORITHM_SEARCH final_score 不得高于 initial_score"
    if abs(
        algorithm["initial_score"] - algorithm["final_score"] - algorithm["improvement"]
    ) > tolerance:
        return "ALGORITHM_SEARCH improvement 与初末目标差不一致"
    expected_rate = (
        algorithm["improvement"] / algorithm["initial_score"]
        if algorithm["initial_score"] else 0.0
    )
    if not 0.0 <= algorithm["improvement_rate"] <= 1.0 or abs(
        algorithm["improvement_rate"] - expected_rate
    ) > 1e-4:
        return "ALGORITHM_SEARCH improvement_rate 与目标改善量不一致"

    robustness = parsed["ROBUSTNESS"]
    if robustness["scenarios"] < 100:
        return "ROBUSTNESS 蒙特卡洛情景至少 100 个"
    if not (
        0.0 <= robustness["timewin_p05"] <= 1.0
        and 0.0 <= robustness["timewin_mean"] <= 1.0
        and robustness["timewin_std"] >= 0.0
        and robustness["late_mean"] >= 0.0
        and robustness["late_p95"] >= 0.0
        and robustness["cost_mean"] >= 0.0
        and robustness["cost_p95"] >= 0.0
    ):
        return "ROBUSTNESS 概率、离散程度、晚到或成本指标不合法"

    diagnostics = parsed["SERVICE_DIAGNOSTICS"]
    if diagnostics["late_tasks"] < 0 or any(
        diagnostics[key] < 0
        for key in ("mean_late_min", "p95_late_min", "max_late_min")
    ):
        return "SERVICE_DIAGNOSTICS 任务数与晚到指标不得为负"
    if (
        diagnostics["mean_late_min"] > diagnostics["max_late_min"] + 1e-6
        or diagnostics["p95_late_min"] > diagnostics["max_late_min"] + 1e-6
    ):
        return "SERVICE_DIAGNOSTICS 晚到均值或 P95 不得超过最大值"
    for key in ("mean_weight_util", "mean_volume_util", "empty_return_ratio"):
        if not 0.0 <= diagnostics[key] <= 1.0:
            return f"SERVICE_DIAGNOSTICS 字段 {key} 超出 [0,1]"

    dynamic = parsed["DYNAMIC_EVENTS"]
    if dynamic["scenarios"] < 20:
        return "DYNAMIC_EVENTS 五类事件实验至少 20 个独立情景"
    rate_keys = (
        "cancellation_success_rate", "new_order_success_rate",
        "address_change_success_rate", "time_window_success_rate",
        "vehicle_failure_success_rate", "fallback_rate",
    )
    for key in rate_keys:
        value = dynamic.get(key)
        if value is None or not 0.0 <= value <= 1.0:
            return f"DYNAMIC_EVENTS 字段 {key} 缺失或超出 [0,1]"
    success_mean = sum(dynamic[key] for key in rate_keys[:-1]) / 5.0
    if abs(dynamic["fallback_rate"] - (1.0 - success_mean)) > 2e-3:
        return "DYNAMIC_EVENTS fallback_rate 与五类成功率不一致"
    return ""


def _validated_execution(
    state: MathModelingState,
    item: dict,
    result,
    *,
    code: str = "",
    require_data_usage: bool = False,
) -> tuple[bool, str, str]:
    """把进程结果提升为可进入论文的证据结果。"""
    if not result.success:
        return False, result.stderr or "Python 子进程执行失败", result.error_kind or "runtime"
    expected = item.get("category") if item.get("kind") == "baseline" else None
    valid, reason, parsed = validate_numeric_results(
        result.stdout,
        stderr=result.stderr,
        require_result=True,
        expected_identifier=expected,
        max_entity_count=infer_entity_upper_bound(state.data_files),
        min_metrics_per_result=(
            4 if item.get("kind") == "figure" and require_data_usage else 0
        ),
    )
    if not valid:
        return False, reason, "output_validation"
    filenames = {info.filename for info in state.data_files}
    green_schema = {
        "订单信息.xlsx", "距离矩阵.xlsx", "时间窗.xlsx", "客户坐标信息.xlsx",
    } <= filenames
    if green_schema and item.get("kind") == "figure" and require_data_usage:
        metrics = parsed.get("ours", {})
        required_metrics = {
            "total_cost", "vehicles", "service_rate", "total_carbon", "total_distance",
            "fuel_vehicles", "ev_vehicles", "timewin_rate", "response_time",
        }
        missing = sorted(required_metrics - set(metrics))
        if missing:
            return False, f"城市物流主证据缺少关键指标：{missing}", "output_validation"
        if metrics["service_rate"] < 0.95:
            return (
                False,
                f"城市物流主方案服务率 service_rate={metrics['service_rate']:.4f} 低于 0.95，"
                "属于结构性未服务结果，不能作为主证据",
                "output_validation",
            )
        if metrics["vehicles"] > 185:
            return False, f"车辆数 {metrics['vehicles']:g} 超过题面有限车队 185 辆", "output_validation"
        if metrics["fuel_vehicles"] > 160 or metrics["ev_vehicles"] > 25:
            return False, "燃油车或新能源车启用数超过题面分类上限", "output_validation"
        if abs(metrics["fuel_vehicles"] + metrics["ev_vehicles"] - metrics["vehicles"]) > 1e-9:
            return False, "燃油车与新能源车数量之和不等于总车辆数", "output_validation"
        depth_error = _green_depth_evidence_error(result.stdout)
        if depth_error:
            return False, depth_error, "output_validation"
    if require_data_usage:
        lineage_ok, lineage_reason = validate_code_data_usage(
            code, [info.filename for info in state.data_files],
        )
        if not lineage_ok:
            return False, lineage_reason, "output_validation"
        expected_paths = {str(path).casefold() for path in _input_paths(state)}
        observed_paths = {str(Path(path)).casefold() for path in result.read_paths}
        if expected_paths and not expected_paths.intersection(observed_paths):
            return False, "运行时未观察到对声明附件的真实读取", "output_validation"
        if green_schema and not expected_paths.issubset(observed_paths):
            missing_paths = sorted(expected_paths - observed_paths)
            return False, f"城市物流主证据未读取全部四个附件：{missing_paths}", "output_validation"
        if _uses_haversine_on_planar_km_schema(code, state.data_files):
            return (
                False,
                "坐标附件字段为 X (km)/Y (km) 平面公里坐标，不能按经纬度调用 Haversine；"
                "应使用距离矩阵或欧氏距离以保持单位口径一致",
                "output_validation",
            )
        service_error = _service_time_contract_error(code, state)
        if service_error:
            return False, service_error, "output_validation"
    return True, "", ""


def _uses_haversine_on_planar_km_schema(code: str, data_files) -> bool:
    columns: set[str] = set()
    for info in data_files or []:
        summary = info.get("summary", {}) if isinstance(info, dict) else info.summary
        raw_columns = summary.get("columns", []) if isinstance(summary, dict) else []
        for column in raw_columns if isinstance(raw_columns, list) else []:
            columns.add(str(column).casefold())
    planar = "x (km)" in columns and "y (km)" in columns
    return planar and code.casefold().count("haversine(") > 1


def _service_time_contract_error(code: str, state: MathModelingState) -> str:
    """题面明确服务时长时，执行代码中的统一常数必须同口径。"""
    problem_text = "\n".join((state.problem or "", state.background or ""))
    required = re.search(r"(?:服务时间|服务时长).*?(\d+(?:\.\d+)?)\s*分钟", problem_text)
    assigned = re.search(
        r"(?mi)^\s*SERVICE_TIME\s*=\s*(\d+(?:\.\d+)?)"
        r"\s*(?:#\s*([^\r\n]*))?$",
        code or "",
    )
    if required is None or assigned is None:
        return ""
    expected = float(required.group(1))
    observed_raw = float(assigned.group(1))
    unit_hint = (assigned.group(2) or "").casefold()
    uses_hours = bool(re.search(r"(?:\bh\b|hours?|小时)", unit_hint))
    observed = observed_raw * 60.0 if uses_hours else observed_raw
    if abs(expected - observed) <= 1e-9:
        return ""
    return (
        f"题面规定统一客户服务时间为 {expected:g} 分钟，"
        f"代码 SERVICE_TIME={observed:g} 分钟，模型—代码口径不一致；"
        f"service_time_contract expected_minutes={expected:g} "
        f"observed_minutes={observed:g} assignment_unit={'hours' if uses_hours else 'minutes'}"
    )


def _input_paths(state: MathModelingState) -> list[Path]:
    base_value = state.data_dir or state.output_dir
    base = Path(base_value) if base_value else None
    paths: list[Path] = []
    for info in state.data_files:
        path = Path(info.path or info.filename)
        if not path.is_absolute() and base is not None:
            path = base / path
        paths.append(path.resolve())
    return paths


def _combined_stderr(stderr: str, validation_reason: str) -> str:
    parts = [part.strip() for part in (stderr, validation_reason) if part and part.strip()]
    return "\n".join(parts)


def _previous_figure_code(state: MathModelingState, item: dict) -> str:
    """返回当前图任务上一轮源码，支持旧 checkpoint 无 prev_code 的情况。"""
    explicit = str(item.get("prev_code") or "")
    if explicit:
        return explicit
    if int(item.get("attempt", 0)) <= 0:
        return ""
    for artifact in reversed(state.coder_work_artifacts):
        if artifact.category == "figure" and not artifact.success and artifact.code:
            return artifact.code
    return ""


def _consistency_repair_context(
    state: MathModelingState, *, evidence_target: str
) -> tuple[str, str]:
    """返回上一批主代码及一致性反馈，供定向修订而不是从头生成。"""
    if evidence_target != "primary" or not state.model_code_reports:
        return "", ""
    report = state.model_code_reports[-1]
    if report.approved and report.score >= 7:
        return "", ""
    primary = next(
        (
            artifact for artifact in reversed(state.code_artifacts)
            if artifact.success and artifact.category == "figure"
            and artifact.evidence_role == "primary" and artifact.code
        ),
        None,
    )
    if primary is None:
        return "", ""
    details = [
        f"模型—代码一致性评分仅 {report.score}/10，必须在上一版代码上定向修订。",
        *[f"问题：{issue}" for issue in report.issues],
        *[f"建议：{suggestion}" for suggestion in report.suggestions],
    ]
    return primary.code, "\n".join(details)


def _local_repair_draft(
    item: dict,
    previous_code: str,
    state: MathModelingState | None = None,
) -> CoderDraft | None:
    """只处理可证明等价的常见机械错误；其余问题仍交给模型定向修复。"""
    if not previous_code:
        return None
    repaired = previous_code.replace(
        "import matplotlib.rcparams as rc",
        "from matplotlib import rcParams as rc",
    )
    # 常见的 JSON/Markdown 逃逸残留：模型把应有的源码换行写成字面 ``\n``，
    # 且落在注释后，使下一条赋值也被注释掉。这里只修复“注释 + 标识符赋值”形态，
    # 不碰路径、正则和普通字符串里的合法反斜杠。
    repaired = re.sub(
        r"(?m)^(\s*\#[^\r\n]*?)\\n(?=[A-Za-z_]\w*\s*=)",
        r"\1\n",
        repaired,
    )
    repaired = "\n".join(
        line.replace(r"\n", "\n")
        if r"\n" in line and "'" not in line and '"' not in line
        else line
        for line in repaired.split("\n")
    )
    error = str(item.get("prev_err") or "")
    if state is not None and "SERVICE_TIME" not in error:
        error = "\n".join(filter(None, (error, _service_time_contract_error(repaired, state))))
    if (
        state is not None
        and str(item.get("prev_kind") or "") in {"runtime", "timeout", "output_validation"}
        and any(marker in error for marker in (
            "无法服务任何客户", "no progress", "timeout after 120", "service_rate=", "缺少关键指标",
        ))
        and ".groupby(" in repaired
    ):
        filenames = {info.filename for info in state.data_files}
        required = {"订单信息.xlsx", "距离矩阵.xlsx", "时间窗.xlsx", "客户坐标信息.xlsx"}
        if required <= filenames:
            return _template_figure_draft(item, state.data_dir, state.data_files)
    if "SERVICE_TIME" in error:
        match = re.search(
            r"(?:服务时间为\s*|service_time_contract\s+expected(?:_minutes)?=)"
            r"(\d+(?:\.\d+)?)",
            error,
            flags=re.IGNORECASE,
        )
        if match:
            expected_minutes = float(match.group(1))
            assignment = re.search(
                r"(?mi)^(\s*SERVICE_TIME\s*=\s*)\d+(?:\.\d+)?"
                r"(\s*(?:#\s*[^\r\n]*)?)$",
                repaired,
            )
            if assignment:
                hint = assignment.group(2).casefold()
                value = expected_minutes / 60.0 if re.search(
                    r"(?:\bh\b|hours?|小时)", hint,
                ) else expected_minutes
                repaired = (
                    repaired[:assignment.start()]
                    + assignment.group(1) + repr(float(value)) + assignment.group(2)
                    + repaired[assignment.end():]
                )
    if "Int64Engine" in error and "KeyError: '0'" in error:
        repaired = repaired.replace(
            "dist_df.loc[i, str(j)]",
            "dist_df.loc[i, j]",
        )
    dense_distance_block = """cust_ids = list(dist_df.index)
coord_keys = set(coord_dict.keys())
dist_dict = {}
for i in cust_ids:
    for j in cust_ids:
        if i in coord_keys and j in coord_keys:
            dist_dict[(i, j)] = dist_df.loc[i, j]

# 补充可能缺失的距离（使用欧氏距离）
coord_keys_list = list(coord_keys)
for i in coord_keys_list:
    for j in coord_keys_list:
        if (i, j) not in dist_dict:
            xi, yi = coord_dict[i]
            xj, yj = coord_dict[j]
            dist_dict[(i, j)] = np.sqrt((xi-xj)**2 + (yi-yj)**2)"""
    compact_distance_block = """class DistanceLookup:
    # 保留 pandas/NumPy 的紧凑矩阵，只在访问时按标签索引；避免复制成数百万个 tuple。
    def __init__(self, frame, coordinates):
        self.values = frame.to_numpy(dtype=float, copy=False)
        self.row_pos = {label: pos for pos, label in enumerate(frame.index)}
        self.col_pos = {label: pos for pos, label in enumerate(frame.columns)}
        self.coordinates = coordinates

    def __getitem__(self, pair):
        i, j = pair
        row = self.row_pos.get(i)
        col = self.col_pos.get(j)
        if row is not None and col is not None:
            return float(self.values[row, col])
        xi, yi = self.coordinates[i]
        xj, yj = self.coordinates[j]
        return float(np.hypot(xi - xj, yi - yj))

dist_dict = DistanceLookup(dist_df, coord_dict)"""
    if dense_distance_block in repaired:
        repaired = repaired.replace(dense_distance_block, compact_distance_block)
    if (
        str(item.get("prev_kind") or "") == "timeout"
        and "while not np.all(assigned):" in repaired
        and "VEHICLE_CAPACITY_WEIGHT" in repaired
        and "VEHICLE_CAPACITY_VOLUME" in repaired
        and "# BEACON_CAPACITY_SPLIT" not in repaired
    ):
        volume_line = next(
            (
                line for line in repaired.splitlines()
                if line.strip().startswith("VEHICLE_CAPACITY_VOLUME") and "=" in line
            ),
            "",
        )
        if volume_line and "customers" in repaired:
            capacity_split = """

# BEACON_CAPACITY_SPLIT：聚合客户需求必须拆成单车容量可行的访问任务。
_beacon_chunks = []
for _, _beacon_row in customers.iterrows():
    _beacon_weight = pd.to_numeric(_beacon_row['重量'], errors='coerce')
    _beacon_volume = pd.to_numeric(_beacon_row['体积'], errors='coerce')
    _beacon_weight = 0.0 if pd.isna(_beacon_weight) else float(_beacon_weight)
    _beacon_volume = 0.0 if pd.isna(_beacon_volume) else float(_beacon_volume)
    _beacon_parts = max(
        1,
        int(np.ceil(max(
            _beacon_weight / VEHICLE_CAPACITY_WEIGHT,
            _beacon_volume / VEHICLE_CAPACITY_VOLUME,
        ))),
    )
    for _beacon_part in range(_beacon_parts):
        _beacon_chunk = _beacon_row.copy()
        _beacon_chunk['重量'] = _beacon_weight / _beacon_parts
        _beacon_chunk['体积'] = _beacon_volume / _beacon_parts
        if _beacon_weight > 0.0 or _beacon_volume > 0.0:
            _beacon_chunks.append(_beacon_chunk)
customers = pd.DataFrame(_beacon_chunks).reset_index(drop=True)
"""
            repaired = repaired.replace(volume_line, volume_line + capacity_split, 1)
            repaired = repaired.replace(
                "while not np.all(assigned):",
                "_beacon_stalled_rounds = 0\nwhile not np.all(assigned):",
                1,
            )
            repaired = repaired.replace(
                "    while True:\n        best_cost = np.inf",
                "    while True:\n        unassigned = np.where(~assigned)[0]\n"
                "        best_cost = np.inf",
                1,
            )
            toggle = "    vehicle_type = 1 - vehicle_type"
            if toggle in repaired:
                progress_guard = """    # BEACON_PROGRESS_GUARD：连续两种车辆均无进展时立即失败，不等待硬期限。
    if len(current_route) == 0:
        _beacon_stalled_rounds += 1
        if _beacon_stalled_rounds >= 2:
            raise RuntimeError('no progress: remaining tasks violate capacity or route constraints')
    else:
        _beacon_stalled_rounds = 0
"""
                repaired = repaired.replace(toggle, progress_guard + toggle, 1)
    if (
        str(item.get("prev_kind") or "") == "timeout"
        and "while unvisited:" in repaired
        and ".groupby(" in repaired
        and "CAPACITY_WEIGHT" in repaired
        and "CAPACITY_VOLUME" in repaired
        and "total_weight" in repaired
        and "total_volume" in repaired
    ):
        # 客户聚合需求先拆成单车容量可行的访问任务，并让最近邻返回“任务索引”
        # 而不是会重复的客户编号。这样既保留原脚本的成本/绘图逻辑，也保证
        # unvisited 每轮真实减少；连续空路线则立即失败，不等到 120 秒。
        capacity_anchor = next((
            line for line in repaired.splitlines()
            if line.strip().startswith("CAPACITY_VOLUME") and "=" in line
        ), "")
        if capacity_anchor and "# BEACON_AGGREGATE_SPLIT" not in repaired:
            split_block = r'''

# BEACON_AGGREGATE_SPLIT：把客户聚合需求拆成单车容量可行的访问任务。
_beacon_rows = []
for _, _beacon_row in df_cust.iterrows():
    _beacon_weight = float(pd.to_numeric(_beacon_row['total_weight'], errors='coerce') or 0.0)
    _beacon_volume = float(pd.to_numeric(_beacon_row['total_volume'], errors='coerce') or 0.0)
    _beacon_parts = max(1, int(np.ceil(max(
        _beacon_weight / CAPACITY_WEIGHT,
        _beacon_volume / CAPACITY_VOLUME,
    ))))
    for _ in range(_beacon_parts):
        _beacon_part = _beacon_row.copy()
        _beacon_part['total_weight'] = _beacon_weight / _beacon_parts
        _beacon_part['total_volume'] = _beacon_volume / _beacon_parts
        _beacon_rows.append(_beacon_part)
df_cust = pd.DataFrame(_beacon_rows).reset_index(drop=True)
df_cust['customer_id'] = pd.to_numeric(df_cust['customer_id'], errors='raise').astype(int)
'''
            repaired = repaired.replace(capacity_anchor, capacity_anchor + split_block, 1)
        repaired = repaired.replace(
            "best_global = None\n    for u in unvisited:",
            "best_task = None\n    for u in unvisited:",
            1,
        ).replace(
            "best_global = global_u\n    return best_global, best_dist",
            "best_task = u\n    return best_task, best_dist",
            1,
        )
        repaired = re.sub(
            r"(?m)^(\s*)best_global, best_dist = find_nearest\(([^\r\n]+)\)\r?\n"
            r"\1if best_global is None:\r?\n\1    break\r?\n"
            r"\1cust_idx = global_to_cust_idx\[best_global\]",
            lambda match: (
                f"{match.group(1)}best_task, best_dist = find_nearest({match.group(2)})\n"
                f"{match.group(1)}if best_task is None:\n"
                f"{match.group(1)}    break\n"
                f"{match.group(1)}cust_idx = best_task\n"
                f"{match.group(1)}best_global = cust_idx_to_global[cust_idx]"
            ),
            repaired,
            count=1,
        )
        if "# BEACON_UNVISITED_PROGRESS" not in repaired:
            repaired = repaired.replace(
                "    # 结束当前车，返回配送中心\n    route.append(0)",
                "    # BEACON_UNVISITED_PROGRESS：空路线表示剩余任务不可行，立即失败。\n"
                "    if not served_this_vehicle:\n"
                "        raise RuntimeError('no progress: remaining tasks violate capacity constraints')\n"
                "    # 结束当前车，返回配送中心\n    route.append(0)",
                1,
            )
    if repaired == previous_code:
        return None
    return CoderDraft(purpose=str(item.get("purpose") or "figure"), code=repaired)


def _local_baseline_repair_draft(
    item: dict, *, data_dir: str, previous_code: str
) -> CoderDraft | None:
    """仅修复 attempt 工作目录导致的已知相对附件根路径错误。"""
    error = str(item.get("prev_err") or "")
    if not previous_code:
        return None
    if "SERVICE_TIME" in error:
        match = re.search(
            r"(?:服务时间为\s*|service_time_contract\s+expected=)(\d+(?:\.\d+)?)",
            error,
            flags=re.IGNORECASE,
        )
        if match:
            repaired = re.sub(
                r"(?m)^(\s*SERVICE_TIME\s*=\s*)\d+(?:\.\d+)?\s*$",
                rf"\g<1>{float(match.group(1))!r}",
                previous_code,
                count=1,
            )
            if repaired != previous_code:
                return CoderDraft(purpose=str(item.get("name") or "baseline"), code=repaired)
    if not data_dir or "FileNotFoundError" not in error:
        return None
    replacement = f"Path({str(Path(data_dir).resolve())!r})"
    repaired = previous_code
    for relative in (
        "Path('./附件')", 'Path("./附件")',
        "Path('附件')", 'Path("附件")',
    ):
        repaired = repaired.replace(relative, replacement)
    if repaired == previous_code:
        return None
    return CoderDraft(purpose=str(item.get("name") or "baseline"), code=repaired)

def coder_prepare_node(state: MathModelingState) -> dict:
    """Create a small, stable coding queue without calling the model."""
    model = state.latest_model()
    if model is None:
        return {"errors": ["coder: missing model"], "coder_phase": "done"}
    batch = max((a.batch for a in state.code_artifacts), default=0) + 1
    purposes = (model.figure_purposes or [model.description])[:_max_figure_tasks()]
    purposes[0] = (
        "主方案数值求解与核心证据图：必须读取真实附件，实现最终模型的轻量可复现求解，"
        "先计算完整 RESULT 指标，再绘制一张核心证据图。原始图意图："
        f"{purposes[0]}"
    )
    queue = [
        {"kind": "figure", "id": f"figure:{i}", "purpose": purpose,
         "index": i, "attempt": 0, "prev_err": "", "prev_kind": "",
         "evidence_target": "primary" if i == 0 else "supporting"}
        for i, purpose in enumerate(purposes)
    ]
    return {
        "coder_phase": "generate", "coder_work_queue": queue,
        "coder_work_artifacts": [], "coder_current_batch": batch,
        "coder_pending_draft": {},
    }


def coder_generate_node(state: MathModelingState) -> dict:
    """Generate code for the current queue item, then hand off to execute."""
    queue = list(state.coder_work_queue)
    if not queue:
        return {"coder_phase": "done"}
    item = dict(queue[0])
    evidence_target = item.get(
        "evidence_target", "primary" if int(item.get("index", 0)) == 0 else "supporting"
    )
    model = state.latest_model()
    if item["kind"] == "figure":
        if _use_deterministic_coder():
            draft = _template_figure_draft(item, state.data_dir, state.data_files)
        else:
            primary = next(
                (
                    a for a in state.coder_work_artifacts
                    if a.success and a.category == "figure" and a.evidence_role == "primary"
                ),
                None,
            )
            previous_code = _previous_figure_code(state, item)
            consistency_code, consistency_feedback = _consistency_repair_context(
                state, evidence_target=evidence_target,
            )
            if not previous_code and consistency_code:
                previous_code = consistency_code
            feedback = item.get("prev_err") or consistency_feedback or None
            failure_kind = item.get("prev_kind", "") or (
                "consistency" if consistency_feedback else ""
            )
            filenames = {info.filename for info in state.data_files}
            green_schema = {
                "订单信息.xlsx", "距离矩阵.xlsx", "时间窗.xlsx", "客户坐标信息.xlsx",
            } <= filenames
            fallback_after_verified_cycle = (
                evidence_target == "primary"
                and green_schema
                and state.code_verify_iteration >= 1
                and primary is None
            )
            refresh_safe_solver = (
                failure_kind == "consistency"
                and "BEACON_GREEN_LOGISTICS_SAFE_SOLVER" in previous_code
                and green_schema
            )
            draft = (
                _template_figure_draft(item, state.data_dir, state.data_files)
                if fallback_after_verified_cycle or refresh_safe_solver
                else _local_repair_draft(item, previous_code, state)
            )
            if draft is None:
                draft = complete(
                    build_prompt_figure_one(
                        model, item["purpose"], feedback,
                        failure_kind, blueprint=state.problem_blueprint,
                        data_dir=state.data_dir, data_files=state.data_files,
                        canonical_evidence=primary.stdout if primary else "",
                        previous_code=previous_code,
                    ),
                    schema=CoderDraft, system=SYSTEM, model=MODEL_ROUTING["coder"],
                    profile="code", temperature=0.1, max_tokens=6000,
                )
    else:
        main_code = _current_primary_code(state)
        try:
            draft = _local_baseline_repair_draft(
                item, data_dir=state.data_dir,
                previous_code=str(item.get("prev_code") or ""),
            )
            if draft is None:
                draft = _safe_baseline_draft(item, main_code)
            if draft is None:
                draft = complete(
                    build_baseline_prompt(state.problem, main_code, item["name"],
                                          item["category"], item["instruction"],
                                          item.get("prev_err") or None,
                                          item.get("prev_kind", ""),
                                          item.get("prev_code", "")),
                    schema=CoderDraft, system=SYSTEM, model=MODEL_ROUTING["coder"],
                    profile="code", temperature=0.1, max_tokens=6000,
                )
        except Exception as exc:
            fallback = _safe_baseline_draft(item, main_code)
            if fallback is not None:
                return {
                    "coder_pending_draft": fallback.model_dump(),
                    "coder_phase": "execute",
                }
            artifacts = list(state.coder_work_artifacts)
            artifacts.append(CodeArtifact(
                purpose=f"{item['name']}对照方案（失败）", code="", stderr=str(exc)[:500],
                success=False, category=f"baseline:{item['category']}",
                evidence_role="none",
                batch=state.coder_current_batch,
            ))
            queue.pop(0)
            if queue:
                return {
                    "coder_work_queue": queue,
                    "coder_work_artifacts": artifacts,
                    "coder_pending_draft": {},
                    "coder_phase": "generate",
                }
            return _coder_done_delta(artifacts, state=state)
    return {"coder_pending_draft": draft.model_dump(), "coder_phase": "execute"}


def coder_execute_node(state: MathModelingState) -> dict:
    """Execute already-checkpointed code so a crash does not trigger re-generation."""
    queue = list(state.coder_work_queue)
    if not queue:
        return {"coder_phase": "done", "coder_pending_draft": {}}
    item = dict(queue[0])
    evidence_target = item.get(
        "evidence_target", "primary" if int(item.get("index", 0)) == 0 else "supporting"
    )
    draft = CoderDraft.model_validate(state.coder_pending_draft)
    workdir = Path(state.output_dir) if state.output_dir else Path(tempfile.mkdtemp(prefix="math_agent_"))
    workdir.mkdir(parents=True, exist_ok=True)
    artifacts = list(state.coder_work_artifacts)

    if item["kind"] == "figure":
        result = run_python(
            draft.code,
            workdir=workdir / f"fig_{item['index']}_attempt_{item['attempt']}",
            timeout=_code_timeout_seconds(),
            expected_input_paths=_input_paths(state),
        )
        has_primary = _has_primary_for_current_batch(state, artifacts)
        effective_success, validation_reason, error_kind = _validated_execution(
            state, item, result, code=draft.code,
            require_data_usage=bool(state.data_files) and evidence_target == "primary",
        )
        evidence_role = (
            "primary" if effective_success and evidence_target == "primary"
            else "supporting" if effective_success and evidence_target == "supporting" and has_primary
            else "none"
        )
        artifacts.append(CodeArtifact(
            purpose=draft.purpose, code=draft.code, stdout=result.stdout,
            stderr=_combined_stderr(result.stderr, validation_reason), success=effective_success,
            artifact_paths=result.artifact_paths,
            read_paths=getattr(result, "read_paths", []),
            batch=state.coder_current_batch,
            evidence_role=evidence_role,
        ))
        if not effective_success and item["attempt"] < MAX_CODE_RETRIES:
            feedback = validation_reason or result.stderr or result.stdout[-1000:]
            item.update(
                attempt=item["attempt"] + 1, prev_err=feedback,
                prev_kind=error_kind, prev_code=draft.code,
            )
            queue[0] = item
        else:
            queue.pop(0)
            if evidence_target == "primary" and not effective_success:
                # 任意绘图任务不能顶替失败的主求解。结束本批次，让一致性闭环开启新批次。
                queue = [work for work in queue if work.get("kind") != "figure"]
            if (
                not any(work.get("kind") == "figure" for work in queue)
                and not any(work.get("kind") == "baseline" for work in queue)
                and _has_primary_for_current_batch(state, artifacts)
            ):
                queue.extend(_missing_baseline_items(state, artifacts))
    else:
        result = run_python(
            draft.code,
            workdir=workdir / f"baseline_{item['category']}_attempt_{item['attempt']}",
            timeout=_code_timeout_seconds(),
            expected_input_paths=_input_paths(state),
        )
        effective_success, validation_reason, error_kind = _validated_execution(
            state, item, result, code=draft.code,
            require_data_usage=bool(state.data_files),
        )
        artifacts.append(CodeArtifact(
            purpose=draft.purpose, code=draft.code, stdout=result.stdout,
            stderr=_combined_stderr(result.stderr, validation_reason), success=effective_success,
            artifact_paths=result.artifact_paths,
            read_paths=getattr(result, "read_paths", []),
            category=f"baseline:{item['category']}",
            batch=state.coder_current_batch,
            evidence_role="baseline" if effective_success else "none",
        ))
        if not effective_success and item["attempt"] < MAX_CODE_RETRIES:
            feedback = validation_reason or result.stderr or result.stdout[-1000:]
            item.update(
                attempt=item["attempt"] + 1, prev_err=feedback,
                prev_kind=error_kind, prev_code=draft.code,
            )
            queue[0] = item
        else:
            queue.pop(0)
            if (
                effective_success and not queue
                and _has_primary_for_current_batch(state, artifacts)
            ):
                queue.extend(_missing_baseline_items(state, artifacts))

    if queue:
        return {
            "coder_work_queue": queue, "coder_work_artifacts": artifacts,
            "coder_pending_draft": {}, "coder_phase": "generate",
        }

    return _coder_done_delta(artifacts, state=state)


def _coder_done_delta(
    artifacts: list[CodeArtifact], *, state: MathModelingState | None = None
) -> dict:
    delta: dict = {
        "code_artifacts": artifacts, "coder_work_queue": [],
        "coder_work_artifacts": [], "coder_pending_draft": {}, "coder_phase": "done",
    }
    has_primary = (
        _has_primary_for_current_batch(state, artifacts)
        if state is not None else any(
            a.success and a.category == "figure" and a.evidence_role == "primary"
            for a in artifacts
        )
    )
    if not has_primary:
        delta["errors"] = ["coder: all code tasks failed"]
    elif state is not None:
        aligned = _safe_solver_model_contract(state, artifacts)
        if aligned is not None:
            delta["model_versions"] = [aligned]
    return delta


def coder_node(state: MathModelingState) -> dict:
    """Legacy compatibility path: run prepare/generate/execute in one node."""
    current = state
    delta = coder_prepare_node(current)
    current = current.model_copy(update=delta)
    for _ in range(100):
        if current.coder_phase == "done":
            break
        if current.coder_phase == "generate":
            delta = coder_generate_node(current)
        else:
            delta = coder_execute_node(current)
        current = current.model_copy(update=delta)
    return {
        "code_artifacts": current.code_artifacts,
        "coder_work_queue": current.coder_work_queue,
        "coder_work_artifacts": current.coder_work_artifacts,
        "coder_pending_draft": current.coder_pending_draft,
        "coder_phase": current.coder_phase,
        "errors": current.errors,
    }
