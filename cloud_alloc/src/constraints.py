"""Feasibility and constraint evaluation module for cloud allocation.

Constraints:
  C1: CPU, RAM, and min_instance requirements satisfied for each workload.
  C2: DB tier strictly has 0 spot instances.
  C3: Total cost f1 <= Budget (bindings at ~85% of all-on-demand cost).
  C4: Instances placed strictly in allowed regions for each workload.
  C5: Web capacity after worst-case spot reclamation (loss of all spot instances
      in the largest spot-hosting region) >= Peak_load * (1 + headroom).
  C6: Web latency (regional network latency + utilization queuing term) <= L_max.
  C7: Integer counts within quota [0, MAX_QUOTA_PER_SLOT].
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np

import config
from src.model import SystemData, compute_f1_cost


def check_feasibility(
    x: Union[np.ndarray, Dict[str, Dict[Tuple[str, int, str], int]]],
    system: SystemData,
    budget: Optional[float] = None,
    scenario_high_spot: bool = False,
) -> Tuple[bool, Dict[str, float], float]:
    """Check all constraints C1-C7 for an allocation.

    Returns:
        (is_feasible, violations_dict, total_violation)
    """
    if isinstance(x, np.ndarray):
        x_alloc = system.decode_vector(x)
        raw_vec = x
    else:
        x_alloc = x
        raw_vec = system.encode_dict(x)

    violations: Dict[str, float] = {}

    # C1: Resource requirements (CPU, RAM, min instances)
    c1_cpu_viol = 0.0
    c1_ram_viol = 0.0
    c1_inst_viol = 0.0

    for w_key, w in system.workloads.items():
        alloc = x_alloc.get(w_key, {})
        tot_cpu = sum(system.instances[t].vcpu * cnt for (t, _, _), cnt in alloc.items())
        tot_ram = sum(system.instances[t].ram_gb * cnt for (t, _, _), cnt in alloc.items())
        tot_inst = sum(cnt for cnt in alloc.values())

        if tot_cpu < w.cpu_req:
            c1_cpu_viol += (w.cpu_req - tot_cpu) / max(1.0, float(w.cpu_req))
        if tot_ram < w.ram_req:
            c1_ram_viol += (w.ram_req - tot_ram) / max(1.0, float(w.ram_req))
        if tot_inst < w.min_instances:
            c1_inst_viol += (w.min_instances - tot_inst) / max(1.0, float(w.min_instances))

    violations["C1_cpu"] = float(c1_cpu_viol)
    violations["C1_ram"] = float(c1_ram_viol)
    violations["C1_min_instances"] = float(c1_inst_viol)

    # C2: DB workload has zero spot instances
    db_spot_count = sum(
        cnt for (t, opt, r), cnt in x_alloc.get("db", {}).items() if opt == 1
    )
    violations["C2_db_zero_spot"] = float(db_spot_count)

    # C3: Budget constraint (f1 <= Budget)
    if budget is not None and budget > 0:
        cost = compute_f1_cost(system, x_alloc, scenario_high_spot)
        cost_over = max(0.0, cost - budget)
        violations["C3_budget"] = float(cost_over / budget)
    else:
        violations["C3_budget"] = 0.0

    # C4: Instances only in allowed regions
    c4_viol = 0.0
    for w_key, alloc in x_alloc.items():
        allowed = set(system.workloads[w_key].allowed_regions)
        for (_, _, reg), cnt in alloc.items():
            if reg not in allowed and cnt > 0:
                c4_viol += float(cnt)
    violations["C4_allowed_regions"] = float(c4_viol)

    # C5: Web resilience under worst-case single-region spot reclaim
    web_alloc = x_alloc.get("web", {})
    web_w = system.workloads["web"]
    total_web_vcpu = sum(
        system.instances[t].vcpu * cnt for (t, _, _), cnt in web_alloc.items()
    )

    spot_vcpu_by_region: Dict[str, float] = {r: 0.0 for r in config.ALL_REGIONS}
    for (t, opt, reg), cnt in web_alloc.items():
        if opt == 1:
            spot_vcpu_by_region[reg] = spot_vcpu_by_region.get(reg, 0.0) + (
                system.instances[t].vcpu * cnt
            )

    worst_spot_loss = max(spot_vcpu_by_region.values()) if spot_vcpu_by_region else 0.0
    remaining_capacity = total_web_vcpu - worst_spot_loss
    resilience_target = web_w.peak_load * (1.0 + web_w.headroom)

    if remaining_capacity < resilience_target:
        violations["C5_web_spot_reclaim"] = float(
            (resilience_target - remaining_capacity) / resilience_target
        )
    else:
        violations["C5_web_spot_reclaim"] = 0.0

    # C6: Web latency penalty <= L_max
    if total_web_vcpu <= 0:
        violations["C6_web_latency"] = 10.0
    else:
        reg_vcpu: Dict[str, float] = {}
        for (t, _, reg), cnt in web_alloc.items():
            reg_vcpu[reg] = reg_vcpu.get(reg, 0.0) + system.instances[t].vcpu * cnt

        base_lat = sum(
            reg_vcpu[r] * web_w.base_latency_by_region.get(r, 60.0) for r in reg_vcpu
        ) / total_web_vcpu

        utilization = web_w.peak_load / total_web_vcpu
        if utilization >= 0.98:
            queueing_latency = 40.0 * (utilization / 0.02)
        else:
            queueing_latency = 15.0 * (utilization / (1.0 - utilization))

        est_latency = base_lat + queueing_latency
        if est_latency > web_w.latency_target_ms:
            violations["C6_web_latency"] = float(
                (est_latency - web_w.latency_target_ms) / web_w.latency_target_ms
            )
        else:
            violations["C6_web_latency"] = 0.0

    # C7: Counts within quota [0, MAX_QUOTA_PER_SLOT]
    quota_viol = 0.0
    for val in raw_vec:
        if val < 0:
            quota_viol += abs(val)
        elif val > config.MAX_QUOTA_PER_SLOT:
            quota_viol += val - config.MAX_QUOTA_PER_SLOT
    violations["C7_quota"] = float(quota_viol)

    total_violation = float(sum(violations.values()))
    is_feasible = total_violation <= 1e-5

    return is_feasible, violations, total_violation


def get_constraint_vector(
    x: np.ndarray,
    system: SystemData,
    budget: Optional[float] = None,
    scenario_high_spot: bool = False,
) -> np.ndarray:
    """Return constraints in G <= 0 pymoo format."""
    _, violations, _ = check_feasibility(x, system, budget, scenario_high_spot)
    # Each violation v >= 0 is represented as g = v <= 0 (feasible when <= 0)
    return np.array(list(violations.values()), dtype=np.float64)
