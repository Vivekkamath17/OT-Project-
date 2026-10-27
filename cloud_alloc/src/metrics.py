"""Metrics and post-processing module:
  - Hypervolume (pymoo HV with documented normalized reference point)
  - Pareto front spacing metric
  - Knee-point (Balanced) and representative plan selection (Cost-Saver, Balanced, Reliability-First)
  - Dominance checks vs Baselines B1, B2, B3
  - Plan comparison tables
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

import config
from src.constraints import check_feasibility
from src.model import SystemData, evaluate


def calculate_hypervolume(
    F: np.ndarray,
    ref_point: Optional[np.ndarray] = None,
    ideal_point: Optional[np.ndarray] = None,
    nadir_point: Optional[np.ndarray] = None,
) -> float:
    """Calculate normalized hypervolume for a Pareto front.

    To ensure fairness across objectives with different physical scales
    (Cost in $1000s, Unavailability in [0, 0.05], Makespan in [200, 720] hrs,
    Data loss in [0, 50] GB), objectives are min-max normalized using
    a fixed bounding box [ideal, nadir], with reference point at [1.1, 1.1, 1.1, 1.1].
    """
    if len(F) == 0:
        return 0.0

    n_obj = F.shape[1]
    if ideal_point is None:
        ideal_point = np.zeros(n_obj)
    if nadir_point is None:
        # Default nadir bounding box across objectives
        if n_obj == 4:
            nadir_point = np.array([5000.0, 0.05, 800.0, 50.0])
        else:
            nadir_point = np.array([5000.0, 0.05, 800.0])

    denom = np.maximum(nadir_point - ideal_point, 1e-6)
    F_norm = (F - ideal_point) / denom
    # Clip to avoid negatives or values beyond nadir
    F_norm = np.clip(F_norm, 0.0, 1.0)

    # Filter out points that do not strictly dominate the reference point
    norm_ref = np.full(n_obj, 1.1)
    valid_mask = np.all(F_norm <= norm_ref, axis=1)
    if not np.any(valid_mask):
        return 0.0

    hv_calc = HV(ref_point=norm_ref)
    return float(hv_calc(F_norm[valid_mask]))


def calculate_spacing(F: np.ndarray) -> float:
    """Schott's spacing metric measuring uniformity of solution distribution."""
    if len(F) <= 1:
        return 0.0

    # Pairwise Manhattan distances between adjacent solutions
    dists = []
    n = len(F)
    for i in range(n):
        diffs = np.abs(F - F[i])
        manhattan = np.sum(diffs, axis=1)
        # Exclude self
        manhattan[i] = np.inf
        dists.append(np.min(manhattan))

    d_mean = np.mean(dists)
    spacing = np.sqrt(np.sum((dists - d_mean) ** 2) / float(n))
    return float(spacing)


def check_dominance(candidate_F: np.ndarray, baseline_F: np.ndarray) -> bool:
    """Return True if candidate_F strictly dominates baseline_F (all minimized)."""
    return bool(np.all(candidate_F <= baseline_F) and np.any(candidate_F < baseline_F))


def count_dominating_solutions(
    front_F: np.ndarray,
    baseline_F: np.ndarray,
) -> int:
    """Count how many solutions on the front dominate the baseline point."""
    count = 0
    for sol in front_F:
        if check_dominance(sol, baseline_F):
            count += 1
    return count


def select_representative_plans(
    X_pareto: np.ndarray,
    F_pareto: np.ndarray,
    system: SystemData,
    budget: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Select three representative allocation plans from the Pareto front:
      1. Cost-Saver: Lowest cost among feasible plans
      2. Balanced: Knee-point maximizing orthogonal distance to the extreme chord
      3. Reliability-First: Lowest unavailability within budget

    Returns:
        Dict: {"Cost-Saver": (x, F), "Balanced": (x, F), "Reliability-First": (x, F)}
    """
    if len(X_pareto) == 0:
        raise ValueError("Pareto front is empty; cannot select representative plans.")

    # 1. Cost-Saver: Min f1 (cost)
    idx_cost = int(np.argmin(F_pareto[:, 0]))
    cost_saver = (X_pareto[idx_cost], F_pareto[idx_cost])

    # 2. Reliability-First: Min f2 (unavailability)
    idx_rel = int(np.argmin(F_pareto[:, 1]))
    reliability_first = (X_pareto[idx_rel], F_pareto[idx_rel])

    # 3. Balanced (Knee Point): Max perpendicular distance to the chord joining
    # Cost-Saver and Reliability-First in normalized objective space
    if len(X_pareto) <= 2:
        balanced = (X_pareto[0], F_pareto[0])
    else:
        # Min-max normalize objectives across current front
        f_min = np.min(F_pareto, axis=0)
        f_max = np.max(F_pareto, axis=0)
        denom = np.maximum(f_max - f_min, 1e-6)
        F_norm = (F_pareto - f_min) / denom

        p1 = F_norm[idx_cost]
        p2 = F_norm[idx_rel]
        chord_vec = p2 - p1
        chord_len = np.linalg.norm(chord_vec)

        if chord_len < 1e-6:
            idx_knee = 0
        else:
            chord_unit = chord_vec / chord_len
            # Vector from p1 to each point
            v = F_norm - p1
            # Projection onto chord
            proj = np.outer(np.dot(v, chord_unit), chord_unit)
            # Orthogonal distance
            dist = np.linalg.norm(v - proj, axis=1)
            idx_knee = int(np.argmax(dist))

        balanced = (X_pareto[idx_knee], F_pareto[idx_knee])

    return {
        "Cost-Saver": cost_saver,
        "Balanced": balanced,
        "Reliability-First": reliability_first,
    }


def format_plan_breakdown(
    x: np.ndarray,
    F: np.ndarray,
    system: SystemData,
    plan_name: str,
    b1_cost: float,
) -> Dict[str, Any]:
    """Generate a clean dictionary breakdown for a representative plan."""
    alloc = system.decode_vector(x)
    saving_pct = ((b1_cost - F[0]) / b1_cost) * 100.0 if b1_cost > 0 else 0.0

    workload_summaries = {}
    for w_key in system.workloads.keys():
        w_alloc = alloc.get(w_key, {})
        tot_inst = sum(w_alloc.values())
        tot_vcpu = sum(system.instances[t].vcpu * cnt for (t, _, _), cnt in w_alloc.items())
        tot_ram = sum(system.instances[t].ram_gb * cnt for (t, _, _), cnt in w_alloc.items())
        spot_vcpu = sum(
            system.instances[t].vcpu * cnt for (t, opt, _), cnt in w_alloc.items() if opt == 1
        )
        spot_fraction = (spot_vcpu / max(1, tot_vcpu)) * 100.0
        active_regions = sorted(list({reg for (_, _, reg), cnt in w_alloc.items() if cnt > 0}))

        instances_str = ", ".join(
            f"{cnt}x {t} ({'spot' if opt == 1 else 'od'}, {reg})"
            for (t, opt, reg), cnt in sorted(w_alloc.items())
            if cnt > 0
        )

        workload_summaries[w_key] = {
            "instances_count": tot_inst,
            "vcpu": tot_vcpu,
            "ram_gb": tot_ram,
            "spot_fraction_pct": round(spot_fraction, 1),
            "regions": "/".join(active_regions),
            "instances_detail": instances_str,
        }

    availability_pct = (1.0 - F[1]) * 100.0
    data_loss = F[3] if len(F) >= 4 else 0.0

    return {
        "Plan": plan_name,
        "Cost ($/mo)": round(float(F[0]), 2),
        "Cost Saving vs B1 (%)": round(float(saving_pct), 1),
        "Web Availability (%)": round(float(availability_pct), 4),
        "Web Unavailability": round(float(F[1]), 6),
        "Batch Makespan (hrs)": round(float(F[2]), 1),
        "Data Loss Risk (GB/mo)": round(float(data_loss), 2),
        "Workloads": workload_summaries,
    }
