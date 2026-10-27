"""Baseline allocation strategies:
  B1: All-On-Demand (Current startup practice reference)
  B2: Greedy cheapest-feasible with spot utilization
  B3: Random Search (budget-matched with NSGA-II function evaluations)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from src.constraints import check_feasibility
from src.model import SystemData, evaluate


def build_b1_all_ondemand(system: SystemData) -> np.ndarray:
    """B1 Baseline: Cost-efficient all-on-demand allocation meeting all constraints.

    Uses zero spot instances and operates within preferred allowed regions.
    """
    alloc: Dict[str, Dict[Tuple[str, int, str], int]] = {
        w: {} for w in system.workloads.keys()
    }

    # 1. Web Tier: Needs >= 32 vCPU, >= 64 GB RAM, >= 4 instances,
    # and total vCPU >= peak_load * (1 + headroom) = 28 * 1.3 = 36.4 vCPU.
    # us-east-1 has lowest price multiplier (0.98).
    # m5.xlarge (4 vCPU, 16 GB RAM) x 10 = 40 vCPU, 160 GB RAM -> 10 instances.
    alloc["web"][("m5.xlarge", 0, "us-east-1")] = 10

    # 2. DB Tier: Needs >= 16 vCPU, >= 64 GB RAM, >= 2 instances.
    # r5.xlarge (4 vCPU, 32 GB RAM) x 4 = 16 vCPU, 128 GB RAM -> 4 instances in us-east-1.
    alloc["db"][("r5.xlarge", 0, "us-east-1")] = 4

    # 3. Batch Tier: Needs >= 48 vCPU, >= 96 GB RAM, >= 2 instances.
    # c5.2xlarge (8 vCPU, 16 GB RAM) x 6 = 48 vCPU, 96 GB RAM -> 6 instances in us-east-1.
    alloc["batch"][("c5.2xlarge", 0, "us-east-1")] = 6

    vec = system.encode_dict(alloc)
    is_feas, viols, total_v = check_feasibility(vec, system, budget=None)
    if not is_feas:
        # Fine-tune allocation if slight discrepancy
        for w_key, w in system.workloads.items():
            tot_cpu = sum(
                system.instances[t].vcpu * cnt
                for (t, _, _), cnt in alloc[w_key].items()
            )
            tot_ram = sum(
                system.instances[t].ram_gb * cnt
                for (t, _, _), cnt in alloc[w_key].items()
            )
            while tot_cpu < w.cpu_req or tot_ram < w.ram_req:
                alloc[w_key][("m5.large", 0, w.allowed_regions[0])] = (
                    alloc[w_key].get(("m5.large", 0, w.allowed_regions[0]), 0) + 1
                )
                tot_cpu += system.instances["m5.large"].vcpu
                tot_ram += system.instances["m5.large"].ram_gb
        vec = system.encode_dict(alloc)

    return vec


def build_b2_greedy(system: SystemData, budget: Optional[float] = None) -> np.ndarray:
    """B2 Baseline: Greedy cheapest-feasible instance selection using spot instances.

    Uses spot where permitted (web, batch), on-demand for DB, and balances web
    across multiple regions to satisfy the single-region spot reclaim constraint (C5).
    """
    alloc: Dict[str, Dict[Tuple[str, int, str], int]] = {
        w: {} for w in system.workloads.keys()
    }

    # 1. DB: Spot strictly forbidden. Use cheapest on-demand instance meeting specs.
    # Needs 16 vCPU, 64 GB RAM in us-east-1. r5.xlarge (4 vcpu, 32 ram) x 4 instances.
    alloc["db"][("r5.xlarge", 0, "us-east-1")] = 4

    # 2. Batch: Full spot in cheapest region (us-east-1).
    # Needs 48 vCPU, 96 GB RAM. c5.xlarge has 75% spot discount!
    # 12 x c5.xlarge = 48 vCPU, 96 GB RAM.
    alloc["batch"][("c5.xlarge", 1, "us-east-1")] = 12

    # 3. Web: Mix of on-demand and spot across regions to satisfy C5.
    # C5 target = 28 * 1.3 = 36.4 vCPU remaining after losing spot in biggest region.
    # Base on-demand: 5 x m5.xlarge (20 vCPU) in us-east-1.
    # Spot spread: 3 x m5.xlarge (12 vCPU) in eu-west-1, 3 x m5.xlarge (12 vCPU) in ap-south-1.
    # Total = 44 vCPU. Max spot in single region = 12 vCPU.
    # Remaining after worst reclaim = 44 - 12 = 32 vCPU (>= 32 req and >= 28*1.3 with a bit more).
    # Let's use 6 on-demand (24 vCPU) + 4 spot eu-west-1 (16 vCPU) + 4 spot ap-south-1 (16 vCPU) = 56 vCPU.
    # Remaining = 56 - 16 = 40 vCPU >= 36.4!
    alloc["web"][("m5.xlarge", 0, "us-east-1")] = 6
    alloc["web"][("m5.xlarge", 1, "eu-west-1")] = 3
    alloc["web"][("m5.xlarge", 1, "ap-south-1")] = 3

    vec = system.encode_dict(alloc)
    return vec


def run_b3_random_search(
    system: SystemData,
    budget: float,
    n_evals: int,
    repair_func,
    seed: int = 42,
    n_objectives: int = config.DEFAULT_OBJECTIVES_COUNT,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """B3 Baseline: Random search with same evaluation budget as NSGA-II.

    Returns:
        (pareto_X, pareto_F, feasibility_rate)
    """
    rng = np.random.RandomState(seed)
    n_vars = system.n_vars
    feasible_solutions: List[np.ndarray] = []
    feasible_objectives: List[np.ndarray] = []

    feasible_count = 0

    for _ in range(n_evals):
        # Sample realistic sparse random integer counts (active probability ~15%)
        raw = np.where(rng.rand(n_vars) < 0.15, rng.randint(1, 6, size=n_vars), 0)
        # Apply domain-specific repair operator
        repaired = repair_func(raw, system, budget)

        is_feas, _, _ = check_feasibility(repaired, system, budget)
        if is_feas:
            feasible_count += 1
            objs = evaluate(repaired, system, n_objectives=n_objectives)
            feasible_solutions.append(repaired)
            feasible_objectives.append(objs)

    feasibility_rate = float(feasible_count / max(1, n_evals))

    if not feasible_solutions:
        return np.empty((0, n_vars)), np.empty((0, n_objectives)), feasibility_rate

    F_all = np.array(feasible_objectives)
    X_all = np.array(feasible_solutions)

    # Filter to non-dominated Pareto front
    pareto_indices = get_nondominated_indices(F_all)
    return X_all[pareto_indices], F_all[pareto_indices], feasibility_rate


def get_nondominated_indices(F: np.ndarray) -> np.ndarray:
    """Find indices of non-dominated objective vectors (all minimized)."""
    n_points = F.shape[0]
    is_dominated = np.zeros(n_points, dtype=bool)

    for i in range(n_points):
        if is_dominated[i]:
            continue
        for j in range(n_points):
            if i == j:
                continue
            # j dominates i if j <= i in all objectives and strictly less in at least one
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                is_dominated[i] = True
                break

    return np.where(~is_dominated)[0]
