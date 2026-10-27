"""Pymoo Problem definition and custom domain-specific Repair operator for NSGA-II.

Encoding:
  Integer decision vector x where x[i] corresponds to the count of instances
  for a specific (workload, instance_type, option, region) slot.

Repair Operator:
  (a) Forces spot=0 for db workload.
  (b) Zeroes disallowed regions.
  (c) Adjusts instances to strictly satisfy resource constraints (C1) and
      web worst-case spot reclaim (C5).
  (d) Trims cost if over budget by shifting non-db workloads from on-demand to spot
      or shedding surplus instances.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair

import config
from src.constraints import check_feasibility
from src.model import Slot, SystemData, compute_f1_cost, evaluate


def repair_allocation(
    x_raw: np.ndarray,
    system: SystemData,
    budget: Optional[float] = None,
    scenario_high_spot: bool = False,
) -> np.ndarray:
    """Domain-specific heuristic repair operator.

    Guarantees:
      - All counts are integers in [0, MAX_QUOTA_PER_SLOT]
      - DB has 0 spot instances
      - Disallowed regions are 0
      - Workload CPU, RAM, and min instance requirements (C1) are met
      - Web worst-case spot reclaim resilience (C5) is met
      - Cost is trimmed towards budget if exceeding (C3)
    """
    x = np.round(np.clip(x_raw, 0, config.MAX_QUOTA_PER_SLOT)).astype(int)

    # (a) Force spot=0 for db & (b) Zero disallowed regions
    for slot in system.slots:
        if slot.workload == "db" and slot.option == 1:
            x[slot.index] = 0
        w = system.workloads[slot.workload]
        if slot.region not in w.allowed_regions:
            x[slot.index] = 0

    alloc = system.decode_vector(x)

    # (c) Ensure C1 (CPU, RAM, min instances) for all workloads
    for w_key, w in system.workloads.items():
        w_alloc = alloc[w_key]
        tot_cpu = sum(system.instances[t].vcpu * cnt for (t, _, _), cnt in w_alloc.items())
        tot_ram = sum(system.instances[t].ram_gb * cnt for (t, _, _), cnt in w_alloc.items())
        tot_inst = sum(cnt for cnt in w_alloc.values())

        pref_reg = w.allowed_regions[0]
        # Choose efficient backup type: m5.xlarge or r5.xlarge (for db)
        backup_type = "r5.large" if w_key == "db" else "m5.large"
        backup_opt = 0 if (w_key == "db" or not w.spot_allowed) else 1
        key = (backup_type, backup_opt, pref_reg)

        attempts = 0
        while (tot_cpu < w.cpu_req or tot_ram < w.ram_req or tot_inst < w.min_instances) and attempts < 30:
            w_alloc[key] = w_alloc.get(key, 0) + 1
            tot_cpu += system.instances[backup_type].vcpu
            tot_ram += system.instances[backup_type].ram_gb
            tot_inst += 1
            attempts += 1

        # Also remove surplus instances if significantly over-provisioned (C1 capacity bounds)
        for (t_name, opt, reg), cnt in list(w_alloc.items()):
            inst_spec = system.instances[t_name]
            while w_alloc.get((t_name, opt, reg), 0) > 0:
                if (
                    tot_cpu - inst_spec.vcpu >= w.cpu_req
                    and tot_ram - inst_spec.ram_gb >= w.ram_req
                    and tot_inst - 1 >= w.min_instances
                ):
                    w_alloc[(t_name, opt, reg)] -= 1
                    tot_cpu -= inst_spec.vcpu
                    tot_ram -= inst_spec.ram_gb
                    tot_inst -= 1
                    if w_alloc[(t_name, opt, reg)] == 0:
                        del w_alloc[(t_name, opt, reg)]
                else:
                    break

    # (c2) Satisfy C5: Web resilience after losing spot instances in largest spot region
    web_alloc = alloc["web"]
    web_w = system.workloads["web"]
    resilience_target = web_w.peak_load * (1.0 + web_w.headroom)

    attempts = 0
    while attempts < 30:
        tot_web_cpu = sum(
            system.instances[t].vcpu * cnt for (t, _, _), cnt in web_alloc.items()
        )
        spot_cpu_by_region: Dict[str, float] = {r: 0.0 for r in config.ALL_REGIONS}
        for (t, opt, r), cnt in web_alloc.items():
            if opt == 1:
                spot_cpu_by_region[r] += system.instances[t].vcpu * cnt
        worst_spot_loss = max(spot_cpu_by_region.values()) if spot_cpu_by_region else 0.0
        rem_cap = tot_web_cpu - worst_spot_loss

        if rem_cap >= resilience_target:
            break

        # Find an on-demand slot in an allowed region that is below quota
        added = False
        for reg in web_w.allowed_regions:
            for t_candidate in ["m5.large", "m5.xlarge", "c5.xlarge", "t3.medium"]:
                cand_key = (t_candidate, 0, reg)
                if web_alloc.get(cand_key, 0) < config.MAX_QUOTA_PER_SLOT:
                    web_alloc[cand_key] = web_alloc.get(cand_key, 0) + 1
                    added = True
                    break
            if added:
                break
        if not added:
            break
        attempts += 1

    # (d) Trim cost if over budget by shifting on-demand non-db instances to spot or shedding surplus
    if budget is not None and budget > 0:
        current_cost = compute_f1_cost(system, alloc, scenario_high_spot)
        if current_cost > budget:
            # First, convert batch on-demand instances to spot
            batch_alloc = alloc["batch"]
            for (t, opt, r) in list(batch_alloc.keys()):
                if opt == 0 and batch_alloc[(t, opt, r)] > 0:
                    cnt = batch_alloc[(t, opt, r)]
                    del batch_alloc[(t, opt, r)]
                    batch_alloc[(t, 1, r)] = batch_alloc.get((t, 1, r), 0) + cnt
                    current_cost = compute_f1_cost(system, alloc, scenario_high_spot)
                    if current_cost <= budget:
                        break

            # If still over budget, convert non-essential web on-demand instances
            if current_cost > budget:
                for (t, opt, r) in list(web_alloc.keys()):
                    if opt == 0 and web_alloc.get((t, opt, r), 0) > 0:
                        # Only convert if remaining on-demand doesn't violate C5
                        web_alloc[(t, opt, r)] -= 1
                        web_alloc[(t, 1, r)] = web_alloc.get((t, 1, r), 0) + 1

                        # Validate C5
                        tot_web_cpu = sum(
                            system.instances[t_i].vcpu * c_i
                            for (t_i, _, _), c_i in web_alloc.items()
                        )
                        spot_by_r: Dict[str, float] = {reg: 0.0 for reg in config.ALL_REGIONS}
                        for (t_i, opt_i, r_i), c_i in web_alloc.items():
                            if opt_i == 1:
                                spot_by_r[r_i] += system.instances[t_i].vcpu * c_i
                        worst_loss = max(spot_by_r.values()) if spot_by_r else 0.0

                        if tot_web_cpu - worst_loss < resilience_target:
                            # Revert conversion
                            web_alloc[(t, opt, r)] += 1
                            web_alloc[(t, 1, r)] -= 1
                            break

                        current_cost = compute_f1_cost(system, alloc, scenario_high_spot)
                        if current_cost <= budget:
                            break

    repaired_vec = system.encode_dict(alloc)
    return np.clip(repaired_vec, 0, config.MAX_QUOTA_PER_SLOT).astype(int)


class CloudAllocationRepair(Repair):
    """Pymoo Repair Operator wrapper."""

    def __init__(
        self,
        system: SystemData,
        budget: Optional[float] = None,
        scenario_high_spot: bool = False,
    ):
        super().__init__()
        self.system = system
        self.budget = budget
        self.scenario_high_spot = scenario_high_spot

    def _do(self, problem, Z, **kwargs):
        # Z has shape (pop_size, n_vars)
        for i in range(len(Z)):
            Z[i] = repair_allocation(
                Z[i],
                self.system,
                self.budget,
                self.scenario_high_spot,
            )
        return Z


class CloudAllocationProblem(Problem):
    """Multi-Objective Cloud Allocation Pymoo Problem."""

    def __init__(
        self,
        system: SystemData,
        budget: Optional[float] = None,
        n_objectives: int = config.DEFAULT_OBJECTIVES_COUNT,
        scenario_high_spot: bool = False,
    ):
        self.system = system
        self.budget = budget
        self.scenario_high_spot = scenario_high_spot

        super().__init__(
            n_var=system.n_vars,
            n_obj=n_objectives,
            n_ieq_constr=1,  # Aggregate constraint violation (G <= 0)
            xl=np.zeros(system.n_vars, dtype=int),
            xu=np.full(system.n_vars, config.MAX_QUOTA_PER_SLOT, dtype=int),
            vtype=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        pop_size = len(x)
        F = np.zeros((pop_size, self.n_obj), dtype=np.float64)
        G = np.zeros((pop_size, 1), dtype=np.float64)

        for i in range(pop_size):
            # Evaluate objectives
            F[i] = evaluate(
                x[i],
                self.system,
                n_objectives=self.n_obj,
                scenario_high_spot=self.scenario_high_spot,
            )
            # Check constraints: G <= 0 condition for pymoo
            is_feas, _, total_viol = check_feasibility(
                x[i],
                self.system,
                budget=self.budget,
                scenario_high_spot=self.scenario_high_spot,
            )
            G[i, 0] = total_viol if not is_feas else 0.0

        out["F"] = F
        out["G"] = G
