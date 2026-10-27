"""Monte Carlo simulation module for validating representative allocation plans.

Used solely for post-optimization validation (not inside the search loop).
Draws spot interruptions via Poisson processes and price path variations across N trials.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np

import config
from src.model import SystemData, compute_f1_cost


def simulate(
    x: Union[np.ndarray, Dict[str, Dict[Tuple[str, int, str], int]]],
    system: SystemData,
    n_simulations: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    """Perform Monte Carlo simulation over spot interruptions and price paths.

    Returns:
        Dict containing simulated availability, unavailability, makespan, data loss, and cost.
    """
    rng = np.random.RandomState(seed)
    if isinstance(x, np.ndarray):
        x_alloc = system.decode_vector(x)
    else:
        x_alloc = x

    # 1. Web Simulation
    web_alloc = x_alloc.get("web", {})
    web_w = system.workloads["web"]
    n_req = web_w.min_instances

    web_instances_p: list[float] = []
    for (t, opt, r), cnt in web_alloc.items():
        if opt == 0:
            p_up = config.AVAILABILITY_ON_DEMAND
        else:
            ir = system.get_interruption_rate(t, r)
            p_up = 1.0 - (ir * config.MEAN_OUTAGE_FRACTION)
        web_instances_p.extend([p_up] * cnt)

    total_web_instances = len(web_instances_p)
    if total_web_instances < n_req:
        sim_unavail = 1.0
        sim_avail = 0.0
    else:
        p_vec = np.array(web_instances_p)  # shape: (n_inst,)
        # Draw Bernoulli trials: shape (n_simulations, n_inst)
        draws = rng.rand(n_simulations, total_web_instances) < p_vec
        num_up = np.sum(draws, axis=1)
        up_trials = np.sum(num_up >= n_req)
        sim_avail = float(up_trials / n_simulations)
        sim_unavail = float(1.0 - sim_avail)

    # 2. Batch Makespan & Data Loss Simulation
    batch_alloc = x_alloc.get("batch", {})
    batch_w = system.workloads["batch"]
    total_work = batch_w.batch_total_work

    base_batch_vcpu = 0
    spot_batch_items: list[Tuple[int, float]] = []  # (vcpu, ir)
    for (t, opt, r), cnt in batch_alloc.items():
        vcpu = system.instances[t].vcpu * cnt
        if opt == 0:
            base_batch_vcpu += vcpu
        else:
            ir = system.get_interruption_rate(t, r)
            spot_batch_items.append((vcpu, ir))

    makespan_trials = []
    data_loss_trials = []

    # All spot instances across web and batch for data loss
    all_spot_items: list[Tuple[int, float]] = list(spot_batch_items)
    for (t, opt, r), cnt in web_alloc.items():
        if opt == 1:
            ir = system.get_interruption_rate(t, r)
            all_spot_items.append((system.instances[t].vcpu * cnt, ir))

    for _ in range(n_simulations):
        # Batch rework calculation
        rework_hours_lost = 0.0
        trial_data_loss = 0.0

        for vcpu, ir in spot_batch_items:
            # Number of interruptions in the month drawn from Poisson distribution
            n_interruptions = rng.poisson(lam=ir)
            # Each interruption costs on average checkpoint_interval / 2 hours of lost work
            rework_hours_lost += (
                n_interruptions * (config.FIXED_CHECKPOINT_INTERVAL_HOURS / 2.0) * vcpu
            )

        # Throughput
        effective_hours = max(
            1.0,
            (base_batch_vcpu + sum(v for v, _ in spot_batch_items)) * config.MONTH_HOURS
            - rework_hours_lost,
        )
        effective_throughput = effective_hours / config.MONTH_HOURS
        makespan_trials.append(total_work / effective_throughput)

        # Data Loss calculation
        for vcpu, ir in all_spot_items:
            n_int = rng.poisson(lam=ir)
            unsaved_state = (
                (config.FIXED_CHECKPOINT_INTERVAL_HOURS / 2.0)
                * config.STATE_GEN_RATE_GB_PER_VCPU_HR
                * vcpu
            )
            trial_data_loss += n_int * unsaved_state

        data_loss_trials.append(trial_data_loss)

    sim_makespan = float(np.mean(makespan_trials)) if makespan_trials else 9999.0
    sim_data_loss = float(np.mean(data_loss_trials)) if data_loss_trials else 0.0

    # 3. Cost with price variations
    base_cost = compute_f1_cost(system, x_alloc, scenario_high_spot=False)
    # Add small stochastic volatility noise (+/- 2%)
    cost_noise = rng.normal(0.0, 0.015, size=n_simulations)
    sim_cost = float(base_cost * (1.0 + np.mean(cost_noise)))

    return {
        "cost": sim_cost,
        "availability_pct": sim_avail * 100.0,
        "unavailability": sim_unavail,
        "makespan_hours": sim_makespan,
        "data_loss_gb": sim_data_loss,
    }
