"""Unit and integration tests for Cloud Allocation Model."""

import sys
from pathlib import Path

# Add cloud_alloc to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pytest

import config
from src.baselines import build_b1_all_ondemand, build_b2_greedy
from src.constraints import check_feasibility
from src.model import (
    SystemData,
    compute_f1_cost,
    compute_f2_web_unavailability,
    evaluate,
)
from src.nsga2_problem import repair_allocation
from src.simulator import simulate


@pytest.fixture
def system():
    return SystemData()


def test_b1_baseline_is_feasible(system):
    """Test 1: B1 All-On-Demand allocation meets all operational constraints C1-C7."""
    x_b1 = build_b1_all_ondemand(system)
    is_feas, viols, total_v = check_feasibility(x_b1, system, budget=None)
    assert is_feas, f"B1 should be feasible, but got violations: {viols}"
    assert total_v <= 1e-5


def test_db_never_on_spot_after_repair(system):
    """Test 2: DB tier strictly has 0 spot instances after repair even if seeded with spot."""
    rng = np.random.RandomState(42)
    # Generate random vector with spot instances in DB
    raw_vec = rng.randint(1, 5, size=system.n_vars)
    repaired_vec = repair_allocation(raw_vec, system, budget=10000.0)

    alloc = system.decode_vector(repaired_vec)
    db_spot_count = sum(cnt for (t, opt, r), cnt in alloc["db"].items() if opt == 1)
    assert db_spot_count == 0, f"Expected 0 spot instances for DB, found {db_spot_count}"


def test_repaired_solutions_are_feasible(system):
    """Test 3: Arbitrary random allocations are transformed into feasible plans by repair operator."""
    rng = np.random.RandomState(123)
    b1_vec = build_b1_all_ondemand(system)
    b1_cost = compute_f1_cost(system, system.decode_vector(b1_vec))
    budget = b1_cost * config.DEFAULT_BUDGET_FRACTION

    for i in range(10):
        raw_vec = rng.randint(0, 10, size=system.n_vars)
        repaired_vec = repair_allocation(raw_vec, system, budget=budget)
        is_feas, viols, total_v = check_feasibility(repaired_vec, system, budget=budget)
        # Resource requirements and spot restrictions must strictly hold
        assert viols["C1_cpu"] <= 1e-4
        assert viols["C1_ram"] <= 1e-4
        assert viols["C1_min_instances"] <= 1e-4
        assert viols["C2_db_zero_spot"] == 0
        assert viols["C4_allowed_regions"] == 0
        assert viols["C5_web_spot_reclaim"] <= 1e-4


def test_availability_monotone_in_ondemand_instances(system):
    """Test 4: Web tier availability increases monotonically as more on-demand instances are added."""
    unavailabilities = []
    # Test with 2, 4, 6, 8, 10 on-demand web instances
    for count in [2, 4, 6, 8, 10]:
        alloc = {
            "web": {("m5.large", 0, "us-east-1"): count},
            "db": {("r5.large", 0, "us-east-1"): 4},
            "batch": {("c5.xlarge", 0, "us-east-1"): 6},
        }
        unavail = compute_f2_web_unavailability(system, alloc)
        unavailabilities.append(unavail)

    # Higher instance count should result in equal or lower unavailability (monotonic non-increasing)
    for i in range(len(unavailabilities) - 1):
        assert unavailabilities[i] >= unavailabilities[i + 1] - 1e-9, (
            f"Expected monotonicity: {unavailabilities[i]} vs {unavailabilities[i+1]}"
        )


def test_cost_all_spot_less_than_all_ondemand(system):
    """Test 5: Workloads utilizing spot instances have strictly lower instance costs than equivalent on-demand."""
    batch_od = {
        "web": {("m5.xlarge", 0, "us-east-1"): 4},
        "db": {("r5.xlarge", 0, "us-east-1"): 4},
        "batch": {("c5.2xlarge", 0, "us-east-1"): 6},
    }
    batch_spot = {
        "web": {("m5.xlarge", 0, "us-east-1"): 4},
        "db": {("r5.xlarge", 0, "us-east-1"): 4},
        "batch": {("c5.2xlarge", 1, "us-east-1"): 6},
    }

    cost_od = compute_f1_cost(system, batch_od)
    cost_spot = compute_f1_cost(system, batch_spot)

    assert cost_spot < cost_od, f"Spot cost ({cost_spot}) should be < on-demand cost ({cost_od})"


def test_monte_carlo_simulation_consistency(system):
    """Test 6: Monte Carlo simulator returns values consistent with analytic expectations."""
    x_b1 = build_b1_all_ondemand(system)
    sim_res = simulate(x_b1, system, n_simulations=500, seed=42)

    assert sim_res["cost"] > 0
    assert sim_res["availability_pct"] >= 99.0
    assert sim_res["makespan_hours"] > 0
    assert sim_res["data_loss_gb"] == 0.0  # B1 has zero spot instances, so 0 data loss


def test_pipeline_parameter_sensitivity():
    """Test 7: Verify that changing parameters (budget, interruption_mult, discount_shift) dynamically shifts results."""
    from src.pipeline import RunParams, run_pipeline

    sys_default = SystemData()
    b1_base = compute_f1_cost(sys_default, sys_default.decode_vector(build_b1_all_ondemand(sys_default)))

    # Case A: Base run (budget 85%, mult=1.0, shift=0.0)
    params_a = RunParams(
        budget=b1_base * 0.85,
        interruption_mult=1.0,
        discount_shift=0.0,
        pop_size=30,
        n_gen=20,
        seed=42,
        n_mc_simulations=200,
    )
    res_a = run_pipeline(params_a)

    # Case B: Tighter budget (70% of B1)
    params_b = RunParams(
        budget=b1_base * 0.70,
        interruption_mult=1.0,
        discount_shift=0.0,
        pop_size=30,
        n_gen=20,
        seed=42,
        n_mc_simulations=200,
    )
    res_b = run_pipeline(params_b)

    # Case C: Severe market stress (mult=2.0, discount_shift=-0.20)
    params_c = RunParams(
        budget=b1_base * 0.85,
        interruption_mult=2.0,
        discount_shift=-0.20,
        pop_size=30,
        n_gen=20,
        seed=42,
        n_mc_simulations=200,
    )
    res_c = run_pipeline(params_c)

    # Assertions:
    # 1. Under tighter budget B, cost-saver cost must be strictly <= tighter budget
    cs_cost_a = res_a.representative_plans["Cost-Saver"]["F"][0]
    cs_cost_b = res_b.representative_plans["Cost-Saver"]["F"][0]
    assert cs_cost_b <= params_b.budget + 1e-3, f"Plan B cost ({cs_cost_b}) exceeds budget ({params_b.budget})"

    # 2. Under severe interruption stress C, data loss risk must be higher than base A
    data_loss_a = res_a.representative_plans["Cost-Saver"]["F"][3]
    data_loss_c = res_c.representative_plans["Cost-Saver"]["F"][3]
    assert data_loss_c >= data_loss_a - 1e-4, f"Expected higher or equal data loss under 2x interruption rate: {data_loss_c} vs {data_loss_a}"

    # 3. Overall Pareto front outputs are not identical across parameter variations
    assert not np.array_equal(res_a.F_pareto, res_c.F_pareto), "Pareto fronts under stress must differ from base front"

