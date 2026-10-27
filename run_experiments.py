"""Root launcher for run_experiments.py."""
import sys
from pathlib import Path

# Add cloud_alloc to sys.path
cloud_alloc_dir = Path(__file__).resolve().parent / "cloud_alloc"
sys.path.insert(0, str(cloud_alloc_dir))

from run_experiments import *

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Objective Cloud Allocation Prototype")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds (default: 5)")
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    parser.add_argument("--gen", type=int, default=150, help="Number of generations")
    parser.add_argument("--objectives", type=int, default=4, help="Number of objectives (3 or 4)")
    parser.add_argument("--budget-frac", type=float, default=0.85, help="Budget as fraction of B1 cost")
    parser.add_argument("--interruption-mult", type=float, default=1.0, help="Spot interruption rate multiplier")
    parser.add_argument("--discount-shift", type=float, default=0.0, help="Spot discount shift")
    args = parser.parse_args()

    run_experiment_pipeline(
        n_seeds=args.seeds,
        pop_size=args.pop,
        n_gen=args.gen,
        n_objectives=args.objectives,
        budget_frac=args.budget_frac,
        interruption_mult=args.interruption_mult,
        discount_shift=args.discount_shift,
    )
