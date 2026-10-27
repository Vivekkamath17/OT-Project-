"""Configuration constants and default parameters for Cloud Allocation Prototype.

SYNTHETIC DATA NOTICE:
All instance and spot parameters are synthetic illustrative values calibrated on
public AWS-style price/interruption bands. Replace with real traces/CSVs for production use.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

INSTANCES_CSV = DATA_DIR / "instances.csv"
SPOT_STATS_CSV = DATA_DIR / "spot_stats.csv"
WORKLOADS_JSON = DATA_DIR / "workloads.json"

# Problem Settings
MONTH_HOURS = 720.0  # Standard 30-day month in hours
DEFAULT_OBJECTIVES_COUNT = 4  # 4: [Cost, Unavailability, Makespan, DataLoss] or 3 (DataLoss as constraint)
MAX_TOLERATED_DATA_LOSS_GB = 30.0  # Constraint threshold when 3 objectives are used

# Technical & Model Parameters
FIXED_CHECKPOINT_INTERVAL_HOURS = 0.25  # 15 minutes = 0.25 hr
STATE_GEN_RATE_GB_PER_VCPU_HR = 0.5  # GB of unsaved state generated per vCPU per hour
AVAILABILITY_ON_DEMAND = 0.9995  # 99.95% single-instance availability for on-demand
MEAN_OUTAGE_FRACTION = 0.05  # Average fraction of month an interrupted spot instance is offline

# Cost Factors
DATA_TRANSFER_COST_PER_GB = 0.02  # Inter-region transfer cost ($/GB)
STORAGE_COST_PER_GB_MONTH = 0.10  # Storage cost per GB-month ($/GB-mo)
DEFAULT_BUDGET_FRACTION = 0.85  # Budget set to 85% of all-on-demand B1 baseline cost

# Regions and Price Multipliers
REGION_PRICE_MULTIPLIERS = {
    "ap-south-1": 1.00,
    "eu-west-1": 1.05,
    "us-east-1": 0.98,
}
ALL_REGIONS = ["ap-south-1", "eu-west-1", "us-east-1"]

# Optimization & Search Parameters
MAX_QUOTA_PER_SLOT = 15  # Maximum instance count per (workload, type, option, region)
DEFAULT_POP_SIZE = 100
DEFAULT_N_GEN = 150
SEEDS = [42, 101, 2024, 777, 999]
