"""Domain model, data loader, and multi-objective evaluation functions.

SYNTHETIC DATA NOTICE:
All instance and spot parameters are synthetic illustrative values calibrated on
public AWS-style price/interruption bands. Replace with real traces/CSVs for production use.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import config


@dataclass(frozen=True)
class InstanceType:
    """Cloud instance specification."""
    type_name: str
    vcpu: int
    ram_gb: float
    ondemand_price_hr: float


@dataclass(frozen=True)
class SpotStat:
    """Spot market historical statistics by instance type and region."""
    type_name: str
    region: str
    spot_discount: float
    interruption_rate: float
    price_volatility: float


@dataclass
class Workload:
    """Workload requirements and operational constraints."""
    name: str
    cpu_req: int
    ram_req: float
    storage_req: float
    allowed_regions: List[str]
    peak_load: float
    headroom: float
    min_instances: int
    spot_allowed: bool
    batch_total_work: float
    latency_target_ms: float
    base_latency_by_region: Dict[str, float]
    data_transfer_gb: float


@dataclass(frozen=True)
class Slot:
    """One decision variable slot: x[w, k, m, r]."""
    index: int
    workload: str
    type_name: str
    option: int  # 0: on-demand, 1: spot
    region: str


class SystemData:
    """Container for system-wide catalog, spot stats, and workloads."""

    def __init__(
        self,
        instances_path: Path = config.INSTANCES_CSV,
        spot_stats_path: Path = config.SPOT_STATS_CSV,
        workloads_path: Path = config.WORKLOADS_JSON,
    ):
        self.instances_path = Path(instances_path)
        self.spot_stats_path = Path(spot_stats_path)
        self.workloads_path = Path(workloads_path)

        self.instances: Dict[str, InstanceType] = {}
        self.spot_stats: Dict[Tuple[str, str], SpotStat] = {}
        self.workloads: Dict[str, Workload] = {}

        self.load_data()
        self._build_variable_slots()

    def load_data(self) -> None:
        """Load instances, spot statistics, and workload profiles."""
        # 1. Instances
        with open(self.instances_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].strip().startswith("#") or row[0].strip().lower() == "type":
                    continue
                type_name, vcpu, ram_gb, price = (
                    row[0].strip(),
                    int(row[1]),
                    float(row[2]),
                    float(row[3]),
                )
                self.instances[type_name] = InstanceType(
                    type_name=type_name,
                    vcpu=vcpu,
                    ram_gb=ram_gb,
                    ondemand_price_hr=price,
                )

        # 2. Spot statistics
        with open(self.spot_stats_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].strip().startswith("#") or row[0].strip().lower() == "type":
                    continue
                type_name, region, discount, ir, vol = (
                    row[0].strip(),
                    row[1].strip(),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                )
                self.spot_stats[(type_name, region)] = SpotStat(
                    type_name=type_name,
                    region=region,
                    spot_discount=discount,
                    interruption_rate=ir,
                    price_volatility=vol,
                )

        # 3. Workloads
        with open(self.workloads_path, mode="r", encoding="utf-8") as f:
            payload = json.load(f)
            for w_key, w_val in payload["workloads"].items():
                self.workloads[w_key] = Workload(
                    name=w_val.get("name", w_key),
                    cpu_req=int(w_val["cpu_req"]),
                    ram_req=float(w_val["ram_req"]),
                    storage_req=float(w_val.get("storage_req", 0.0)),
                    allowed_regions=list(w_val["allowed_regions"]),
                    peak_load=float(w_val["peak_load"]),
                    headroom=float(w_val["headroom"]),
                    min_instances=int(w_val["min_instances"]),
                    spot_allowed=bool(w_val["spot_allowed"]),
                    batch_total_work=float(w_val.get("batch_total_work", 0.0)),
                    latency_target_ms=float(w_val.get("latency_target_ms", 120.0)),
                    base_latency_by_region={
                        k: float(v) for k, v in w_val.get("base_latency_by_region", {}).items()
                    },
                    data_transfer_gb=float(w_val.get("data_transfer_gb", 1000.0)),
                )

    def _build_variable_slots(self) -> None:
        """Construct the compact flat decision variable mapping."""
        self.slots: List[Slot] = []
        self.slot_by_key: Dict[Tuple[str, str, int, str], int] = {}

        idx = 0
        for w_key, w in self.workloads.items():
            for r in w.allowed_regions:
                for t_key in self.instances.keys():
                    options = [0, 1] if w.spot_allowed else [0]
                    for opt in options:
                        slot = Slot(
                            index=idx,
                            workload=w_key,
                            type_name=t_key,
                            option=opt,
                            region=r,
                        )
                        self.slots.append(slot)
                        self.slot_by_key[(w_key, t_key, opt, r)] = idx
                        idx += 1

        self.n_vars = len(self.slots)

    def decode_vector(self, x: np.ndarray) -> Dict[str, Dict[Tuple[str, int, str], int]]:
        """Decode a 1D integer vector into workload-keyed dictionaries."""
        allocation: Dict[str, Dict[Tuple[str, int, str], int]] = {
            w: {} for w in self.workloads.keys()
        }
        for slot in self.slots:
            count = int(x[slot.index])
            if count > 0:
                allocation[slot.workload][(slot.type_name, slot.option, slot.region)] = count
        return allocation

    def encode_dict(self, alloc_dict: Dict[str, Dict[Tuple[str, int, str], int]]) -> np.ndarray:
        """Encode a nested dictionary into a 1D integer vector."""
        vec = np.zeros(self.n_vars, dtype=int)
        for w_key, items in alloc_dict.items():
            for (t_name, opt, reg), count in items.items():
                key = (w_key, t_name, opt, reg)
                if key in self.slot_by_key:
                    vec[self.slot_by_key[key]] = count
        return vec

    def get_hourly_price(
        self,
        type_name: str,
        option: int,
        region: str,
        scenario_high_spot: bool = False,
    ) -> float:
        """Calculate hourly price for an instance with optional volatility stress."""
        base_price = self.instances[type_name].ondemand_price_hr
        reg_mult = config.REGION_PRICE_MULTIPLIERS.get(region, 1.0)
        od_price = base_price * reg_mult

        if option == 0:
            return od_price

        stat = self.spot_stats.get((type_name, region))
        discount = stat.spot_discount if stat else 0.65
        volatility = stat.price_volatility if stat else 0.10

        effective_discount = max(0.0, discount - volatility) if scenario_high_spot else discount
        return od_price * (1.0 - effective_discount)

    def get_interruption_rate(self, type_name: str, region: str) -> float:
        """Return the monthly interruption rate for a spot instance."""
        stat = self.spot_stats.get((type_name, region))
        return stat.interruption_rate if stat else 0.10


def compute_f1_cost(
    system: SystemData,
    x_alloc: Dict[str, Dict[Tuple[str, int, str], int]],
    scenario_high_spot: bool = False,
) -> float:
    """Objective f1: Total monthly cloud expenditure (USD/month).

    Formula:
    f1 = sum(x[w,k,m,r] * price(k,m,r) * 720) + data_transfer_cost + storage_cost
    """
    instance_cost = 0.0
    for w_key, items in x_alloc.items():
        for (t_name, opt, reg), count in items.items():
            price_hr = system.get_hourly_price(t_name, opt, reg, scenario_high_spot)
            instance_cost += count * price_hr * config.MONTH_HOURS

    # Storage cost per workload
    storage_cost = sum(
        w.storage_req * config.STORAGE_COST_PER_GB_MONTH for w in system.workloads.values()
    )

    # Data transfer: base inter-region data transfer plus multi-region coordination penalty
    data_transfer_cost = 0.0
    for w_key, items in x_alloc.items():
        w = system.workloads[w_key]
        active_regions = {reg for (_, _, reg), count in items.items() if count > 0}
        base_transfer = w.data_transfer_gb * config.DATA_TRANSFER_COST_PER_GB
        # If instances span across multiple regions, add cross-region data replication fee
        cross_region_fee = max(0, len(active_regions) - 1) * 20.0
        data_transfer_cost += base_transfer + cross_region_fee

    return float(instance_cost + storage_cost + data_transfer_cost)


def compute_f2_web_unavailability(
    system: SystemData,
    x_alloc: Dict[str, Dict[Tuple[str, int, str], int]],
) -> float:
    """Objective f2: Unavailability of customer-facing web tier (1 - A(x)).

    Formula:
    A(x) = P(at least N_req web instances are up)
    Each instance modeled as independent Bernoulli trial:
      a_od = 0.9995
      a_spot = 1 - (interruption_rate * mean_outage_fraction)
    Calculated via exact dynamic programming (Poisson-binomial distribution tail).
    """
    web_alloc = x_alloc.get("web", {})
    web_w = system.workloads["web"]
    n_req = web_w.min_instances

    # Collect individual instance availability probabilities
    p_list: List[float] = []
    for (t_name, opt, reg), count in web_alloc.items():
        if opt == 0:
            p_inst = config.AVAILABILITY_ON_DEMAND
        else:
            ir = system.get_interruption_rate(t_name, reg)
            p_inst = 1.0 - (ir * config.MEAN_OUTAGE_FRACTION)
        p_list.extend([p_inst] * count)

    total_instances = len(p_list)
    if total_instances < n_req:
        return 1.0  # Cannot satisfy min required instances

    # Exact DP for Poisson-binomial PMF: dp[j] = P(exactly j instances up)
    dp = np.zeros(total_instances + 1, dtype=np.float64)
    dp[0] = 1.0

    for p in p_list:
        next_dp = np.zeros_like(dp)
        next_dp[0] = dp[0] * (1.0 - p)
        next_dp[1:] = dp[1:] * (1.0 - p) + dp[:-1] * p
        dp = next_dp

    # A(x) = sum_{j >= n_req} dp[j]
    a_x = float(np.sum(dp[n_req:]))
    a_x = min(1.0, max(0.0, a_x))
    return float(1.0 - a_x)


def compute_f3_batch_makespan(
    system: SystemData,
    x_alloc: Dict[str, Dict[Tuple[str, int, str], int]],
) -> float:
    """Objective f3: Expected batch workload makespan in hours.

    Formula:
    f3 = batch_total_work / effective_throughput
    effective_throughput = sum(vCPUs) reduced by rework loss from interruptions:
      spot vCPUs * interruption_rate * (checkpoint_interval / 2) / 720
    """
    batch_alloc = x_alloc.get("batch", {})
    batch_w = system.workloads["batch"]
    total_work = batch_w.batch_total_work

    effective_throughput = 0.0
    for (t_name, opt, reg), count in batch_alloc.items():
        vcpu = system.instances[t_name].vcpu * count
        if opt == 0:
            effective_throughput += vcpu
        else:
            ir = system.get_interruption_rate(t_name, reg)
            # Rework lost fraction of the 720-hour month
            rework_loss_fraction = ir * (config.FIXED_CHECKPOINT_INTERVAL_HOURS / 2.0) / config.MONTH_HOURS
            effective_throughput += vcpu * (1.0 - rework_loss_fraction)

    if effective_throughput <= 1e-4:
        return 9999.0

    return float(total_work / effective_throughput)


def compute_f4_data_loss(
    system: SystemData,
    x_alloc: Dict[str, Dict[Tuple[str, int, str], int]],
) -> float:
    """Objective f4: Expected data-loss risk (GB of unsaved state per month).

    Formula:
    f4 = sum over spot instances of (interruption_rate * expected_unsaved_state)
    where expected_unsaved_state = (checkpoint_interval / 2) * state_generation_rate * vCPUs
    """
    total_loss = 0.0
    for w_key in ["batch", "web"]:
        alloc = x_alloc.get(w_key, {})
        for (t_name, opt, reg), count in alloc.items():
            if opt == 1:  # Spot instances only
                ir = system.get_interruption_rate(t_name, reg)
                vcpu = system.instances[t_name].vcpu * count
                expected_unsaved_state = (
                    (config.FIXED_CHECKPOINT_INTERVAL_HOURS / 2.0)
                    * config.STATE_GEN_RATE_GB_PER_VCPU_HR
                    * vcpu
                )
                total_loss += ir * expected_unsaved_state

    return float(total_loss)


def evaluate(
    x: Union[np.ndarray, Dict[str, Dict[Tuple[str, int, str], int]]],
    system: SystemData,
    n_objectives: int = config.DEFAULT_OBJECTIVES_COUNT,
    scenario_high_spot: bool = False,
) -> np.ndarray:
    """Unified evaluation function returning [f1, f2, f3, f4] or [f1, f2, f3]."""
    if isinstance(x, np.ndarray):
        x_alloc = system.decode_vector(x)
    else:
        x_alloc = x

    f1 = compute_f1_cost(system, x_alloc, scenario_high_spot)
    f2 = compute_f2_web_unavailability(system, x_alloc)
    f3 = compute_f3_batch_makespan(system, x_alloc)
    f4 = compute_f4_data_loss(system, x_alloc)

    if n_objectives == 3:
        return np.array([f1, f2, f3], dtype=np.float64)
    return np.array([f1, f2, f3, f4], dtype=np.float64)
