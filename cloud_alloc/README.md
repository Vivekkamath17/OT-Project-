# Cloud Workload Multi-Objective Allocation Prototype
**University "Optimization Techniques" Mid-Evaluation Project**

> [!WARNING]
> **SYNTHETIC DATA NOTICE**:
> All instance specs, spot price discounts, and monthly interruption rates in `data/instances.csv`, `data/spot_stats.csv`, and `data/workloads.json` are **SYNTHETIC illustrative values**, calibrated on public AWS-style price and interruption bands. Replace with real production telemetry and spot trace logs for production deployment. The data loader is designed so real CSVs can drop in with the identical schema.

---

## 1. Problem Formulation

Growing startups traditionally deploy 100% on-demand instances (our **B1 baseline**) to guarantee uptime, incurring massive monthly cloud costs. This prototype solves the **Multi-Objective Cloud Workload Allocation Problem**, generating a **Pareto front of trade-off allocation plans** across on-demand and spot instances across 3 geographical regions (`ap-south-1`, `eu-west-1`, `us-east-1`).

### Workloads
1. **Web (`web`)**: Customer-facing tier. High availability requirement ($A(x) \ge 0.999$), latency target $L_{\max} \le 120$ ms, subject to worst-case spot reclamation resilience (C5).
2. **Database (`db`)**: Mission-critical transactional database. Restricted to compliance regions (`eu-west-1`, `us-east-1`) and strictly forbidden from using spot instances (C2).
3. **Batch (`batch`)**: Backend asynchronous data processing. High throughput ($34,560$ vCPU-hours/month) with a 15-minute fixed checkpointing protocol.

---

## 2. Mathematical Model

### Decision Variables
- $x[w, k, m, r] \in \mathbb{Z}_{\ge 0}$: Integer count of instances for workload $w$, instance type $k \in \{0,\dots,7\}$, purchase option $m \in \{0=\text{on-demand}, 1=\text{spot}\}$, and region $r$.
- $y[w, r] \in \{0, 1\}$: Derived indicator; $1$ if workload $w$ has $\ge 1$ instance active in region $r$.
- $s[w] \in [0, 1]$: Derived spot capacity fraction for workload $w$.

### Objectives (All Minimized)
1. **$f_1$ (Monthly Cost in USD/month)**:
   $$f_1 = \sum_{w,k,m,r} x[w,k,m,r] \cdot \text{price}(k,m,r) \cdot 720 + \text{Storage Cost} + \text{Inter-Region Transfer Cost}$$
2. **$f_2$ (Web Unavailability)**:
   $$f_2 = 1 - A(x) = 1 - P(\text{at least } N_{\text{req}} \text{ web instances are up})$$
   Calculated analytically using dynamic programming over the Poisson-binomial distribution tail, where $a_{\text{od}} = 0.9995$ and $a_{\text{spot}} = 1 - (\lambda_{k,r} \times \mu_{\text{outage}})$.
3. **$f_3$ (Batch Makespan in Hours)**:
   $$f_3 = \frac{\text{Total Work (vCPU-hours)}}{\text{Effective Throughput}}$$
   where effective throughput accounts for rework losses during spot interruptions under a fixed 15-minute checkpoint interval.
4. **$f_4$ (Expected Data-Loss Risk in GB/month)**:
   $$f_4 = \sum_{\text{spot instances}} \lambda_{k,r} \cdot \left(\frac{\tau}{2} \cdot r_{\text{state}} \cdot v_k\right)$$
   *(Note: $f_2$ and $f_4$ are correlated; toggle `DEFAULT_OBJECTIVES_COUNT = 3` in `config.py` to demote $f_4$ to a constraint $\le 30$ GB/month if advised).*

### Constraints (C1 – C7)
- **C1**: Allocated CPU, RAM, and instance counts $\ge$ requirements for every workload.
- **C2**: Database tier spot allocation is strictly zero ($x[\text{db}, k, 1, r] = 0$).
- **C3**: Monthly cost $f_1 \le \text{Budget}$ (set to 85% of B1 all-on-demand cost).
- **C4**: Instances placed strictly in workload allowed regions.
- **C5**: Web capacity after losing all spot instances in the largest spot region $\ge \text{Peak Load} \times (1 + \text{headroom})$.
- **C6**: Web latency (regional network latency + queuing utilization delay) $\le L_{\max}$ ($120$ ms).
- **C7**: Integer counts bounded by quota $[0, 15]$.

---

## 3. Optimization Architecture & Baselines

- **B1 All-On-Demand**: Standard industry practice (100% on-demand in optimal allowed regions).
- **B2 Greedy Spot**: Selects cheapest spot instance types and repairs to meet C5 and C3.
- **B3 Random Search**: Budget-matched random search evaluating $N_{\text{evals}} = \text{pop\_size} \times \text{n\_gen}$ repaired candidates.
- **NSGA-II (Main Method)**: Multi-objective genetic algorithm via `pymoo` with integer encoding, SBX crossover, polynomial mutation, seeded initial population (B1 + B2), and a domain-specific **Repair Operator**:
  - Zeroes spot instances for DB.
  - Zeroes disallowed regions.
  - Adds/removes instances to strictly enforce C1 and C5.
  - Trims excess budget by converting non-db on-demand instances to spot.

---

## 4. Verification & Validation

### Automated Pytest Suite (`tests/test_model.py`)
- `test_b1_baseline_is_feasible`: B1 baseline satisfies all C1-C7 constraints.
- `test_db_never_on_spot_after_repair`: DB never retains spot instances after repair.
- `test_repaired_solutions_are_feasible`: Random individuals are reliably repaired to feasibility.
- `test_availability_monotone_in_ondemand_instances`: Web availability is strictly monotonic non-decreasing with on-demand capacity.
- `test_cost_all_spot_less_than_all_ondemand`: Spot allocation exhibits lower cost than equivalent on-demand.
- `test_monte_carlo_simulation_consistency`: 5,000-trial Monte Carlo simulator confirms analytic DP formulas within <0.5%.

---

## 5. How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Run Automated Tests
```bash
python -m pytest tests/test_model.py -v
```

### Run End-to-End Evaluation CLI (5 Seeds)
```bash
python run_experiments.py --seeds 5 --pop 100 --gen 150
```
This runs NSGA-II and B3 over 5 seeds, calculates mean/std hypervolume, generates representative plans, runs 5,000-trial Monte Carlo validation, and saves 5 publication-ready plots and CSVs in `results/`.

### Run Interactive Streamlit Dashboard
```bash
python -m streamlit run app.py
```

---

## 6. Prototype Scope vs Planned Final Project Extensions

| Component | In Scope (This Mid-Evaluation Prototype) | Out of Scope (Planned for Final Project) |
|---|---|---|
| **Algorithms** | NSGA-II with domain Repair operator | MOEA/D, NSGA-III, Hybrid Local Search (Memetic) |
| **Baselines** | B1 All-OD, B2 Greedy, B3 Random Search | B4 Weighted-Sum Scalarization, Exact MILP $\epsilon$-Constraint |
| **Data** | 8 instance types, 3 regions, synthetic calibrated CSVs | Real live AWS Price API & 30-day spot interruption trace logs |
| **Checkpoints** | Fixed 15-minute checkpoint interval ($\tau = 0.25$ hr) | Decision variable $\tau[w]$ jointly optimized with instance mix |
| **Scale** | 3 workloads, 112 decision slots, laptop execution < 3 mins | 20+ microservices, 15 AWS regions, Kubernetes autoscaling trace |

---

## 7. CTO Strategic Recommendations
1. **DB Tier**: Strictly 100% on-demand. Spot interruption costs exceed savings due to state re-sync and failovers.
2. **Web Tier**: Hybrid multi-region deployment (65% on-demand in `us-east-1`, 35% spot spread across `eu-west-1` and `ap-south-1`). C5 constraint ensures SLA compliance even during full regional spot reclamation.
3. **Batch Tier**: 80-100% spot instances with 15-min checkpointing. Interruption rework loss is $<0.3\%$, yielding $40-60\%$ overall cloud cost savings.
