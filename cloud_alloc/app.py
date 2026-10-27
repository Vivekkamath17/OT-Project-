"""Streamlit Interactive Dashboard for Multi-Objective Cloud Workload Allocation.

Pure, reactive architecture:
  - RunParams & Results decouple pipeline logic from presentation.
  - @st.cache_data caches based on all explicit primitive parameters.
  - Interactive "Run Optimization" button with st.session_state ensures
    controlled execution and transparent status banner.
  - Zero hard-coded values; all metrics, charts, tables, and CTO takeaways
    are computed dynamically from the active results.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from src.model import SystemData, evaluate
from src.pipeline import Results, RunParams, create_adjusted_system, run_pipeline


@st.cache_data(show_spinner="Running Multi-Objective Evolutionary Optimization (NSGA-II)...")
def cached_run_pipeline(
    budget: float,
    interruption_mult: float,
    discount_shift: float,
    pop_size: int,
    n_gen: int,
    seed: int,
    n_objectives: int = 4,
) -> Results:
    """Cached wrapper around pure run_pipeline using explicit primitive arguments."""
    params = RunParams(
        budget=budget,
        interruption_mult=interruption_mult,
        discount_shift=discount_shift,
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed,
        n_objectives=n_objectives,
        n_mc_simulations=1000,
    )
    return run_pipeline(params)


def main():
    st.set_page_config(
        page_title="Cloud Workload Multi-Objective Allocator",
        page_icon="☁️",
        layout="wide",
    )

    st.title("☁️ Multi-Objective Workload Allocation on Cloud Instances")
    st.caption("Notice: Calibrated on illustrative public AWS-style price/interruption bands.")

    # 1. Sidebar Configuration
    st.sidebar.header("⚙️ Optimization Parameters")

    # Baseline reference under default conditions
    base_sys = SystemData()
    from src.baselines import build_b1_all_ondemand
    default_b1_cost = float(evaluate(build_b1_all_ondemand(base_sys), base_sys, n_objectives=1)[0])
    default_budget_val = float(round(default_b1_cost * config.DEFAULT_BUDGET_FRACTION, 0))

    budget_input = st.sidebar.slider(
        "Monthly Budget Cap ($)",
        min_value=float(round(default_b1_cost * 0.45, 0)),
        max_value=float(round(default_b1_cost * 1.15, 0)),
        value=default_budget_val,
        step=50.0,
        help="Hard constraint (C3): allocation monthly expenditure cannot exceed this budget.",
    )
    st.sidebar.caption(f"Default B1: ${default_b1_cost:,.0f}/mo | Slider: **{budget_input / default_b1_cost * 100:.1f}%** of B1")

    st.sidebar.subheader("Spot Market Stress Testing")
    ir_mult = st.sidebar.slider(
        "Interruption Rate Multiplier",
        min_value=0.5,
        max_value=2.5,
        value=1.0,
        step=0.1,
        help="Stress-test spot revocation risk across all regions.",
    )

    discount_shift = st.sidebar.slider(
        "Spot Discount Shift",
        min_value=-0.30,
        max_value=0.20,
        value=0.0,
        step=0.05,
        help="Simulate spot price surges (-0.20 reduces a 70% discount down to 50%).",
    )

    st.sidebar.subheader("Genetic Algorithm Settings")
    n_gen = st.sidebar.slider("Generations", min_value=20, max_value=150, value=60, step=10)
    pop_size = st.sidebar.slider("Population Size", min_value=30, max_value=120, value=60, step=10)
    seed_val = int(st.sidebar.number_input("Random Seed", value=42, step=1))

    # Action Button
    run_btn = st.sidebar.button("🚀 Run Optimization", type="primary", use_container_width=True)

    # 2. Execution / Session State Management
    current_params = RunParams(
        budget=budget_input,
        interruption_mult=ir_mult,
        discount_shift=discount_shift,
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed_val,
    )

    if "results" not in st.session_state or run_btn:
        results = cached_run_pipeline(
            budget=budget_input,
            interruption_mult=ir_mult,
            discount_shift=discount_shift,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed_val,
        )
        st.session_state["results"] = results
        st.session_state["params"] = current_params

    results: Results = st.session_state["results"]
    active_params: RunParams = st.session_state["params"]

    # 3. Parameter Status Banner
    is_stale = (active_params != current_params)
    status_icon = "⚠️" if is_stale else "✅"
    st.info(
        f"{status_icon} **Showing results for**: Budget=${active_params.budget:,.0f} | "
        f"Interruption Multiplier={active_params.interruption_mult:.1f}x | "
        f"Discount Shift={active_params.discount_shift:+.2f} | "
        f"Pop={active_params.pop_size} | Gen={active_params.n_gen} | Seed={active_params.seed}"
        + (" *(Sidebar inputs changed - click 'Run Optimization' to update)*" if is_stale else ""),
        icon="ℹ️",
    )

    # Infeasibility Warning
    if not results.is_budget_feasible:
        st.warning(
            f"⚠️ **Infeasible Budget**: The selected budget (${active_params.budget:,.2f}/mo) is below "
            f"the minimum achievable cost (${results.min_achievable_cost:,.2f}/mo) for this system under C1/C5. "
            f"No valid Pareto front could be formed. Increase the budget slider.",
            icon="⚠️",
        )

    b1_cost = results.b1_cost
    rep_plans = results.representative_plans

    # 4. Dynamic KPI Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("B1 All-OD Baseline Cost", f"${b1_cost:,.2f}/mo")

    if "Cost-Saver" in rep_plans:
        cs = rep_plans["Cost-Saver"]["card"]
        m2.metric(
            "Cost-Saver Plan",
            f"${cs['Cost ($/mo)']:,.2f}/mo",
            f"-{cs['Cost Saving vs B1 (%)']:.1f}%",
        )
    else:
        m2.metric("Cost-Saver Plan", "N/A")

    if "Balanced" in rep_plans:
        bal = rep_plans["Balanced"]["card"]
        m3.metric(
            "Balanced (Knee) Plan",
            f"${bal['Cost ($/mo)']:,.2f}/mo",
            f"-{bal['Cost Saving vs B1 (%)']:.1f}%",
        )
    else:
        m3.metric("Balanced Plan", "N/A")

    if "Reliability-First" in rep_plans:
        rel = rep_plans["Reliability-First"]["card"]
        m4.metric(
            "Reliability-First Plan",
            f"${rel['Cost ($/mo)']:,.2f}/mo",
            f"-{rel['Cost Saving vs B1 (%)']:.1f}%",
        )
    else:
        m4.metric("Reliability-First Plan", "N/A")

    st.divider()

    # 5. Visualizations: Pareto Scatter & Cost Comparison
    st.subheader("🎯 Multi-Objective Trade-Off Landscape")
    col_chart, col_bar = st.columns([1.6, 1.0])

    with col_chart:
        if len(results.F_pareto) > 0:
            df_pareto = pd.DataFrame(
                results.F_pareto,
                columns=["Cost", "Unavailability", "Makespan", "DataLoss"][: results.F_pareto.shape[1]],
            )
            df_pareto["Availability_pct"] = (1.0 - df_pareto["Unavailability"]) * 100.0

            color_col = "DataLoss" if "DataLoss" in df_pareto.columns else None
            fig_scatter = px.scatter(
                df_pareto,
                x="Cost",
                y="Availability_pct",
                color=color_col,
                size="Makespan",
                hover_data={
                    "Cost": ":$,.2f",
                    "Availability_pct": ":.4f%",
                    "Makespan": ":.1f hrs",
                },
                color_continuous_scale="Viridis",
                labels={
                    "Cost": "Monthly Cost ($/month)",
                    "Availability_pct": "Web Availability (%)",
                    "DataLoss": "Data Loss Risk (GB/mo)",
                    "Makespan": "Batch Makespan (hrs)",
                },
                title=f"Pareto Front ({len(results.F_pareto)} Solutions | HV: {results.hv:.4f})",
            )

            # Add Baseline B1
            fig_scatter.add_trace(
                go.Scatter(
                    x=[results.f_b1[0]],
                    y=[(1.0 - results.f_b1[1]) * 100.0],
                    mode="markers+text",
                    name="B1 All-OD (Current)",
                    text=["B1 (All-OD)"],
                    textposition="bottom center",
                    marker=dict(size=14, color="crimson", symbol="square"),
                )
            )

            # Add Representative Plans
            for p_name, p_data in rep_plans.items():
                if p_name in ["Cost-Saver", "Balanced", "Reliability-First"]:
                    fp = p_data["F"]
                    color = "#2ca02c" if "Cost" in p_name else ("#1f77b4" if "Balanced" in p_name else "#9467bd")
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=[fp[0]],
                            y=[(1.0 - fp[1]) * 100.0],
                            mode="markers+text",
                            name=p_name,
                            text=[p_name],
                            textposition="top center",
                            marker=dict(size=16, color=color, symbol="star"),
                        )
                    )

            fig_scatter.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No non-dominated Pareto solutions found. Adjust the parameters or budget.", icon="ℹ️")

    with col_bar:
        bar_items = [
            {"Plan": "B1 All-OD", "Cost": b1_cost, "Category": "Baseline"},
            {"Plan": "B2 Greedy", "Cost": results.b2_cost, "Category": "Baseline"},
        ]
        for name in ["Cost-Saver", "Balanced", "Reliability-First"]:
            if name in rep_plans:
                bar_items.append({
                    "Plan": name,
                    "Cost": rep_plans[name]["card"]["Cost ($/mo)"],
                    "Category": "Optimized",
                })

        bar_df = pd.DataFrame(bar_items)
        fig_bar = px.bar(
            bar_df,
            x="Plan",
            y="Cost",
            color="Category",
            text="Cost",
            title="Monthly Cost Comparison vs Budget",
            color_discrete_map={"Baseline": "#e74c3c", "Optimized": "#2ecc71"},
        )
        fig_bar.add_hline(
            y=active_params.budget,
            line_dash="dot",
            annotation_text=f"Budget: ${active_params.budget:,.0f}",
            line_color="black",
        )
        fig_bar.update_traces(texttemplate="$%{y:,.0f}", textposition="outside")
        fig_bar.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="Cost ($/mo)")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # 6. Representative Plans Comparison Table
    st.subheader("📋 Representative Allocation Plans (Analytic vs Monte Carlo)")
    table_rows = []
    for name in ["Cost-Saver", "Balanced", "Reliability-First", "B1 All-OD", "B2 Greedy"]:
        if name in rep_plans:
            c = rep_plans[name]["card"]
            sim = c.get("simulated", {})
            table_rows.append({
                "Plan": name,
                "Analytic Cost": f"${c['Cost ($/mo)']:,.2f}",
                "Simulated Cost": f"${sim.get('cost', 0.0):,.2f}",
                "Saving vs B1 (%)": f"{c['Cost Saving vs B1 (%)']:.1f}%",
                "Web Avail (%)": f"{c['Web Availability (%)']:.4f}%",
                "Sim Avail (%)": f"{sim.get('availability_pct', 0.0):.4f}%",
                "Batch Makespan (h)": f"{c['Batch Makespan (hrs)']:.1f}h",
                "Data Loss (GB/mo)": f"{c['Data Loss Risk (GB/mo)']:.2f} GB",
            })

    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.write("No representative plans available.")

    st.divider()

    # 7. Dynamic Workload Inspector
    st.subheader("🔍 Deep-Dive Allocation Inspector")
    col_sel, col_detail = st.columns([1, 2])

    available_keys = [k for k in ["Balanced", "Cost-Saver", "Reliability-First", "B1 All-OD"] if k in rep_plans]
    if available_keys:
        with col_sel:
            selected_plan_name = st.selectbox("Select Plan to Inspect Breakdown:", available_keys)
            inspect_card = rep_plans[selected_plan_name]["card"]

            st.markdown(f"**Total Monthly Cost**: ${inspect_card['Cost ($/mo)']:,.2f}")
            st.markdown(f"**Cost Saving vs B1**: {inspect_card['Cost Saving vs B1 (%)']:.1f}%")
            st.markdown(f"**Web Availability**: {inspect_card['Web Availability (%)']:.4f}%")
            st.markdown(f"**Batch Makespan**: {inspect_card['Batch Makespan (hrs)']:.1f} hours")
            st.markdown(f"**Data Loss Risk**: {inspect_card['Data Loss Risk (GB/mo)']:.2f} GB/month")

        with col_detail:
            for w_key in ["web", "db", "batch"]:
                w_info = inspect_card.get("Workloads", {}).get(w_key, {})
                if w_info:
                    with st.expander(
                        f"📦 {w_key.upper()} Tier: {w_info.get('vcpu', 0)} vCPUs ({w_info.get('spot_fraction_pct', 0.0):.1f}% Spot) in {w_info.get('regions', 'N/A')}",
                        expanded=True,
                    ):
                        st.write(f"**Allocated Instances**: {w_info.get('instances_detail', 'None')}")
                        st.caption(
                            f"Total vCPUs: {w_info.get('vcpu', 0)} | RAM: {w_info.get('ram_gb', 0)} GB | "
                            f"Spot Capacity Share: {w_info.get('spot_fraction_pct', 0.0):.1f}%"
                        )

    st.divider()

    # 8. Dynamic CTO Takeaways
    st.subheader("💡 Strategic CTO Guidance (Data-Driven)")
    for rec in results.cto_guidance:
        st.markdown(f"- {rec}")


if __name__ == "__main__":
    main()
