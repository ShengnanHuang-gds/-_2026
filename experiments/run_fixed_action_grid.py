#!/usr/bin/env python3
"""
Fixed-action grid experiment: 4 scenarios × 18 actions = 72 combinations.

Writes:
  logs/fixed_action_grid_<scenario>.csv   (per-scenario, all 18 actions)
  logs/fixed_action_grid_summary.md       (markdown pivot table, all scenarios)
  logs/fixed_action_grid_params.log       (run metadata)

Usage:
  cd /path/to/project && python experiments/run_fixed_action_grid.py
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.simulation_config import SimulationConfig
from mdp.actions import all_actions, decode_action, MDPAction
from metrics.performance_tracker import PerformanceTracker
from simulation.simulation_engine import SimulationEngine

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

SCENARIOS = [
    ("Baseline",       "none"),
    ("Supply-only",    "supply_only"),
    ("Transport-only", "transport_only"),
    ("Both",           "both"),
]

BASE_KWARGS = dict(
    warmup_days=30,
    evaluation_days=365,
    num_replications=30,
)

LOG_DIR = ROOT / "logs"

# Metrics to include in the output table
TABLE_METRICS = [
    "profit",
    "fill_rate",
    "lost_sales_rate",
    "avg_inventory",
    "avg_forward_inventory",
    "avg_central_inventory",
    "cw_shortage_freq",
    "fw_stockout_freq",
    "expedite_cost",
]

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def run_fixed_action(config: SimulationConfig, action_id: int) -> dict:
    reps = []
    for r in range(config.num_replications):
        seed = config.random_seed_base + r
        engine = SimulationEngine(config, seed=seed)
        reps.append(engine.run(action=action_id))
    return PerformanceTracker.report_final_metrics(reps)


def _mean_product_metric(product_stats: list) -> float:
    """Average a per-product metric list down to a scalar."""
    return sum(x["mean"] for x in product_stats) / len(product_stats) if product_stats else 0.0


def build_row(scenario_name: str, action: MDPAction, report: dict) -> dict:
    avg_fwd = report["average_forward_inventory"]["mean"]
    avg_cw  = report["average_central_inventory"]["mean"]
    return {
        "scenario":         scenario_name,
        "action_id":        action.action_id,
        "ell":              action.ell,
        "e":                action.expedite,
        "g":                action.allocation,
        "g_name":           action.allocation_rule_name,
        "profit":           report["total_profit"]["mean"],
        "profit_ci":        report["total_profit"]["ci_half_width"],
        "fill_rate":        report["fill_rate"]["mean"],
        "fill_rate_ci":     report["fill_rate"]["ci_half_width"],
        "lost_sales_rate":  report["lost_sales_rate"]["mean"],
        "lost_sales_ci":    report["lost_sales_rate"]["ci_half_width"],
        "avg_inventory":    avg_fwd + avg_cw,
        "avg_forward_inventory": avg_fwd,
        "avg_central_inventory": avg_cw,
        "cw_shortage_freq": _mean_product_metric(
            report["central_shortage_frequency_by_product"]
        ),
        "fw_stockout_freq": _mean_product_metric(
            report["forward_stockout_frequency_by_product"]
        ),
        "expedite_cost":    report["total_expedite_cost"]["mean"],
    }

# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "scenario", "action_id", "ell", "e", "g", "g_name",
    "profit", "profit_ci",
    "fill_rate", "fill_rate_ci",
    "lost_sales_rate", "lost_sales_ci",
    "avg_inventory", "avg_forward_inventory", "avg_central_inventory",
    "cw_shortage_freq", "fw_stockout_freq",
    "expedite_cost",
]


def write_scenario_csv(log_path: Path, rows: List[dict]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{row[k]:.6f}" if isinstance(row[k], float) else row[k])
                             for k in CSV_FIELDS})


def _fmt(v: float, width: int = 10) -> str:
    return f"{v:.4f}".rjust(width)


def write_summary_markdown(log_path: Path, all_rows: List[dict]) -> None:
    lines = [
        "# Fixed-Action Grid: Summary",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Columns: `profit` | `fill_rate` | `lost_sales_rate` | "
        "`avg_inventory` | `cw_shortage_freq` | `fw_stockout_freq` | `expedite_cost`",
        "",
    ]

    for scenario_name, _ in SCENARIOS:
        lines.append(f"## Scenario: {scenario_name}")
        lines.append("")
        header = (
            f"| {'aid':>3} | {'ℓ':>2} | {'e':>2} | {'g':>2} | {'g_name':<14} |"
            f" {'profit':>12} | {'fill_rate':>9} | {'lost_sales':>10} |"
            f" {'avg_inv':>9} | {'cw_short':>8} | {'fw_stock':>8} | {'exp_cost':>10} |"
        )
        sep = "|" + "|".join(["-" * (len(c) - 2) + "--" for c in header.split("|")[1:-1]]) + "|"
        lines.append(header)
        lines.append(sep)

        scenario_rows = [r for r in all_rows if r["scenario"] == scenario_name]
        for row in scenario_rows:
            lines.append(
                f"| {row['action_id']:>3} | {row['ell']:>2} | {row['e']:>2} | {row['g']:>2}"
                f" | {row['g_name']:<14}"
                f" | {row['profit']:>12.1f}"
                f" | {row['fill_rate']:>9.4f}"
                f" | {row['lost_sales_rate']:>10.4f}"
                f" | {row['avg_inventory']:>9.2f}"
                f" | {row['cw_shortage_freq']:>8.4f}"
                f" | {row['fw_stockout_freq']:>8.4f}"
                f" | {row['expedite_cost']:>10.1f} |"
            )

        # top-3 by profit
        top3 = sorted(scenario_rows, key=lambda r: r["profit"], reverse=True)[:3]
        lines.append("")
        lines.append(f"**Top-3 by profit ({scenario_name}):**")
        for rank, r in enumerate(top3, 1):
            lines.append(
                f"  {rank}. action {r['action_id']} "
                f"(ℓ={r['ell']}, e={r['e']}, g={r['g']}/{r['g_name']}) "
                f"profit={r['profit']:.1f}  fill={r['fill_rate']:.4f}"
            )
        lines.append("")

    # Cross-scenario trade-off summary
    lines += [
        "---",
        "## Trade-off notes (auto)",
        "",
        "### ℓ effect (fixed e=0, g=0, across scenarios)",
    ]
    for scenario_name, _ in SCENARIOS:
        ell_rows = {
            r["ell"]: r
            for r in all_rows
            if r["scenario"] == scenario_name and r["e"] == 0 and r["g"] == 0
        }
        lines.append(f"**{scenario_name}:**")
        for ell in sorted(ell_rows):
            r = ell_rows[ell]
            lines.append(
                f"  ℓ={ell}: profit={r['profit']:.1f}  "
                f"lost_sales={r['lost_sales_rate']:.4f}  avg_inv={r['avg_inventory']:.2f}"
            )
    lines.append("")

    lines.append("### e effect (fixed ℓ=0, g=1, across scenarios)")
    for scenario_name, _ in SCENARIOS:
        e_rows = {
            r["e"]: r
            for r in all_rows
            if r["scenario"] == scenario_name and r["ell"] == 0 and r["g"] == 1
        }
        lines.append(f"**{scenario_name}:**")
        for e in sorted(e_rows):
            r = e_rows[e]
            lines.append(
                f"  e={e}: profit={r['profit']:.1f}  "
                f"fill_rate={r['fill_rate']:.4f}  expedite_cost={r['expedite_cost']:.1f}"
            )
    lines.append("")

    lines.append("### g effect (fixed ℓ=0, e=0, across scenarios)")
    for scenario_name, _ in SCENARIOS:
        g_rows = {
            r["g"]: r
            for r in all_rows
            if r["scenario"] == scenario_name and r["ell"] == 0 and r["e"] == 0
        }
        lines.append(f"**{scenario_name}:**")
        for g in sorted(g_rows):
            r = g_rows[g]
            lines.append(
                f"  g={g}/{r['g_name']}: profit={r['profit']:.1f}  "
                f"cw_shortage={r['cw_shortage_freq']:.4f}  fw_stockout={r['fw_stockout_freq']:.4f}"
            )
    lines.append("")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_params_log(log_path: Path, config: SimulationConfig) -> None:
    lines = [
        "=== Fixed-action grid experiment ===",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"Scenarios: {[s for s, _ in SCENARIOS]}",
        f"Actions: 18 (action_id 0..17)",
        f"warmup_days: {config.warmup_days}",
        f"evaluation_days: {config.evaluation_days}",
        f"num_replications: {config.num_replications}",
        f"random_seed_base: {config.random_seed_base}",
        f"central_warehouse_capacity: {config.central_warehouse_capacity}",
        f"forward_warehouse_capacity: {config.forward_warehouse_capacity}",
        f"demand_intensity_scale: {config.demand_intensity_scale}",
        f"forward_holding_cost_ratio: {config.forward_holding_cost_ratio}",
        f"supplier_lead_time_mode: {config.supplier_lead_time_mode.value}",
        "",
        "Action map (action_id -> (ell, e, g, g_name)):",
    ]
    for a in all_actions():
        lines.append(
            f"  {a.action_id:>2}: ell={a.ell} e={a.expedite} g={a.allocation}"
            f" ({a.allocation_rule_name})"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    actions = all_actions()
    all_rows: List[dict] = []

    # Write params log once (use any config for shared params)
    sample_config = SimulationConfig(disruption_mode="none", **BASE_KWARGS)
    params_path = LOG_DIR / "fixed_action_grid_params.log"
    write_params_log(params_path, sample_config)
    print(f"Written {params_path.relative_to(ROOT)}")

    for scenario_name, mode in SCENARIOS:
        config = SimulationConfig(disruption_mode=mode, **BASE_KWARGS)
        scenario_rows: List[dict] = []

        print(f"\n[{scenario_name}] running 18 actions × {config.num_replications} reps...")

        for action in actions:
            report = run_fixed_action(config, action.action_id)
            row = build_row(scenario_name, action, report)
            scenario_rows.append(row)
            all_rows.append(row)
            print(
                f"  action {action.action_id:>2} "
                f"(ℓ={action.ell} e={action.expedite} g={action.allocation}) "
                f"profit={row['profit']:>12.1f}  fill={row['fill_rate']:.4f}  "
                f"lost={row['lost_sales_rate']:.4f}  inv={row['avg_inventory']:.1f}  "
                f"exp_cost={row['expedite_cost']:.1f}"
            )

        csv_path = LOG_DIR / f"fixed_action_grid_{mode}.csv"
        write_scenario_csv(csv_path, scenario_rows)
        print(f"  → Written {csv_path.relative_to(ROOT)}")

    summary_path = LOG_DIR / "fixed_action_grid_summary.md"
    write_summary_markdown(summary_path, all_rows)
    print(f"\nWritten {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
