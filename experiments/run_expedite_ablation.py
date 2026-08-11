#!/usr/bin/env python3
"""
Expedite ablation: e=0 vs e=1 under disruption (fixed ell=0, g=1).

Writes:
  logs/expedite_e0.log
  logs/expedite_e1.log
  logs/expedite_comparison.log   # params + side-by-side summary
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.simulation_config import SimulationConfig
from metrics.performance_tracker import PerformanceTracker
from mdp.actions import decode_action, encode_action
from simulation.simulation_engine import SimulationEngine

LOG_DIR = ROOT / "logs"

# Isolate e only: ell=0, g=1 (lowest_dos). Do NOT use action_id 0 (Phase-2 shortcut).
ACTION_E0 = encode_action(ell=0, expedite=0, allocation=1)
ACTION_E1 = encode_action(ell=0, expedite=1, allocation=1)

METRIC_KEYS = [
    "total_profit",
    "total_expedite_cost",
    "fill_rate",
    "lost_sales_rate",
    "average_forward_inventory",
    "average_central_inventory",
]


def format_metric(stats: dict) -> str:
    return f"{stats['mean']:.4f} ± {stats['ci_half_width']:.4f}"


def run_fixed_action(config: SimulationConfig, action_id: int) -> dict:
    replication_results = []
    for replication_index in range(config.num_replications):
        seed = config.random_seed_base + replication_index
        engine = SimulationEngine(config, seed=seed)
        replication_results.append(engine.run(action=action_id))
    return PerformanceTracker.report_final_metrics(replication_results)


def config_params_lines(config: SimulationConfig) -> list[str]:
    return [
        "=== Experiment parameters ===",
        f"disruption_mode: {config.disruption_mode}",
        f"enable_disruption: {config.enable_disruption}",
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
        "=== Fixed MDP action dimensions ===",
        "ell: 0 (no buffer uplift)",
        "allocation g: 1 (lowest_dos)",
        "varied: expedite e only",
        "note: action_id 0 is NOT used (would shortcut to Phase-2 path)",
    ]


def action_params_lines(action_id: int) -> list[str]:
    action = decode_action(action_id)
    return [
        "=== Action parameters ===",
        f"action_id: {action_id}",
        f"action (ell, e, g): {action.as_tuple()}",
        f"expedite: {action.expedite}",
        f"eta_central_reorder: {action.eta_central_reorder}",
        f"eta_central_order_up_to: {action.eta_central_order_up_to}",
        f"eta_forward_reorder: {action.eta_forward_reorder}",
        f"allocation_rule: {action.allocation_rule_name}",
    ]


def write_scenario_log(
    log_path: Path,
    scenario_name: str,
    config: SimulationConfig,
    action_id: int,
    final_report: dict,
) -> None:
    lines = [
        f"Scenario: {scenario_name}",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        *config_params_lines(config),
        "",
        *action_params_lines(action_id),
        "",
        "=== Summary (mean ± 95% CI) ===",
        "",
    ]
    for key in METRIC_KEYS:
        if key in final_report:
            lines.append(f"{key}: {format_metric(final_report[key])}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_log(
    log_path: Path,
    config: SimulationConfig,
    report_e0: dict,
    report_e1: dict,
) -> None:
    lines = [
        "Scenario: Expedite ablation comparison (e=0 vs e=1)",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        *config_params_lines(config),
        "",
        "=== Actions compared ===",
        f"e0 action_id={ACTION_E0} tuple={decode_action(ACTION_E0).as_tuple()}",
        f"e1 action_id={ACTION_E1} tuple={decode_action(ACTION_E1).as_tuple()}",
        "",
        "=== Side-by-side (mean ± 95% CI) ===",
        "",
        f"{'Metric':<28} {'e=0':<28} {'e=1':<28}",
    ]
    for key in METRIC_KEYS:
        if key not in report_e0 or key not in report_e1:
            continue
        lines.append(
            f"{key:<28} {format_metric(report_e0[key]):<28} "
            f"{format_metric(report_e1[key]):<28}"
        )

    lines.extend(
        [
            "",
            "=== Quick checks ===",
            "- e1 total_expedite_cost should be > e0 (≈ 0)",
            "- e1 may improve fill_rate / lower lost_sales under transport delay",
            "- e1 profit often lower due to expedite fee even if service improves",
        ]
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = SimulationConfig(
        disruption_mode="both",
        warmup_days=30,
        evaluation_days=365,
        num_replications=30,
    )

    scenarios = [
        (
            "Expedite e=0 (ell=0, g=lowest_dos)",
            LOG_DIR / "expedite_e0.log",
            ACTION_E0,
        ),
        (
            "Expedite e=1 (ell=0, g=lowest_dos)",
            LOG_DIR / "expedite_e1.log",
            ACTION_E1,
        ),
    ]

    reports: dict[int, dict] = {}
    written: list[Path] = []

    for scenario_name, log_path, action_id in scenarios:
        report = run_fixed_action(config, action_id)
        write_scenario_log(log_path, scenario_name, config, action_id, report)
        reports[action_id] = report
        written.append(log_path)
        print(f"Written {log_path.relative_to(Path.cwd())}")

    comparison_path = LOG_DIR / "expedite_comparison.log"
    write_comparison_log(
        comparison_path,
        config,
        reports[ACTION_E0],
        reports[ACTION_E1],
    )
    written.append(comparison_path)
    print(f"Written {comparison_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
