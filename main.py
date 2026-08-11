#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from config.simulation_config import SimulationConfig
from metrics.performance_tracker import PerformanceTracker
from simulation.simulation_engine import SimulationEngine

LOG_DIR = Path(__file__).resolve().parent / "logs"

T5_METRIC_KEYS = [
    "total_profit",
    "fill_rate",
    "lost_sales_rate",
    "average_forward_inventory",
    "average_central_inventory",
]

T5_METRIC_LABELS = {
    "total_profit": "total_profit",
    "fill_rate": "fill_rate",
    "lost_sales_rate": "lost_sales_rate",
    "average_forward_inventory": "avg_forward_inventory",
    "average_central_inventory": "avg_central_inventory",
}


def format_metric(stats: dict) -> str:
    return f"{stats['mean']:.4f} ± {stats['ci_half_width']:.4f}"


def run_scenario(config: SimulationConfig) -> dict:
    replication_results = []
    for replication_index in range(config.num_replications):
        seed = config.random_seed_base + replication_index
        engine = SimulationEngine(config, seed=seed)
        replication_results.append(engine.run())
    return PerformanceTracker.report_final_metrics(replication_results)


def write_t5_log(
    log_path: Path,
    scenario_name: str,
    config: SimulationConfig,
    final_report: dict,
) -> None:
    lines = [
        f"Scenario: {scenario_name}",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"disruption_mode: {config.disruption_mode}",
        f"enable_disruption: {config.enable_disruption}",
        (
            f"Replications: {config.num_replications} "
            f"({config.warmup_days} warmup + {config.evaluation_days} evaluation days)"
        ),
        f"random_seed_base: {config.random_seed_base}",
        "",
        "=== Summary (mean ± 95% CI) ===",
        "",
    ]

    for metric_key in T5_METRIC_KEYS:
        label = T5_METRIC_LABELS[metric_key]
        lines.append(f"{label}: {format_metric(final_report[metric_key])}")

    lines.append("")
    lines.append("--- CW shortage frequency by product ---")
    shortage_stats = final_report["central_shortage_frequency_by_product"]
    for product_index, product_stats in enumerate(shortage_stats, start=1):
        lines.append(f"P{product_index}: {format_metric(product_stats)}")

    lines.append("")
    lines.append("--- CW zero-inventory days by product (end-of-day) ---")
    zero_inventory_stats = final_report["central_zero_inventory_frequency_by_product"]
    for product_index, product_stats in enumerate(zero_inventory_stats, start=1):
        lines.append(f"P{product_index}: {format_metric(product_stats)}")

    lines.append("")
    lines.append("--- FW stockout days by product (lost_sales > 0) ---")
    stockout_stats = final_report["forward_stockout_frequency_by_product"]
    for product_index, product_stats in enumerate(stockout_stats, start=1):
        lines.append(f"P{product_index}: {format_metric(product_stats)}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base_kwargs = dict(
        warmup_days=30,
        evaluation_days=365,
        num_replications=30,
    )

    scenarios = [
        (
            "Baseline (no disruption, always A0/U0)",
            LOG_DIR / "baseline.log",
            SimulationConfig(disruption_mode="none", **base_kwargs),
        ),
        (
            "Supply only (A evolves, U fixed at U0)",
            LOG_DIR / "supply_only.log",
            SimulationConfig(disruption_mode="supply_only", **base_kwargs),
        ),
        (
            "Transport only (U evolves, A fixed at A0)",
            LOG_DIR / "transport_only.log",
            SimulationConfig(disruption_mode="transport_only", **base_kwargs),
        ),
        (
            "Markov Disruption (A and U both evolve)",
            LOG_DIR / "markov_disruption.log",
            SimulationConfig(disruption_mode="both", **base_kwargs),
        ),
    ]

    written_paths: list[Path] = []
    for scenario_name, log_path, config in scenarios:
        final_report = run_scenario(config)
        write_t5_log(log_path, scenario_name, config, final_report)
        written_paths.append(log_path)

    for log_path in written_paths:
        print(f"Written {log_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
