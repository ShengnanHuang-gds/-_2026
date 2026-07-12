#!/usr/bin/env python3
from __future__ import annotations

from config.simulation_config import SimulationConfig
from metrics.performance_tracker import PerformanceTracker
from simulation.simulation_engine import SimulationEngine


def format_metric(name: str, stats: dict) -> str:
    return f"{name}: {stats['mean']:.4f} ± {stats['ci_half_width']:.4f}"


METRIC_DISPLAY_NAMES = {
    "evaluation_days": "evaluation_days",
    "total_profit": "total_profit",
    "fill_rate": "fill_rate",
    "lost_sales_rate": "lost_sales_rate",
    "average_forward_inventory": (
        "daily_average_forward_inventory_per_fw_product"
    ),
    "average_central_inventory": (
        "daily_average_central_inventory_per_product"
    ),
    "forward_capacity_utilization_mean": (
        "daily_forward_capacity_utilization_mean"
    ),
    "forward_capacity_utilization_max": (
        "daily_forward_capacity_utilization_max"
    ),
    "forward_capacity_utilization_p95": (
        "daily_forward_capacity_utilization_p95"
    ),
    "central_capacity_utilization_mean": (
        "daily_central_capacity_utilization_mean"
    ),
    "central_capacity_utilization_max": (
        "daily_central_capacity_utilization_max"
    ),
    "central_capacity_utilization_p95": (
        "daily_central_capacity_utilization_p95"
    ),
}

METRIC_SECTIONS = [
    ("Profit & Service", [
        "evaluation_days",
        "total_profit",
        "fill_rate",
        "lost_sales_rate",
    ]),
   
    (
        "Daily Capacity Utilization (inventory / capacity)",
        [
            "forward_capacity_utilization_mean",
            "forward_capacity_utilization_max",
            "forward_capacity_utilization_p95",
            "central_capacity_utilization_mean",
            "central_capacity_utilization_max",
            "central_capacity_utilization_p95",
        ],
    ),
]


def print_baseline_summary(final_report: dict) -> None:
    print("\n=== Baseline Summary (mean ± 95% CI) ===")

    for section_title, metric_names in METRIC_SECTIONS:
        print(f"\n--- {section_title} ---")
        for metric_name in metric_names:
            if metric_name not in final_report:
                continue
            display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
            print(format_metric(display_name, final_report[metric_name]))

    shortage_stats = final_report.get("central_shortage_frequency_by_product")
    if shortage_stats:
        print("\n--- Daily CW Shortage Frequency by Product ---")
        for product_index, product_stats in enumerate(shortage_stats, start=1):
            print(
                f"  P{product_index}: "
                f"{product_stats['mean']:.4f} ± {product_stats['ci_half_width']:.4f}"
            )


def main() -> None:
    config = SimulationConfig()
    replication_results = []

    print(
        f"Running {config.num_replications} replications "
        f"({config.warmup_days} warmup + {config.evaluation_days} evaluation days)"
    )

    for replication_index in range(config.num_replications):
        seed = config.random_seed_base + replication_index
        engine = SimulationEngine(config, seed=seed)
        result = engine.run()
        replication_results.append(result)
        print(
            f"  Replication {replication_index + 1:2d} "
            f"(seed={seed}): profit={result['total_profit']:.2f}, "
            f"fill_rate={result['fill_rate']:.4f}"
        )

    final_report = PerformanceTracker.report_final_metrics(replication_results)
    print_baseline_summary(final_report)


if __name__ == "__main__":
    main()
