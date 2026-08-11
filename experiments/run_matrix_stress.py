#!/usr/bin/env python3
"""
Matrix stress test: compare PDF vs harsh transition matrices under disruption_mode=both.

Keeps the PDF profile as the course baseline; harsh_* are stress overlays only.

Writes:
  logs/matrix_stress_comparison.log
  logs/matrix_stress_<profile>.log
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.disruption_chain import MATRIX_PROFILES
from config.simulation_config import SimulationConfig
from metrics.performance_tracker import PerformanceTracker
from simulation.simulation_engine import SimulationEngine

LOG_DIR = ROOT / "logs"

PROFILES = ["pdf", "harsh_mild", "harsh_strong"]

METRIC_KEYS = [
    "total_profit",
    "fill_rate",
    "lost_sales_rate",
    "average_forward_inventory",
    "average_central_inventory",
]


def format_metric(stats: dict) -> str:
    return f"{stats['mean']:.4f} ± {stats['ci_half_width']:.4f}"


def mean_product(product_stats: list) -> float:
    return sum(x["mean"] for x in product_stats) / len(product_stats)


def run_scenario(config: SimulationConfig) -> dict:
    reps = []
    for r in range(config.num_replications):
        seed = config.random_seed_base + r
        engine = SimulationEngine(config, seed=seed)
        reps.append(engine.run())
    return PerformanceTracker.report_final_metrics(reps)


def write_profile_log(
    path: Path,
    profile: str,
    config: SimulationConfig,
    report: dict,
) -> None:
    desc = MATRIX_PROFILES[profile]["description"]
    lines = [
        f"Scenario: matrix stress / profile={profile}",
        f"Description: {desc}",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"disruption_mode: {config.disruption_mode}",
        f"disruption_matrix_profile: {config.disruption_matrix_profile}",
        f"Replications: {config.num_replications} "
        f"({config.warmup_days} warmup + {config.evaluation_days} eval)",
        f"random_seed_base: {config.random_seed_base}",
        f"demand_intensity_scale: {config.demand_intensity_scale}",
        "",
        "=== Summary (mean ± 95% CI) ===",
        "",
    ]
    for key in METRIC_KEYS:
        lines.append(f"{key}: {format_metric(report[key])}")
    lines.append("")
    lines.append(
        f"cw_shortage_freq_mean: "
        f"{mean_product(report['central_shortage_frequency_by_product']):.4f}"
    )
    lines.append(
        f"cw_zero_inv_freq_mean: "
        f"{mean_product(report['central_zero_inventory_frequency_by_product']):.4f}"
    )
    lines.append(
        f"fw_stockout_freq_mean: "
        f"{mean_product(report['forward_stockout_frequency_by_product']):.4f}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison(path: Path, configs: dict, reports: dict) -> None:
    lines = [
        "Scenario: Matrix stress comparison (disruption_mode=both)",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Profiles:",
    ]
    for profile in PROFILES:
        lines.append(f"  - {profile}: {MATRIX_PROFILES[profile]['description']}")
    lines.append("")
    header = (
        f"{'Metric':<28}"
        + "".join(f"{p:<28}" for p in PROFILES)
    )
    lines.append(header)
    lines.append("-" * len(header))
    for key in METRIC_KEYS:
        row = f"{key:<28}"
        for profile in PROFILES:
            row += f"{format_metric(reports[profile][key]):<28}"
        lines.append(row)

    lines.append("")
    lines.append("Derived frequencies (product-mean):")
    for label, metric in [
        ("cw_shortage_freq", "central_shortage_frequency_by_product"),
        ("cw_zero_inv_freq", "central_zero_inventory_frequency_by_product"),
        ("fw_stockout_freq", "forward_stockout_frequency_by_product"),
    ]:
        row = f"{label:<28}"
        for profile in PROFILES:
            row += f"{mean_product(reports[profile][metric]):<28.4f}"
        lines.append(row)

    lines.extend(
        [
            "",
            "Notes:",
            "- pdf is the course baseline; harsh_* are stress overlays only.",
            "- Expect fill_rate to fall and lost_sales / shortage freqs to rise",
            "  as the profile becomes harsher.",
            "- Switch with SimulationConfig(disruption_matrix_profile=...).",
            "- Or pass custom_m_a / custom_m_u for a fully custom pair.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base_kwargs = dict(
        disruption_mode="both",
        warmup_days=30,
        evaluation_days=365,
        num_replications=30,
        demand_intensity_scale=1.2,
    )

    reports = {}
    configs = {}
    for profile in PROFILES:
        config = SimulationConfig(
            disruption_matrix_profile=profile,
            **base_kwargs,
        )
        configs[profile] = config
        print(f"Running profile={profile} ...")
        report = run_scenario(config)
        reports[profile] = report
        log_path = LOG_DIR / f"matrix_stress_{profile}.log"
        write_profile_log(log_path, profile, config, report)
        print(
            f"  fill={report['fill_rate']['mean']:.4f}  "
            f"lost={report['lost_sales_rate']['mean']:.4f}  "
            f"profit={report['total_profit']['mean']:.1f}"
        )
        print(f"  → {log_path.relative_to(ROOT)}")

    comparison = LOG_DIR / "matrix_stress_comparison.log"
    write_comparison(comparison, configs, reports)
    print(f"Written {comparison.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
