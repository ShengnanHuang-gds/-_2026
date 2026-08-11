#!/usr/bin/env python3
"""
Compare baseline policies across three disruption matrix profiles.

Policies:
  1) fixed_0_(s,S)     – always action 0
  2) best_fixed_*      – profile-specific best from fixed-action grids
  3) rule_based        – A_t/U_t severity rule (no always-on expedite)

Profiles: pdf | harsh_mild | harsh_strong
Mode: disruption_mode=both

Writes:
  logs/policy_baselines_<profile>.log
  logs/policy_baselines_comparison.log
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.disruption_chain import MATRIX_PROFILES
from config.simulation_config import SimulationConfig
from metrics.performance_tracker import PerformanceTracker
from mdp.env import RetailMDPEnv
from mdp.policies import make_baseline_policies

LOG_DIR = ROOT / "logs"
PROFILES = ["pdf", "harsh_mild", "harsh_strong"]

METRIC_KEYS = [
    "total_profit",
    "fill_rate",
    "lost_sales_rate",
    "average_forward_inventory",
    "average_central_inventory",
    "total_expedite_cost",
]


def format_metric(stats: dict) -> str:
    return f"{stats['mean']:.4f} ± {stats['ci_half_width']:.4f}"


def run_policy_episode(config: SimulationConfig, seed: int, policy) -> dict:
    """One replication via RetailMDPEnv (warmup silent; policy only on eval days)."""
    env = RetailMDPEnv(config, seed=seed)
    state = env.reset()
    while not env.done:
        action_id = policy.select(state)
        state, _reward, _done, _info = env.step(action_id)
    return env.engine.performance_tracker.summarize()


def evaluate_policy(config: SimulationConfig, policy) -> dict:
    reps = []
    for r in range(config.num_replications):
        seed = config.random_seed_base + r
        reps.append(run_policy_episode(config, seed, policy))
    return PerformanceTracker.report_final_metrics(reps)


def write_profile_log(
    path: Path,
    profile: str,
    config: SimulationConfig,
    results: Dict[str, dict],
) -> None:
    lines = [
        f"Scenario: policy baselines / matrix={profile}",
        f"Description: {MATRIX_PROFILES[profile]['description']}",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"disruption_mode: {config.disruption_mode}",
        f"disruption_matrix_profile: {config.disruption_matrix_profile}",
        f"Replications: {config.num_replications} "
        f"({config.warmup_days} warmup + {config.evaluation_days} eval)",
        f"random_seed_base: {config.random_seed_base}",
        f"demand_intensity_scale: {config.demand_intensity_scale}",
        "",
        "Rule (v1): A0/U0→(0,0,1); mild A1/U1→(1,0,1); "
        "severe A>=2 or U>=2→(2,0,1); U3 keeps e=0",
        "",
    ]
    for policy_name, report in results.items():
        lines.append(f"=== {policy_name} ===")
        for key in METRIC_KEYS:
            lines.append(f"  {key}: {format_metric(report[key])}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison(path: Path, all_results: dict) -> None:
    lines = [
        "Policy baselines comparison (disruption_mode=both)",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Policies: fixed_0_(s,S) | best_fixed_* | rule_based",
        "Profiles: pdf | harsh_mild | harsh_strong",
        "",
    ]

    for profile in PROFILES:
        results = all_results[profile]
        policy_names = list(results.keys())
        lines.append(f"## {profile}")
        lines.append(f"  {MATRIX_PROFILES[profile]['description']}")
        header = f"{'Metric':<28}" + "".join(f"{n:<28}" for n in policy_names)
        lines.append(header)
        lines.append("-" * len(header))
        for key in METRIC_KEYS:
            row = f"{key:<28}"
            for name in policy_names:
                row += f"{format_metric(results[name][key]):<28}"
            lines.append(row)

        # Gaps vs fixed_0
        base = results[policy_names[0]]
        lines.append("")
        lines.append("  Delta vs fixed_0 (mean only):")
        for name in policy_names[1:]:
            dp = results[name]["total_profit"]["mean"] - base["total_profit"]["mean"]
            df = results[name]["fill_rate"]["mean"] - base["fill_rate"]["mean"]
            dl = results[name]["lost_sales_rate"]["mean"] - base["lost_sales_rate"]["mean"]
            lines.append(
                f"    {name}: profit {dp:+.1f}, fill {df:+.4f}, lost {dl:+.4f}"
            )
        lines.append("")

    lines.extend(
        [
            "Notes:",
            "- All policies evaluated via RetailMDPEnv with common seeds.",
            "- Warmup uses baseline (no MDP action); policy starts at eval day 1.",
            "- best_fixed is profile-specific (pdf:1, harsh_mild:13, harsh_strong:12).",
            "- rule_based does not always-on expedite (cost dominated grids).",
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

    all_results = {}
    for profile in PROFILES:
        config = SimulationConfig(
            disruption_matrix_profile=profile,
            **base_kwargs,
        )
        policies = make_baseline_policies(profile)
        results = {}
        print(f"\n[{profile}]")
        for policy in policies:
            print(f"  running {policy.name} ...", flush=True)
            report = evaluate_policy(config, policy)
            results[policy.name] = report
            print(
                f"    profit={report['total_profit']['mean']:.1f}  "
                f"fill={report['fill_rate']['mean']:.4f}  "
                f"lost={report['lost_sales_rate']['mean']:.4f}  "
                f"exp={report['total_expedite_cost']['mean']:.1f}"
            )
        all_results[profile] = results
        log_path = LOG_DIR / f"policy_baselines_{profile}.log"
        write_profile_log(log_path, profile, config, results)
        print(f"  → {log_path.relative_to(ROOT)}")

    comparison = LOG_DIR / "policy_baselines_comparison.log"
    write_comparison(comparison, all_results)
    print(f"\nWritten {comparison.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
