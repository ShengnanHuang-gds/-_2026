#!/usr/bin/env python3
"""
Fixed-action grid under a chosen disruption matrix profile.

Default: Both disruption + harsh_strong (aggressive stress).

Usage:
  PYTHONPATH=. python experiments/run_fixed_action_harsh_grid.py
  PYTHONPATH=. python experiments/run_fixed_action_harsh_grid.py harsh_mild
  PYTHONPATH=. python experiments/run_fixed_action_harsh_grid.py harsh_strong
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.simulation_config import SimulationConfig
from mdp.actions import all_actions
from metrics.performance_tracker import PerformanceTracker
from simulation.simulation_engine import SimulationEngine

LOG_DIR = ROOT / "logs"


def main() -> None:
    profile = sys.argv[1] if len(sys.argv) > 1 else "harsh_strong"
    out = LOG_DIR / f"fixed_action_grid_{profile}_both.csv"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(
        disruption_mode="both",
        disruption_matrix_profile=profile,
        warmup_days=30,
        evaluation_days=365,
        num_replications=30,
        demand_intensity_scale=1.2,
    )
    print(config)

    rows = []
    for action in all_actions():
        reps = []
        for r in range(config.num_replications):
            engine = SimulationEngine(config, seed=config.random_seed_base + r)
            reps.append(engine.run(action=action.action_id))
        report = PerformanceTracker.report_final_metrics(reps)
        profit = report["total_profit"]["mean"]
        fill = report["fill_rate"]["mean"]
        lost = report["lost_sales_rate"]["mean"]
        inv = (
            report["average_forward_inventory"]["mean"]
            + report["average_central_inventory"]["mean"]
        )
        exp = report["total_expedite_cost"]["mean"]
        print(
            f"action {action.action_id:>2} "
            f"(ell={action.ell} e={action.expedite} g={action.allocation}) "
            f"profit={profit:.1f} fill={fill:.4f} lost={lost:.4f} "
            f"inv={inv:.1f} exp={exp:.1f}"
        )
        rows.append(
            f"{action.action_id},{action.ell},{action.expedite},{action.allocation},"
            f"{action.allocation_rule_name},{profit:.4f},{fill:.6f},{lost:.6f},"
            f"{inv:.4f},{exp:.4f}"
        )

    header = (
        "action_id,ell,e,g,g_name,profit,fill_rate,"
        "lost_sales_rate,avg_inventory,expedite_cost"
    )
    out.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"Written {out.relative_to(ROOT)}")

    # Quick ranking
    parsed = []
    for line in rows:
        parts = line.split(",")
        parsed.append(
            {
                "aid": int(parts[0]),
                "ell": int(parts[1]),
                "e": int(parts[2]),
                "g": int(parts[3]),
                "name": parts[4],
                "profit": float(parts[5]),
                "fill": float(parts[6]),
                "lost": float(parts[7]),
            }
        )
    top = sorted(parsed, key=lambda x: -x["profit"])[:5]
    print("\nTop-5 by profit:")
    for i, r in enumerate(top, 1):
        print(
            f"  {i}. action {r['aid']} "
            f"(ell={r['ell']},e={r['e']},g={r['g']}/{r['name']}) "
            f"profit={r['profit']:.1f} fill={r['fill']:.4f} lost={r['lost']:.4f}"
        )
    base = next(x for x in parsed if x["aid"] == 0)
    best = top[0]
    print(
        f"\nGap best vs action0: "
        f"profit {best['profit'] - base['profit']:+.1f}, "
        f"fill {best['fill'] - base['fill']:+.4f}, "
        f"lost {best['lost'] - base['lost']:+.4f}"
    )


if __name__ == "__main__":
    main()
