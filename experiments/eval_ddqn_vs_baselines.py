#!/usr/bin/env python3
"""
Formal multi-rep profit comparison for a trained DDQN checkpoint.

Default: harsh_strong, forbid_e1 rule-warmstart ckpt, 30 shared seeds vs
fixed_0 / best_fixed / rule_based.

Example::

    python experiments/eval_ddqn_vs_baselines.py --reps 30
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.simulation_config import SimulationConfig
from metrics.performance_tracker import PerformanceTracker
from mdp.env import RetailMDPEnv
from mdp.policies import make_baseline_policies
from mdp.state_encoder import build_state_encoder_from_env
from rl.dqn import build_q_network, states_to_tensor
from rl.dqn_train import DQNTrainConfig, build_env

LOG_DIR = ROOT / "logs"
CKPT_DIR = LOG_DIR / "checkpoints"

METRIC_KEYS = [
    "total_profit",
    "fill_rate",
    "lost_sales_rate",
    "average_forward_inventory",
    "average_central_inventory",
    "total_expedite_cost",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval DDQN ckpt vs baselines")
    p.add_argument("--profile", default="harsh_strong")
    p.add_argument(
        "--checkpoint",
        default=str(CKPT_DIR / "ddqn_harsh_strong_forbid_e1_rule_ws_best.pt"),
    )
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-days", type=int, default=30)
    p.add_argument("--evaluation-days", type=int, default=365)
    p.add_argument("--demand-intensity-scale", type=float, default=1.2)
    p.add_argument("--forbid-expedite", action="store_true", default=True)
    p.add_argument("--allow-expedite", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--log-path",
        default="",
    )
    args = p.parse_args(argv)
    if args.allow_expedite:
        args.forbid_expedite = False
    if not args.log_path:
        tag = "forbid_e1_rule_ws" if "rule_ws" in args.checkpoint else "ddqn"
        args.log_path = str(
            LOG_DIR / f"ddqn_{args.profile}_{tag}_vs_baselines_{args.reps}rep.log"
        )
    return args


def format_metric(stats: dict) -> str:
    return f"{stats['mean']:.4f} ± {stats['ci_half_width']:.4f}"


def evaluate_baseline(
    config: SimulationConfig, policy
) -> Tuple[dict, List[float]]:
    profits: List[float] = []
    reps = []
    for r in range(config.num_replications):
        seed = config.random_seed_base + r
        env = RetailMDPEnv(config, seed=seed, forbid_expedite=False)
        state = env.reset(seed=seed)
        while not env.done:
            state, _, _, _ = env.step(policy.select(state))
        summary = env.engine.performance_tracker.summarize()
        reps.append(summary)
        profits.append(float(summary["total_profit"]))
        print(
            f"  [{policy.name}] rep {r + 1}/{config.num_replications} "
            f"profit={profits[-1]:.1f}",
            flush=True,
        )
    return PerformanceTracker.report_final_metrics(reps), profits


@torch.no_grad()
def evaluate_ddqn(
    online,
    train_cfg: DQNTrainConfig,
    *,
    num_reps: int,
    seed_base: int,
    device: torch.device,
) -> Tuple[dict, List[float]]:
    online.eval()
    profits: List[float] = []
    reps = []
    for r in range(num_reps):
        seed = seed_base + r
        env = build_env(
            train_cfg,
            seed=seed,
            evaluation_days=train_cfg.evaluation_days,
            warmup_days=train_cfg.warmup_days,
        )
        state = env.reset(seed=seed)
        encoder = build_state_encoder_from_env(
            env, encoder_type=train_cfg.state_encoder
        )
        while not env.done:
            mask = env.get_action_mask()
            vec = encoder.encode(state)
            action = online.select_greedy(
                states_to_tensor(vec, device=device),
                action_mask=torch.as_tensor(mask, dtype=torch.bool, device=device),
            )
            state, _, _, _ = env.step(action)
        summary = env.engine.performance_tracker.summarize()
        reps.append(summary)
        profits.append(float(summary["total_profit"]))
        print(
            f"  [ddqn_greedy] rep {r + 1}/{num_reps} profit={profits[-1]:.1f}",
            flush=True,
        )
    return PerformanceTracker.report_final_metrics(reps), profits


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    device = torch.device(args.device)
    train_cfg = DQNTrainConfig(
        algorithm="ddqn",
        disruption_matrix_profile=args.profile,
        warmup_days=args.warmup_days,
        evaluation_days=args.evaluation_days,
        demand_intensity_scale=args.demand_intensity_scale,
        forbid_expedite=bool(args.forbid_expedite),
        seed=args.seed,
        device=args.device,
    )

    # Infer network dim
    probe = build_env(train_cfg, seed=args.seed)
    probe.reset(seed=args.seed)
    encoder = build_state_encoder_from_env(
        probe, encoder_type=train_cfg.state_encoder
    )
    online = build_q_network(state_dim=encoder.dim, device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    online.load_state_dict(ckpt["model_state_dict"])

    sim = SimulationConfig(
        disruption_mode="both",
        disruption_matrix_profile=args.profile,
        warmup_days=args.warmup_days,
        evaluation_days=args.evaluation_days,
        num_replications=args.reps,
        demand_intensity_scale=args.demand_intensity_scale,
        random_seed_base=args.seed,
    )

    print(
        f"Formal eval | profile={args.profile} reps={args.reps} "
        f"forbid_expedite={args.forbid_expedite} "
        f"demand_scale={args.demand_intensity_scale}\n"
        f"ckpt={ckpt_path} (train_best={ckpt.get('best_eval_profit')})",
        flush=True,
    )

    t0 = time.time()
    results: Dict[str, dict] = {}
    profit_lists: Dict[str, List[float]] = {}

    print("\n=== ddqn_greedy ===", flush=True)
    report, profits = evaluate_ddqn(
        online,
        train_cfg,
        num_reps=args.reps,
        seed_base=args.seed,
        device=device,
    )
    results["ddqn_greedy"] = report
    profit_lists["ddqn_greedy"] = profits

    for policy in make_baseline_policies(args.profile):
        print(f"\n=== {policy.name} ===", flush=True)
        report, profits = evaluate_baseline(sim, policy)
        results[policy.name] = report
        profit_lists[policy.name] = profits

    elapsed = time.time() - t0

    # Write log
    lines = [
        "Formal DDQN vs baselines profit comparison",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"profile: {args.profile}",
        f"checkpoint: {ckpt_path}",
        f"ckpt_train_best_eval: {ckpt.get('best_eval_profit')}",
        f"forbid_expedite (DDQN): {args.forbid_expedite}",
        f"demand_intensity_scale: {args.demand_intensity_scale}",
        f"reps: {args.reps}  seed_base: {args.seed}  "
        f"(seeds = {args.seed} .. {args.seed + args.reps - 1})",
        f"horizon: {args.warmup_days}+{args.evaluation_days}",
        f"elapsed_sec: {elapsed:.1f}",
        f"CI: mean ± 1.96*std/sqrt(n)  [95% half-width]",
        "",
    ]

    for name, report in results.items():
        lines.append(f"=== {name} ===")
        for key in METRIC_KEYS:
            if key in report:
                lines.append(f"  {key}: {format_metric(report[key])}")
        lines.append("")

    ddqn_mean = results["ddqn_greedy"]["total_profit"]["mean"]
    lines.append("=== Δ total_profit (ddqn − baseline) ===")
    for name, report in results.items():
        if name == "ddqn_greedy":
            continue
        base = report["total_profit"]["mean"]
        lines.append(
            f"  vs {name}: {ddqn_mean - base:+.1f} "
            f"({100.0 * (ddqn_mean / base - 1):+.2f}%)"
        )
    lines.append("")

    # Per-rep table (compact)
    lines.append("=== per-rep total_profit ===")
    names = list(profit_lists.keys())
    header = f"{'rep':>4s} {'seed':>6s} " + " ".join(f"{n[:12]:>12s}" for n in names)
    lines.append(header)
    for r in range(args.reps):
        seed = args.seed + r
        row = f"{r + 1:4d} {seed:6d} " + " ".join(
            f"{profit_lists[n][r]:12.1f}" for n in names
        )
        lines.append(row)

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY (total_profit mean ± 95% CI half-width)", flush=True)
    for name, report in results.items():
        print(f"  {name}: {format_metric(report['total_profit'])}", flush=True)
    print("Deltas:", flush=True)
    for name, report in results.items():
        if name == "ddqn_greedy":
            continue
        base = report["total_profit"]["mean"]
        print(
            f"  vs {name}: {ddqn_mean - base:+.1f} "
            f"({100.0 * (ddqn_mean / base - 1):+.2f}%)",
            flush=True,
        )
    print(f"Wrote {log_path}", flush=True)


if __name__ == "__main__":
    main()
