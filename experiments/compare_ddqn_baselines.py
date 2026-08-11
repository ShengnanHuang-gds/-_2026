#!/usr/bin/env python3
"""
Compare DDQN vs baselines across matrix profiles.

Profiles (user names → config keys):
  normal → pdf
  mild   → harsh_mild
  harsh  → harsh_strong

For each profile:
  1) train DDQN (optionally forbid e=1)
  2) greedy-eval DDQN on shared seeds
  3) evaluate fixed_0 / best_fixed / rule_based on the same seeds
  4) write comparison log

Example::

    # allow e=1 (18 actions)
    python experiments/compare_ddqn_baselines.py --allow-expedite

    # hard-ban e=1 (9 actions)
    python experiments/compare_ddqn_baselines.py --forbid-expedite
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.disruption_chain import MATRIX_PROFILES
from config.simulation_config import SimulationConfig
from metrics.performance_tracker import PerformanceTracker
from mdp.actions import decode_action
from mdp.env import RetailMDPEnv
from mdp.policies import make_baseline_policies
from mdp.state_encoder import build_state_encoder_from_env
from rl.dqn import QNetwork, states_to_tensor
from rl.dqn_train import DQNTrainConfig, build_env, run_dqn_training

LOG_DIR = ROOT / "logs"
CKPT_DIR = LOG_DIR / "checkpoints"

PROFILE_ALIASES = {
    "normal": "pdf",
    "mild": "harsh_mild",
    "harsh": "harsh_strong",
}
PROFILES = ["pdf", "harsh_mild", "harsh_strong"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDQN vs baselines across matrices")
    p.add_argument(
        "--profiles",
        nargs="+",
        default=PROFILES,
        help="matrix profiles or aliases normal/mild/harsh",
    )
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--eps-decay-steps", type=int, default=15000)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-reps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--demand-intensity-scale", type=float, default=1.2)
    p.add_argument("--warmup-days", type=int, default=30)
    p.add_argument("--evaluation-days", type=int, default=365)
    expedite = p.add_mutually_exclusive_group()
    expedite.add_argument(
        "--forbid-expedite",
        dest="forbid_expedite",
        action="store_true",
        help="hard-ban e=1 (9-action subspace)",
    )
    expedite.add_argument(
        "--allow-expedite",
        dest="forbid_expedite",
        action="store_false",
        help="allow e=1 (full 18 actions)",
    )
    p.set_defaults(forbid_expedite=True)
    p.add_argument("--log-path", default="")
    args = p.parse_args(argv)
    if not args.log_path:
        tag = "forbid_e1" if args.forbid_expedite else "allow_e1"
        args.log_path = str(LOG_DIR / f"ddqn_baseline_comparison_{tag}.log")
    return args


def resolve_profiles(names: Sequence[str]) -> List[str]:
    out: List[str] = []
    for name in names:
        key = PROFILE_ALIASES.get(name, name)
        if key not in MATRIX_PROFILES:
            raise ValueError(f"unknown profile {name!r}")
        if key not in out:
            out.append(key)
    return out


def make_train_config(profile: str, args: argparse.Namespace) -> DQNTrainConfig:
    tag = "forbid_e1" if args.forbid_expedite else "allow_e1"
    ckpt = str(CKPT_DIR / f"ddqn_{profile}_{tag}_best.pt")
    return DQNTrainConfig(
        algorithm="ddqn",
        disruption_matrix_profile=profile,
        warmup_days=args.warmup_days,
        evaluation_days=args.evaluation_days,
        demand_intensity_scale=args.demand_intensity_scale,
        episodes=args.episodes,
        forbid_expedite=bool(args.forbid_expedite),
        lr=args.lr,
        eps_start=1.0,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        seed=args.seed,
        device=args.device,
        eval_every_episodes=max(5, args.episodes // 10),
        eval_evaluation_days=args.evaluation_days,
        eval_warmup_days=args.warmup_days,
        eval_num_seeds=3,
        checkpoint_path=ckpt,
    )


def make_sim_config(profile: str, args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        disruption_mode="both",
        disruption_matrix_profile=profile,
        warmup_days=args.warmup_days,
        evaluation_days=args.evaluation_days,
        num_replications=args.eval_reps,
        demand_intensity_scale=args.demand_intensity_scale,
        random_seed_base=args.seed,
    )


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
    return PerformanceTracker.report_final_metrics(reps), profits


@torch.no_grad()
def evaluate_ddqn_greedy(
    online: QNetwork,
    train_cfg: DQNTrainConfig,
    *,
    num_reps: int,
    seed_base: int,
    device: torch.device,
) -> Tuple[dict, List[float], Counter]:
    online.eval()
    profits: List[float] = []
    reps = []
    action_counts: Counter = Counter()

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
            action_counts[action] += 1
            state, _, _, _ = env.step(action)
        summary = env.engine.performance_tracker.summarize()
        reps.append(summary)
        profits.append(float(summary["total_profit"]))

    report = PerformanceTracker.report_final_metrics(reps)
    return report, profits, action_counts


def format_metric(stats: dict) -> str:
    return f"{stats['mean']:.4f} ± {stats['ci_half_width']:.4f}"


def action_dist_lines(counts: Counter, total: int) -> List[str]:
    if total <= 0:
        return ["  (no actions)"]
    e1 = sum(
        c for aid, c in counts.items() if decode_action(int(aid)).expedite == 1
    )
    lines = [
        f"  e=1 ratio: {e1 / total:.1%}  ({e1}/{total})",
        "  top actions:",
    ]
    for aid, c in counts.most_common(8):
        act = decode_action(int(aid))
        lines.append(f"    id={aid:2d} {act.as_tuple()}  {c / total:.1%}")
    return lines


def run_one_profile(profile: str, args: argparse.Namespace) -> Dict[str, object]:
    label = {v: k for k, v in PROFILE_ALIASES.items()}.get(profile, profile)
    mode = "forbid_e1" if args.forbid_expedite else "allow_e1"
    print("\n" + "=" * 60, flush=True)
    print(
        f"[{label}/{profile}] train DDQN {mode} + baselines "
        f"(reps={args.eval_reps})",
        flush=True,
    )
    print("=" * 60, flush=True)

    train_cfg = make_train_config(profile, args)
    Path(train_cfg.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train_summary = run_dqn_training(train_cfg)
    train_sec = time.time() - t0
    online = train_summary["online"]
    assert isinstance(online, QNetwork)

    ckpt_path = Path(train_cfg.checkpoint_path)
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        online.load_state_dict(ckpt["model_state_dict"])
        print(
            f"loaded best ckpt profit={ckpt.get('best_eval_profit')}",
            flush=True,
        )

    device = torch.device(args.device)
    ddqn_report, ddqn_profits, action_counts = evaluate_ddqn_greedy(
        online,
        train_cfg,
        num_reps=args.eval_reps,
        seed_base=args.seed,
        device=device,
    )

    sim = make_sim_config(profile, args)
    baseline_reports: Dict[str, dict] = {}
    for policy in make_baseline_policies(profile):
        print(f"  baseline {policy.name} ...", flush=True)
        report, _ = evaluate_baseline(sim, policy)
        baseline_reports[policy.name] = report

    return {
        "profile": profile,
        "label": label,
        "forbid_expedite": args.forbid_expedite,
        "train_sec": train_sec,
        "train_best_eval": train_summary.get("best_eval_profit"),
        "final_eps": train_summary.get("final_eps"),
        "ddqn": ddqn_report,
        "ddqn_profits": ddqn_profits,
        "action_counts": action_counts,
        "baselines": baseline_reports,
        "demand_intensity_scale": args.demand_intensity_scale,
    }


def write_comparison(path: Path, args: argparse.Namespace, rows: List[dict]) -> None:
    mode = "forbid e=1 (9 actions)" if args.forbid_expedite else "allow e=1 (18 actions)"
    lines = [
        f"DDQN ({mode}) vs baselines across matrices",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"profiles: {[r['profile'] for r in rows]}",
        f"episodes: {args.episodes}  eps_decay_steps: {args.eps_decay_steps}  "
        f"eps_end: {args.eps_end}  lr: {args.lr}",
        f"forbid_expedite: {args.forbid_expedite}",
        f"demand_intensity_scale: {args.demand_intensity_scale}",
        f"eval_reps: {args.eval_reps}  seed: {args.seed}",
        f"horizon: {args.warmup_days}+{args.evaluation_days}",
        "",
        "Aliases: normal=pdf, mild=harsh_mild, harsh=harsh_strong",
        "",
    ]

    for row in rows:
        profile = row["profile"]
        lines.append(f"=== {row['label']} / {profile} ===")
        lines.append(f"Description: {MATRIX_PROFILES[profile]['description']}")
        lines.append(f"train_sec: {row['train_sec']:.1f}")
        lines.append(f"train_best_eval (internal): {row['train_best_eval']}")
        lines.append(f"final_eps: {row['final_eps']}")
        lines.append("")
        lines.append("Profit comparison (mean ± half-width CI):")
        lines.append(f"  ddqn_greedy: {format_metric(row['ddqn']['total_profit'])}")
        for name, report in row["baselines"].items():
            lines.append(f"  {name}: {format_metric(report['total_profit'])}")

        ddqn_mean = row["ddqn"]["total_profit"]["mean"]
        fixed_mean = row["baselines"]["fixed_0_(s,S)"]["total_profit"]["mean"]
        best_name = [k for k in row["baselines"] if k.startswith("best_fixed")][0]
        best_mean = row["baselines"][best_name]["total_profit"]["mean"]
        rule_mean = row["baselines"]["rule_based"]["total_profit"]["mean"]
        lines.append("")
        lines.append(
            f"  Δ vs fixed_0: {ddqn_mean - fixed_mean:+.1f} "
            f"({100 * (ddqn_mean / fixed_mean - 1):+.2f}%)"
        )
        lines.append(
            f"  Δ vs {best_name}: {ddqn_mean - best_mean:+.1f} "
            f"({100 * (ddqn_mean / best_mean - 1):+.2f}%)"
        )
        lines.append(
            f"  Δ vs rule_based: {ddqn_mean - rule_mean:+.1f} "
            f"({100 * (ddqn_mean / rule_mean - 1):+.2f}%)"
        )
        lines.append("")
        lines.append("DDQN greedy action distribution:")
        total_actions = sum(row["action_counts"].values())
        lines.extend(action_dist_lines(row["action_counts"], total_actions))
        lines.append("")
        lines.append(
            f"  fill_rate ddqn: {format_metric(row['ddqn']['fill_rate'])}"
        )
        lines.append(
            f"  expedite_cost ddqn: "
            f"{format_metric(row['ddqn']['total_expedite_cost'])}"
        )
        lines.append("")

    lines.append("=== summary table (total_profit mean) ===")
    header = (
        f"{'profile':14s} {'ddqn':>12s} {'fixed_0':>12s} "
        f"{'best_fixed':>12s} {'rule':>12s} {'e1%':>8s}"
    )
    lines.append(header)
    for row in rows:
        best_name = [k for k in row["baselines"] if k.startswith("best_fixed")][0]
        total_a = sum(row["action_counts"].values()) or 1
        e1 = sum(
            c
            for aid, c in row["action_counts"].items()
            if decode_action(int(aid)).expedite == 1
        )
        lines.append(
            f"{row['profile']:14s} "
            f"{row['ddqn']['total_profit']['mean']:12.1f} "
            f"{row['baselines']['fixed_0_(s,S)']['total_profit']['mean']:12.1f} "
            f"{row['baselines'][best_name]['total_profit']['mean']:12.1f} "
            f"{row['baselines']['rule_based']['total_profit']['mean']:12.1f} "
            f"{100 * e1 / total_a:7.1f}%"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    profiles = resolve_profiles(args.profiles)
    mode = "forbid_e1" if args.forbid_expedite else "allow_e1"
    print(
        f"Compare DDQN({mode}) vs baselines | profiles={profiles} "
        f"episodes={args.episodes} eval_reps={args.eval_reps} "
        f"demand_scale={args.demand_intensity_scale}",
        flush=True,
    )

    rows = []
    for profile in profiles:
        rows.append(run_one_profile(profile, args))

    log_path = Path(args.log_path)
    write_comparison(log_path, args, rows)
    print(f"\nWrote {log_path}", flush=True)
    # Print summary table only (full log is on disk)
    text = log_path.read_text(encoding="utf-8")
    idx = text.find("=== summary table")
    print(text[idx:] if idx >= 0 else text, flush=True)


if __name__ == "__main__":
    main()
