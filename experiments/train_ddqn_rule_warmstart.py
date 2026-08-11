#!/usr/bin/env python3
"""
DDQN + rule warm-start (forbid e=1).

Pipeline:
  1) Roll out RuleBasedPolicy → fill replay
     (normal→(0,0,1), mild→(1,0,1), severe→(2,0,1))
  2) Behavioral cloning on those (s, a)
  3) Short TD pre-updates on rule data
  4) DDQN fine-tune with lower ε

Default target: harsh_strong, demand_scale=1.2, forbid_expedite=True.

Example::

    python experiments/train_ddqn_rule_warmstart.py
    python experiments/train_ddqn_rule_warmstart.py --profile harsh_mild
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.dqn_train import DQNTrainConfig, run_dqn_training

LOG_DIR = ROOT / "logs"
CKPT_DIR = LOG_DIR / "checkpoints"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDQN with rule warm-start")
    p.add_argument("--profile", default="harsh_strong")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--warmstart-episodes", type=int, default=20)
    p.add_argument("--bc-updates", type=int, default=2000)
    p.add_argument("--bc-lr", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eps-start", type=float, default=0.2)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--eps-decay-steps", type=int, default=10000)
    p.add_argument("--start-learning-after", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--demand-intensity-scale", type=float, default=1.2)
    p.add_argument("--warmup-days", type=int, default=30)
    p.add_argument("--evaluation-days", type=int, default=365)
    p.add_argument(
        "--checkpoint-path",
        default="",
    )
    p.add_argument(
        "--log-path",
        default="",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    profile = args.profile
    ckpt = args.checkpoint_path or str(
        CKPT_DIR / f"ddqn_{profile}_forbid_e1_rule_ws_best.pt"
    )
    log_path = Path(
        args.log_path
        or (LOG_DIR / f"ddqn_{profile}_forbid_e1_rule_ws.log")
    )
    Path(ckpt).parent.mkdir(parents=True, exist_ok=True)

    cfg = DQNTrainConfig(
        algorithm="ddqn",
        disruption_matrix_profile=profile,
        warmup_days=args.warmup_days,
        evaluation_days=args.evaluation_days,
        demand_intensity_scale=args.demand_intensity_scale,
        forbid_expedite=True,
        episodes=args.episodes,
        lr=args.lr,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        start_learning_after=args.start_learning_after,
        seed=args.seed,
        device=args.device,
        eval_every_episodes=args.eval_every,
        eval_evaluation_days=args.evaluation_days,
        eval_warmup_days=args.warmup_days,
        eval_num_seeds=args.eval_seeds,
        checkpoint_path=ckpt,
        rule_warmstart_episodes=args.warmstart_episodes,
        rule_bc_updates=args.bc_updates,
        rule_bc_lr=args.bc_lr,
    )

    print(
        f"DDQN rule-warmstart | profile={profile} forbid_e1=True "
        f"warmstart_eps={args.warmstart_episodes} bc={args.bc_updates} "
        f"rl_episodes={args.episodes} eps={args.eps_start}→{args.eps_end}",
        flush=True,
    )
    summary = run_dqn_training(cfg)

    ws = summary.get("warmstart", {})
    lines = [
        "DDQN + rule warm-start (forbid e=1)",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"profile: {profile}",
        f"forbid_expedite: True",
        f"demand_intensity_scale: {args.demand_intensity_scale}",
        f"warmstart_episodes: {args.warmstart_episodes}",
        f"bc_updates: {args.bc_updates}  bc_lr: {args.bc_lr}",
        f"rl_episodes: {args.episodes}",
        f"eps: {args.eps_start} → {args.eps_end} over {args.eps_decay_steps}",
        f"lr: {args.lr}",
        "",
        f"warmstart transitions: {ws.get('transitions')}",
        f"warmstart rule mean raw profit: {ws.get('rule_mean_raw_profit')}",
        f"warmstart pre-RL greedy eval: {ws.get('pre_rl_eval_profit')}",
        f"td_pre_updates: {ws.get('td_pre_updates')}",
        "",
        f"best_eval_profit: {summary.get('best_eval_profit')}",
        f"final_eps: {summary.get('final_eps')}",
        f"global_steps: {summary.get('global_steps')}",
        f"updates: {summary.get('updates')}",
        f"elapsed_sec: {summary.get('elapsed_sec'):.1f}",
        f"checkpoint: {ckpt}",
        "",
        "=== eval history ===",
    ]
    for ev in summary.get("eval_history", []):
        lines.append(
            f"ep {ev.episode:03d}: {ev.mean_raw_profit:.1f} ± {ev.std_raw_profit:.1f}"
        )
    lines.extend(
        [
            "",
            "Reference (same setting, 10-rep comparison logs):",
            "  fixed_0 ≈ 1.960M",
            "  best_fixed_12 ≈ 2.129M",
            "  rule_based ≈ 2.126M",
            "  ddqn forbid (no warmstart) ≈ 2.070M",
        ]
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"done | best_eval={summary.get('best_eval_profit')} | log={log_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
