#!/usr/bin/env python3
"""
Full-horizon Double DQN training with periodic greedy evaluation.

Defaults:
  algorithm=ddqn, harsh_mild, evaluation_days=365, episodes=50,
  lr=3e-4, eps 1.0 → 0.01 over 30k steps, target sync every 500 updates,
  greedy eval every 5 episodes on 365-day horizon (3 seeds).

Example::

    python experiments/train_dqn.py
    python experiments/train_dqn.py --algorithm ddqn --episodes 100
    python experiments/train_dqn.py --algorithm dqn   # vanilla ablation
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.dqn_train import DQNTrainConfig, EvalResult, run_dqn_training

LOG_DIR = ROOT / "logs"
CKPT_DIR = LOG_DIR / "checkpoints"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-horizon DQN / DDQN trainer")
    p.add_argument(
        "--algorithm",
        choices=["dqn", "ddqn"],
        default="ddqn",
        help="dqn=vanilla max target; ddqn=Double DQN (default)",
    )
    p.add_argument("--profile", default="harsh_mild")
    p.add_argument("--warmup-days", type=int, default=30)
    p.add_argument("--evaluation-days", type=int, default=365,
                   help="train episode horizon")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--start-learning-after", type=int, default=1000)
    p.add_argument("--buffer-capacity", type=int, default=100000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reward-scale", type=float, default=10000.0)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.01,
                   help="low exploration floor for long training")
    p.add_argument("--eps-decay-steps", type=int, default=30000)
    p.add_argument("--target-update-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=5,
                   help="greedy full-horizon eval every N train episodes (0=off)")
    p.add_argument("--eval-days", type=int, default=365)
    p.add_argument("--eval-warmup-days", type=int, default=30)
    p.add_argument("--eval-seeds", type=int, default=3)
    p.add_argument(
        "--forbid-expedite",
        action="store_true",
        default=True,
        help="hard-ban all e=1 actions (default on)",
    )
    p.add_argument(
        "--allow-expedite",
        action="store_true",
        help="disable forbid_expedite (full 18-action space)",
    )
    p.add_argument(
        "--log-path",
        default=str(LOG_DIR / "ddqn_train.log"),
    )
    p.add_argument(
        "--checkpoint-path",
        default=str(CKPT_DIR / "ddqn_best.pt"),
    )
    p.add_argument("--no-checkpoint", action="store_true")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> DQNTrainConfig:
    ckpt = None if args.no_checkpoint else args.checkpoint_path
    forbid = False if args.allow_expedite else True
    return DQNTrainConfig(
        algorithm=args.algorithm,
        disruption_matrix_profile=args.profile,
        warmup_days=args.warmup_days,
        evaluation_days=args.evaluation_days,
        episodes=args.episodes,
        batch_size=args.batch_size,
        start_learning_after=args.start_learning_after,
        buffer_capacity=args.buffer_capacity,
        lr=args.lr,
        gamma=args.gamma,
        seed=args.seed,
        device=args.device,
        reward_scale=args.reward_scale,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        target_update_every=args.target_update_every,
        eval_every_episodes=args.eval_every,
        eval_evaluation_days=args.eval_days,
        eval_warmup_days=args.eval_warmup_days,
        eval_num_seeds=args.eval_seeds,
        forbid_expedite=forbid,
        checkpoint_path=ckpt,
    )


def write_log(path: Path, cfg: DQNTrainConfig, summary: Dict[str, object]) -> None:
    lines = [
        f"{cfg.algorithm.upper()} full-horizon training",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"algorithm: {cfg.algorithm}",
        f"forbid_expedite: {cfg.forbid_expedite}",
        f"profile: {cfg.disruption_matrix_profile}",
        f"mode: {cfg.disruption_mode}",
        f"warmup_days: {cfg.warmup_days}",
        f"train evaluation_days: {cfg.evaluation_days}",
        f"episodes: {cfg.episodes}",
        f"reward_scale: {cfg.reward_scale}",
        f"gamma: {cfg.gamma}  lr: {cfg.lr}",
        f"batch_size: {cfg.batch_size}  "
        f"start_learning_after: {cfg.start_learning_after}",
        f"buffer_capacity: {cfg.buffer_capacity}",
        f"target_update_every: {cfg.target_update_every}",
        f"eps: {cfg.eps_start} → {cfg.eps_end} over {cfg.eps_decay_steps} steps",
        f"eval_every: {cfg.eval_every_episodes}  "
        f"eval_days: {cfg.eval_evaluation_days}  "
        f"eval_seeds: {cfg.eval_num_seeds}",
        f"seed: {cfg.seed}  device: {cfg.device}",
        "",
        f"global_steps: {summary['global_steps']}",
        f"updates: {summary['updates']}",
        f"elapsed_sec: {summary['elapsed_sec']:.2f}",
        f"buffer_size: {summary['buffer_size']}",
        f"final_eps: {summary['final_eps']:.4f}",
        f"best_eval_profit: {summary['best_eval_profit']}",
        "",
        "=== train episodes ===",
        "episode | scaled_return | raw_profit | mean_loss",
    ]
    returns = summary["episode_returns"]
    raws = summary["episode_raw_profits"]
    losses = summary["episode_losses"]
    for i, (ret, raw, loss) in enumerate(zip(returns, raws, losses), start=1):
        loss_s = f"{loss:.5f}" if loss == loss else "nan"
        lines.append(f"{i:7d} | {ret:13.3f} | {raw:10.1f} | {loss_s}")

    lines.append("")
    lines.append("=== greedy eval (full horizon) ===")
    lines.append("episode | mean_profit | std_profit | profits")
    eval_history: Sequence[EvalResult] = summary["eval_history"]  # type: ignore
    for ev in eval_history:
        profits_s = ", ".join(f"{p:.1f}" for p in ev.profits)
        lines.append(
            f"{ev.episode:7d} | {ev.mean_raw_profit:11.1f} | "
            f"{ev.std_raw_profit:10.1f} | [{profits_s}]"
        )

    lines.extend(
        [
            "",
            "Baseline reference (harsh_mild, from logs/policy_baselines_harsh_mild.log):",
            "  fixed_0 ≈ 2.453e6",
            "  best_fixed_13 ≈ 2.467e6",
            "  rule_based ≈ 2.486e6",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)
    if cfg.checkpoint_path:
        Path(cfg.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    print(
        f"{cfg.algorithm.upper()} train | profile={cfg.disruption_matrix_profile} "
        f"forbid_expedite={cfg.forbid_expedite} "
        f"train_days={cfg.evaluation_days} episodes={cfg.episodes} "
        f"lr={cfg.lr} eps_end={cfg.eps_end} "
        f"eval_every={cfg.eval_every_episodes} "
        f"eval_days={cfg.eval_evaluation_days} device={cfg.device}",
        flush=True,
    )
    summary = run_dqn_training(cfg)
    summary_for_log = dict(summary)
    summary_for_log.pop("online", None)
    summary_for_log.pop("best_state_dict", None)

    log_path = Path(args.log_path)
    write_log(log_path, cfg, summary_for_log)
    print(
        f"done in {summary['elapsed_sec']:.1f}s | "
        f"updates={summary['updates']} | "
        f"best_eval={summary['best_eval_profit']} | log={log_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
