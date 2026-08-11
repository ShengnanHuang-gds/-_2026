#!/usr/bin/env python3
"""
DQN smoke training: short loop to verify QNetwork + ReplayBuffer wiring.

For full-horizon training with periodic greedy eval, use::

    python experiments/train_dqn.py

Example::

    python experiments/train_dqn_smoke.py
    python experiments/train_dqn_smoke.py --episodes 10 --evaluation-days 30
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.dqn_train import DQNTrainConfig, run_dqn_training

LOG_DIR = ROOT / "logs"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DQN smoke trainer for retail MDP")
    p.add_argument("--profile", default="harsh_mild")
    p.add_argument("--warmup-days", type=int, default=5)
    p.add_argument("--evaluation-days", type=int, default=20)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--start-learning-after", type=int, default=64)
    p.add_argument("--buffer-capacity", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reward-scale", type=float, default=10000.0)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=500)
    p.add_argument("--target-update-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=0,
                   help="0 disables full-horizon eval in smoke mode")
    p.add_argument(
        "--log-path",
        default=str(LOG_DIR / "dqn_smoke.log"),
    )
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> DQNTrainConfig:
    return DQNTrainConfig(
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
        eval_evaluation_days=args.evaluation_days,
        eval_warmup_days=args.warmup_days,
        eval_num_seeds=1,
        checkpoint_path=None,
    )


def write_log(path: Path, cfg: DQNTrainConfig, summary: Dict[str, object]) -> None:
    lines = [
        "DQN smoke training",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"profile: {cfg.disruption_matrix_profile}",
        f"mode: {cfg.disruption_mode}",
        f"warmup_days: {cfg.warmup_days}",
        f"evaluation_days: {cfg.evaluation_days}",
        f"episodes: {cfg.episodes}",
        f"reward_scale: {cfg.reward_scale}",
        f"gamma: {cfg.gamma}  lr: {cfg.lr}",
        f"batch_size: {cfg.batch_size}  "
        f"start_learning_after: {cfg.start_learning_after}",
        f"eps: {cfg.eps_start} → {cfg.eps_end} over {cfg.eps_decay_steps} steps",
        f"seed: {cfg.seed}  device: {cfg.device}",
        "",
        f"global_steps: {summary['global_steps']}",
        f"updates: {summary['updates']}",
        f"elapsed_sec: {summary['elapsed_sec']:.2f}",
        f"buffer_size: {summary['buffer_size']}",
        f"final_eps: {summary['final_eps']:.4f}",
        "",
        "episode | scaled_return | raw_profit | mean_loss",
    ]
    returns = summary["episode_returns"]
    raws = summary["episode_raw_profits"]
    losses = summary["episode_losses"]
    for i, (ret, raw, loss) in enumerate(zip(returns, raws, losses), start=1):
        loss_s = f"{loss:.5f}" if loss == loss else "nan"
        lines.append(f"{i:7d} | {ret:13.3f} | {raw:10.1f} | {loss_s}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)
    print(
        f"DQN smoke | profile={cfg.disruption_matrix_profile} "
        f"eval_days={cfg.evaluation_days} episodes={cfg.episodes} "
        f"device={cfg.device}",
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
        f"updates={summary['updates']} | log={log_path}",
        flush=True,
    )


# Back-compat alias used by older tests
SmokeConfig = DQNTrainConfig


def run_smoke_training(cfg: DQNTrainConfig) -> Dict[str, object]:
    return run_dqn_training(cfg)


if __name__ == "__main__":
    main()
