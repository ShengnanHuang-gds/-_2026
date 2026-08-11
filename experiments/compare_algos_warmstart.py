#!/usr/bin/env python3
"""
Compare DQN / DDQN / Dueling / D3QN / PPO × {no warmstart, rule warmstart}.

Fixed setting (per user request):
  - state encoder: raw 94-d
  - action space: coarse (ell,e,g)=18
  - forbid_expedite: True
  - profile: harsh_strong (default)
  - demand_intensity_scale: 1.2

Example::

    python experiments/compare_algos_warmstart.py
    python experiments/compare_algos_warmstart.py --n-rep 10 --episodes 30
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.simulation_config import SimulationConfig
from experiments.compare_ddqn_baselines import (
    evaluate_baseline,
    evaluate_ddqn_greedy,
)
from mdp.actions import num_actions_for
from mdp.policies import make_baseline_policies
from mdp.state_encoder import build_state_encoder_from_env
from rl.dqn import build_q_network
from rl.dqn_train import DQNTrainConfig, build_env, run_dqn_training
from rl.ppo import build_actor_critic
from rl.ppo_train import PPOTrainConfig, run_ppo_training

LOG_DIR = ROOT / "logs"
CKPT_DIR = LOG_DIR / "checkpoints"

VALUE_ALGOS = ("dqn", "ddqn", "dueling", "d3qn")
ALL_ALGOS = VALUE_ALGOS + ("ppo",)


def _ci95(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    return 1.96 * float(np.std(xs, ddof=1)) / math.sqrt(n)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default="harsh_strong")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--n-rep", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--algos",
        default=",".join(ALL_ALGOS),
        help="comma list: dqn,ddqn,dueling,d3qn,ppo",
    )
    p.add_argument(
        "--warmstarts",
        default="0,1",
        help="comma list of 0/1 for no-ws / rule-ws",
    )
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--log-path", default="")
    return p.parse_args(argv)


def ckpt_path(profile: str, algo: str, warm: bool) -> Path:
    tag = "rule_ws" if warm else "no_ws"
    return CKPT_DIR / f"ddqn_{profile}_raw94_{algo}_{tag}_best.pt"


def make_value_cfg(
    algo: str,
    warm: bool,
    args: argparse.Namespace,
) -> DQNTrainConfig:
    if warm:
        return DQNTrainConfig(
            algorithm=algo,
            state_encoder="raw",
            action_space="coarse",
            disruption_matrix_profile=args.profile,
            demand_intensity_scale=1.2,
            forbid_expedite=True,
            episodes=args.episodes,
            lr=3e-4,
            eps_start=0.2,
            eps_end=0.01,
            eps_decay_steps=10_000,
            start_learning_after=0,
            seed=args.seed,
            device=args.device,
            eval_every_episodes=5,
            eval_num_seeds=3,
            log_every_episode=10,
            checkpoint_path=str(ckpt_path(args.profile, algo, True)),
            rule_warmstart_episodes=20,
            rule_bc_updates=2000,
            rule_td_pre_updates=500,
            rule_bc_lr=1e-3,
        )
    return DQNTrainConfig(
        algorithm=algo,
        state_encoder="raw",
        action_space="coarse",
        disruption_matrix_profile=args.profile,
        demand_intensity_scale=1.2,
        forbid_expedite=True,
        episodes=args.episodes,
        lr=3e-4,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay_steps=15_000,
        start_learning_after=1_000,
        seed=args.seed,
        device=args.device,
        eval_every_episodes=5,
        eval_num_seeds=3,
        log_every_episode=10,
        checkpoint_path=str(ckpt_path(args.profile, algo, False)),
        rule_warmstart_episodes=0,
        rule_bc_updates=0,
        rule_td_pre_updates=0,
    )


def make_ppo_cfg(warm: bool, args: argparse.Namespace) -> PPOTrainConfig:
    base = dict(
        state_encoder="raw",
        action_space="coarse",
        disruption_matrix_profile=args.profile,
        demand_intensity_scale=1.2,
        forbid_expedite=True,
        episodes=args.episodes,
        lr=3e-4,
        seed=args.seed,
        device=args.device,
        eval_every_episodes=5,
        eval_num_seeds=3,
        log_every_episode=10,
        checkpoint_path=str(ckpt_path(args.profile, "ppo", warm)),
        ppo_epochs=4,
        ppo_minibatch_size=256,
        clip_eps=0.2,
        entropy_coef=0.01 if not warm else 0.005,
    )
    if warm:
        return PPOTrainConfig(
            algorithm="ppo",
            rule_warmstart_episodes=20,
            rule_bc_updates=2000,
            rule_bc_lr=1e-3,
            rule_td_pre_updates=0,
            **base,
        )
    return PPOTrainConfig(
        algorithm="ppo",
        rule_warmstart_episodes=0,
        rule_bc_updates=0,
        rule_td_pre_updates=0,
        **base,
    )


def load_policy(algo: str, cfg, device: torch.device):
    probe = build_env(cfg, seed=cfg.seed)
    probe.reset(seed=cfg.seed)
    enc = build_state_encoder_from_env(probe, encoder_type=cfg.state_encoder)
    n_act = num_actions_for(cfg.action_space)
    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    if algo == "ppo":
        net = build_actor_critic(
            state_dim=enc.dim, num_actions=n_act, device=device
        )
    else:
        net = build_q_network(
            state_dim=enc.dim,
            num_actions=n_act,
            device=device,
            algorithm=algo,
        )
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net, ckpt.get("best_eval_profit")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    algos = [a.strip().lower() for a in args.algos.split(",") if a.strip()]
    warms = []
    for w in args.warmstarts.split(","):
        w = w.strip()
        if w in ("0", "false", "no"):
            warms.append(False)
        elif w in ("1", "true", "yes"):
            warms.append(True)
        else:
            raise SystemExit(f"bad warmstart flag {w!r}")

    for a in algos:
        if a not in ALL_ALGOS:
            raise SystemExit(f"unknown algo {a}; expected {ALL_ALGOS}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    log_path = Path(
        args.log_path
        or (
            LOG_DIR
            / f"algo_warmstart_raw94_{args.profile}_{args.n_rep}rep.log"
        )
    )

    jobs: List[Tuple[str, bool, object]] = []
    for algo in algos:
        for warm in warms:
            if algo == "ppo":
                jobs.append((algo, warm, make_ppo_cfg(warm, args)))
            else:
                jobs.append((algo, warm, make_value_cfg(algo, warm, args)))

    # ---- train ----
    train_meta: Dict[Tuple[str, bool], dict] = {}
    for algo, warm, cfg in jobs:
        tag = f"{algo}/{'ws' if warm else 'no_ws'}"
        path = Path(cfg.checkpoint_path)
        if args.skip_train and path.is_file():
            print(f"[skip-train] {tag} → {path}", flush=True)
            train_meta[(algo, warm)] = {"cfg": cfg, "best": None}
            continue
        print(f"\n===== TRAIN {tag} =====", flush=True)
        t0 = time.time()
        if algo == "ppo":
            result = run_ppo_training(cfg)
        else:
            result = run_dqn_training(cfg)
        print(
            f"[{tag}] best_eval={result.get('best_eval_profit')} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        train_meta[(algo, warm)] = {
            "cfg": cfg,
            "best": result.get("best_eval_profit"),
            "warmstart": result.get("warmstart"),
        }

    # ---- baselines once ----
    sim = SimulationConfig(
        disruption_mode="both",
        disruption_matrix_profile=args.profile,
        warmup_days=30,
        evaluation_days=365,
        num_replications=args.n_rep,
        demand_intensity_scale=1.2,
        random_seed_base=args.seed,
    )
    print(f"\n===== {args.n_rep}-rep baselines =====", flush=True)
    baseline_profits: Dict[str, List[float]] = {}
    for policy in make_baseline_policies(args.profile):
        _, profits = evaluate_baseline(sim, policy)
        baseline_profits[policy.name] = profits
        print(
            f"  {policy.name}: {float(np.mean(profits)):.2f} ± {_ci95(profits):.2f}",
            flush=True,
        )

    # ---- eval ----
    rows = []
    for algo, warm, cfg in jobs:
        tag = f"{algo}/{'ws' if warm else 'no_ws'}"
        print(f"\n===== EVAL {tag} =====", flush=True)
        net, train_best = load_policy(algo, cfg, device)
        _, profits, _ = evaluate_ddqn_greedy(
            net,
            cfg,
            num_reps=args.n_rep,
            seed_base=args.seed,
            device=device,
        )
        m = float(np.mean(profits))
        ci = _ci95(profits)
        print(f"  {tag} mean={m:.2f} ± {ci:.2f}", flush=True)
        rows.append(
            {
                "algo": algo,
                "warm": warm,
                "mean": m,
                "ci": ci,
                "train_best": train_best,
                "profits": profits,
            }
        )

    rule_mean = float(np.mean(baseline_profits["rule_based"]))
    best_name = [k for k in baseline_profits if k.startswith("best_fixed")][0]
    fixed0_name = [k for k in baseline_profits if k.startswith("fixed_0")][0]
    best_mean = float(np.mean(baseline_profits[best_name]))
    fixed0_mean = float(np.mean(baseline_profits[fixed0_name]))

    lines = [
        "Algorithm × warmstart comparison (raw 94-d state, coarse actions)",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"profile: {args.profile}",
        f"state_encoder: raw (94)",
        f"action_space: coarse (18; forbid_e1 → 9 legal)",
        f"episodes: {args.episodes}",
        f"n_rep: {args.n_rep} seeds {args.seed}..{args.seed + args.n_rep - 1}",
        f"demand_intensity_scale: 1.2",
        "",
        "Warmstart = 20 rule episodes + BC 2000 (+ TD-pre 500 for value methods).",
        "No-ws = learn from scratch (ε: 1→0.01 for value methods).",
        "",
        f"{'algo':<10} {'ws':>3} {'profit':>12} {'±CI':>10} {'Δrule':>10} {'Δbest':>10}",
    ]
    for row in rows:
        ws = "Y" if row["warm"] else "N"
        lines.append(
            f"{row['algo']:<10} {ws:>3} {row['mean']:12.2f} {row['ci']:10.2f} "
            f"{row['mean'] - rule_mean:+10.2f} {row['mean'] - best_mean:+10.2f}"
        )
    lines.append("")
    lines.append(
        f"baselines: rule={rule_mean:.2f}±{_ci95(baseline_profits['rule_based']):.2f} | "
        f"{best_name}={best_mean:.2f}±{_ci95(baseline_profits[best_name]):.2f} | "
        f"{fixed0_name}={fixed0_mean:.2f}±{_ci95(baseline_profits[fixed0_name]):.2f}"
    )

    # Pivot: warmstart lift per algo
    lines.append("")
    lines.append("Warmstart lift (ws − no_ws):")
    by_algo = {}
    for row in rows:
        by_algo.setdefault(row["algo"], {})[row["warm"]] = row["mean"]
    for algo in algos:
        if True in by_algo.get(algo, {}) and False in by_algo.get(algo, {}):
            d = by_algo[algo][True] - by_algo[algo][False]
            lines.append(f"  {algo:<10} {d:+.2f}")

    text = "\n".join(lines) + "\n"
    log_path.write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)
    print(f"log → {log_path}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
