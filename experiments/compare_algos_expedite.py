#!/usr/bin/env python3
"""
Compare RL algorithms under forbid_e1 vs allow_e1.

Default: dqn/ddqn/dueling/d3qn/ppo × {forbid, allow} with rule warmstart,
raw94, coarse, harsh_strong, 50 episodes, 30-rep greedy eval.

Reports profit + greedy_e1 + train sample_e1 + e1_legal_days.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
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
from mdp.actions import decode_action, num_actions_for
from mdp.policies import make_baseline_policies
from mdp.state_encoder import build_state_encoder_from_env
from rl.dqn import build_q_network, states_to_tensor
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
    p.add_argument("--algos", default=",".join(ALL_ALGOS))
    p.add_argument(
        "--warmstart",
        type=int,
        default=1,
        choices=(0, 1),
        help="0=no ws, 1=rule warmstart (default)",
    )
    p.add_argument("--skip-train", action="store_true")
    return p.parse_args(argv)


def ckpt_path(profile: str, algo: str, forbid: bool, warm: bool) -> Path:
    e = "forbid_e1" if forbid else "allow_e1"
    ws = "ws" if warm else "no_ws"
    return CKPT_DIR / f"{algo}_{profile}_raw94_{ws}_{e}_ablation_best.pt"


def make_cfg(
    algo: str,
    forbid: bool,
    warm: bool,
    args: argparse.Namespace,
):
    common = dict(
        state_encoder="raw",
        action_space="coarse",
        disruption_mode="both",
        disruption_matrix_profile=args.profile,
        demand_intensity_scale=1.2,
        forbid_expedite=forbid,
        rule_buffer_mask=False,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        eval_every_episodes=5,
        eval_num_seeds=3,
        log_every_episode=10,
        checkpoint_path=str(
            ckpt_path(args.profile, algo, forbid, warm)
        ),
    )
    if algo == "ppo":
        if warm:
            return PPOTrainConfig(
                algorithm="ppo",
                lr=3e-4,
                rule_warmstart_episodes=20,
                rule_bc_updates=2000,
                rule_td_pre_updates=0,
                entropy_coef=0.01,
                **common,
            )
        return PPOTrainConfig(
            algorithm="ppo",
            lr=3e-4,
            rule_warmstart_episodes=0,
            rule_bc_updates=0,
            rule_td_pre_updates=0,
            entropy_coef=0.01,
            **common,
        )
    if warm:
        return DQNTrainConfig(
            algorithm=algo,
            lr=3e-4,
            eps_start=0.2,
            eps_end=0.01,
            eps_decay_steps=10_000,
            start_learning_after=0,
            rule_warmstart_episodes=20,
            rule_bc_updates=2000,
            rule_td_pre_updates=500,
            **common,
        )
    return DQNTrainConfig(
        algorithm=algo,
        lr=3e-4,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay_steps=15_000,
        start_learning_after=1_000,
        rule_warmstart_episodes=0,
        rule_bc_updates=0,
        rule_td_pre_updates=0,
        **common,
    )


def load_net(algo: str, cfg, device: torch.device):
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
    return net


@torch.no_grad()
def diagnose_e1(
    net,
    cfg,
    *,
    n_rep: int,
    seed0: int,
    device: torch.device,
) -> Dict[str, float]:
    e1 = 0
    n = 0
    cost = 0.0
    for r in range(n_rep):
        seed = seed0 + r
        env = build_env(cfg, seed=seed)
        state = env.reset(seed=seed)
        enc = build_state_encoder_from_env(env, encoder_type=cfg.state_encoder)
        while not env.done:
            mask = env.get_action_mask()
            vec = enc.encode(state)
            action = net.select_greedy(
                states_to_tensor(vec, device=device),
                action_mask=torch.as_tensor(mask, dtype=torch.bool, device=device),
            )
            if decode_action(action, space=cfg.action_space).expedite == 1:
                e1 += 1
            n += 1
            state, _, _, info = env.step(action)
            cost += float(info.get("expedite_cost", 0.0) or 0.0)
    return {
        "e1_ratio": e1 / n if n else float("nan"),
        "expedite_cost_per_ep": cost / n_rep if n_rep else float("nan"),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    algos = [a.strip().lower() for a in args.algos.split(",") if a.strip()]
    warm = bool(args.warmstart)
    device = torch.device(args.device)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    jobs: List[Tuple[str, bool, object]] = []
    for algo in algos:
        for forbid in (True, False):
            jobs.append((algo, forbid, make_cfg(algo, forbid, warm, args)))

    train_stats: Dict[Tuple[str, bool], Dict[str, float]] = {}
    for algo, forbid, cfg in jobs:
        tag = f"{algo}/{'forbid' if forbid else 'allow'}"
        path = Path(cfg.checkpoint_path)
        if args.skip_train and path.is_file():
            print(f"[skip-train] {tag}", flush=True)
            train_stats[(algo, forbid)] = {
                "sample_e1": float("nan"),
                "e1_legal": float("nan"),
            }
            continue
        print(f"\n===== TRAIN {tag} =====", flush=True)
        if algo == "ppo":
            result = run_ppo_training(cfg)
        else:
            result = run_dqn_training(cfg)
        train_stats[(algo, forbid)] = {
            "sample_e1": float(result["mean_train_e1_ratio"]),
            "e1_legal": float(result["mean_train_e1_legal_ratio"]),
        }
        print(
            f"[{tag}] best_eval={result.get('best_eval_profit')} "
            f"train_e1={train_stats[(algo, forbid)]['sample_e1']:.1%} "
            f"e1_legal={train_stats[(algo, forbid)]['e1_legal']:.1%}",
            flush=True,
        )

    sim = SimulationConfig(
        disruption_mode="both",
        disruption_matrix_profile=args.profile,
        warmup_days=30,
        evaluation_days=365,
        num_replications=args.n_rep,
        demand_intensity_scale=1.2,
        random_seed_base=args.seed,
    )
    print(f"\n===== baselines n_rep={args.n_rep} =====", flush=True)
    basel: Dict[str, List[float]] = {}
    for policy in make_baseline_policies(args.profile):
        _, profits = evaluate_baseline(sim, policy)
        basel[policy.name] = profits
        print(
            f"  {policy.name}: {float(np.mean(profits)):.2f} ± {_ci95(profits):.2f}",
            flush=True,
        )
    rule_mean = float(np.mean(basel["rule_based"]))
    best_name = [k for k in basel if k.startswith("best_fixed")][0]
    best_mean = float(np.mean(basel[best_name]))

    rows = []
    for algo, forbid, cfg in jobs:
        tag = f"{algo}/{'forbid' if forbid else 'allow'}"
        print(f"\n===== EVAL {tag} =====", flush=True)
        net = load_net(algo, cfg, device)
        _, profits, _ = evaluate_ddqn_greedy(
            net, cfg, num_reps=args.n_rep, seed_base=args.seed, device=device
        )
        diag = diagnose_e1(
            net, cfg, n_rep=min(10, args.n_rep), seed0=args.seed, device=device
        )
        ts = train_stats[(algo, forbid)]
        m = float(np.mean(profits))
        ci = _ci95(profits)
        print(
            f"  {tag}: {m:.2f} ± {ci:.2f} | greedy_e1={diag['e1_ratio']:.1%} "
            f"| train_e1={ts['sample_e1']:.1%} | e1_legal={ts['e1_legal']:.1%} "
            f"| Δbest={m - best_mean:+.0f}",
            flush=True,
        )
        rows.append(
            {
                "algo": algo,
                "forbid": forbid,
                "mean": m,
                "ci": ci,
                "greedy_e1": diag["e1_ratio"],
                "train_e1": ts["sample_e1"],
                "e1_legal": ts["e1_legal"],
                "exp_cost": diag["expedite_cost_per_ep"],
            }
        )

    lines = [
        "RL algos × forbid_e1 vs allow_e1",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"profile={args.profile} raw94 coarse rule_ws={warm} "
        f"episodes={args.episodes} n_rep={args.n_rep}",
        "",
        f"{'algo':<8} {'e1':>6} {'profit':>12} {'±CI':>10} "
        f"{'greedy_e1':>10} {'train_e1':>10} {'e1_legal':>10} "
        f"{'Δrule':>10} {'Δbest':>10}",
    ]
    for r in rows:
        te = (
            f"{r['train_e1']:10.1%}"
            if r["train_e1"] == r["train_e1"]
            else f"{'n/a':>10}"
        )
        tl = (
            f"{r['e1_legal']:10.1%}"
            if r["e1_legal"] == r["e1_legal"]
            else f"{'n/a':>10}"
        )
        lines.append(
            f"{r['algo']:<8} {'forbid' if r['forbid'] else 'allow':>6} "
            f"{r['mean']:12.2f} {r['ci']:10.2f} {r['greedy_e1']:10.1%} "
            f"{te} {tl} "
            f"{r['mean'] - rule_mean:+10.2f} {r['mean'] - best_mean:+10.2f}"
        )

    lines.append("")
    lines.append("Allow − forbid lift by algo:")
    keyed = {(r["algo"], r["forbid"]): r for r in rows}
    for algo in algos:
        off = keyed.get((algo, True))
        on = keyed.get((algo, False))
        if off and on:
            lines.append(
                f"  {algo}: profit {on['mean'] - off['mean']:+.2f} | "
                f"greedy_e1 {off['greedy_e1']:.1%}→{on['greedy_e1']:.1%} | "
                f"train_e1 {off['train_e1']:.1%}→{on['train_e1']:.1%} | "
                f"e1_legal {off['e1_legal']:.1%}→{on['e1_legal']:.1%}"
            )

    lines.append("")
    lines.append(
        f"baselines: rule={rule_mean:.2f}±{_ci95(basel['rule_based']):.2f} | "
        f"{best_name}={best_mean:.2f}±{_ci95(basel[best_name]):.2f}"
    )

    text = "\n".join(lines) + "\n"
    log_path = (
        LOG_DIR
        / f"algo_expedite_ablation_{args.profile}_ws{int(warm)}_{args.n_rep}rep.log"
    )
    log_path.write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)
    print(f"log → {log_path}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
