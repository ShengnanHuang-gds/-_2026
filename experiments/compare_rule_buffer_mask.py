#!/usr/bin/env python3
"""
Compare RL with vs without rule-buffer mask on harsh_strong.

Rule-buffer mask:
  if severe (A>=2 or U>=2) OR min CW/FW DOS < threshold:
      only ell=2 is legal
  else:
      ell in {0,1,2} (still forbid e=1)

Default: DDQN + PPO, ± warmstart, ± rule mask, 30-rep eval.
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
    p.add_argument("--algos", default="ddqn,ppo")
    p.add_argument("--dos-threshold", type=float, default=1.0)
    p.add_argument("--skip-train", action="store_true")
    return p.parse_args(argv)


def ckpt_path(profile: str, algo: str, warm: bool, rule_mask: bool) -> Path:
    ws = "ws" if warm else "no_ws"
    rm = "rulemask" if rule_mask else "nomask"
    return CKPT_DIR / f"ddqn_{profile}_raw94_{algo}_{ws}_{rm}_best.pt"


def make_cfg(
    algo: str,
    warm: bool,
    rule_mask: bool,
    args: argparse.Namespace,
):
    common = dict(
        state_encoder="raw",
        action_space="coarse",
        disruption_mode="both",
        disruption_matrix_profile=args.profile,
        demand_intensity_scale=1.2,
        forbid_expedite=True,
        rule_buffer_mask=rule_mask,
        dos_threshold=args.dos_threshold,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        eval_every_episodes=5,
        eval_num_seeds=3,
        log_every_episode=10,
        checkpoint_path=str(ckpt_path(args.profile, algo, warm, rule_mask)),
    )
    if algo == "ppo":
        if warm:
            return PPOTrainConfig(
                algorithm="ppo",
                lr=3e-4,
                rule_warmstart_episodes=20,
                rule_bc_updates=2000,
                rule_td_pre_updates=0,
                entropy_coef=0.005,
                **common,
            )
        return PPOTrainConfig(
            algorithm="ppo",
            lr=3e-4,
            rule_warmstart_episodes=0,
            rule_bc_updates=0,
            rule_td_pre_updates=0,
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
def diagnose_ell2(
    net,
    cfg,
    *,
    n_rep: int,
    seed0: int,
    device: torch.device,
) -> Dict[str, float]:
    n_sev = 0
    ell2 = 0
    n_steps = 0
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
            act = decode_action(action, space=cfg.action_space)
            a = int(state.supplier_availability)
            u = int(state.transport_state)
            if a >= 2 or u >= 2:
                n_sev += 1
                if act.ell == 2:
                    ell2 += 1
            n_steps += 1
            state, _, _, _ = env.step(action)
    return {
        "ell2_given_severe": ell2 / n_sev if n_sev else float("nan"),
        "n_severe": float(n_sev),
        "n_steps": float(n_steps),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    algos = [a.strip().lower() for a in args.algos.split(",") if a.strip()]
    device = torch.device(args.device)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    jobs: List[Tuple[str, bool, bool, object]] = []
    for algo in algos:
        for warm in (False, True):
            for rule_mask in (False, True):
                jobs.append((algo, warm, rule_mask, make_cfg(algo, warm, rule_mask, args)))

    for algo, warm, rule_mask, cfg in jobs:
        tag = f"{algo}/{'ws' if warm else 'no_ws'}/{'rulemask' if rule_mask else 'nomask'}"
        path = Path(cfg.checkpoint_path)
        if args.skip_train and path.is_file():
            print(f"[skip-train] {tag}", flush=True)
            continue
        print(f"\n===== TRAIN {tag} =====", flush=True)
        if algo == "ppo":
            result = run_ppo_training(cfg)
        else:
            result = run_dqn_training(cfg)
        print(f"[{tag}] best_eval={result.get('best_eval_profit')}", flush=True)

    # Baselines
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
    for algo, warm, rule_mask, cfg in jobs:
        tag = f"{algo}/{'ws' if warm else 'no_ws'}/{'rulemask' if rule_mask else 'nomask'}"
        print(f"\n===== EVAL {tag} =====", flush=True)
        net = load_net(algo, cfg, device)
        _, profits, _ = evaluate_ddqn_greedy(
            net, cfg, num_reps=args.n_rep, seed_base=args.seed, device=device
        )
        diag = diagnose_ell2(
            net, cfg, n_rep=min(10, args.n_rep), seed0=args.seed, device=device
        )
        m = float(np.mean(profits))
        ci = _ci95(profits)
        print(
            f"  {tag}: {m:.2f} ± {ci:.2f} | ell2|sev={diag['ell2_given_severe']:.1%} "
            f"| Δrule={m - rule_mean:+.0f} Δbest={m - best_mean:+.0f}",
            flush=True,
        )
        rows.append(
            {
                "algo": algo,
                "warm": warm,
                "rule_mask": rule_mask,
                "mean": m,
                "ci": ci,
                "ell2_sev": diag["ell2_given_severe"],
            }
        )

    lines = [
        "Rule-buffer mask ablation (harsh_strong, raw94, forbid_e1)",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"dos_threshold={args.dos_threshold}",
        f"episodes={args.episodes} n_rep={args.n_rep}",
        "",
        "Rule: severe OR min DOS < threshold → only ell=2 legal.",
        "",
        f"{'algo':<8} {'ws':>3} {'mask':>8} {'profit':>12} {'±CI':>10} "
        f"{'ell2|sev':>10} {'Δrule':>10} {'Δbest':>10}",
    ]
    for r in rows:
        lines.append(
            f"{r['algo']:<8} {'Y' if r['warm'] else 'N':>3} "
            f"{'ON' if r['rule_mask'] else 'OFF':>8} "
            f"{r['mean']:12.2f} {r['ci']:10.2f} {r['ell2_sev']:10.1%} "
            f"{r['mean'] - rule_mean:+10.2f} {r['mean'] - best_mean:+10.2f}"
        )
    lines.append("")
    lines.append(
        f"baselines: rule={rule_mean:.2f}±{_ci95(basel['rule_based']):.2f} | "
        f"{best_name}={best_mean:.2f}±{_ci95(basel[best_name]):.2f}"
    )
    # Mask lift
    lines.append("")
    lines.append("Mask lift (ON − OFF) by algo/ws:")
    keyed = {(r["algo"], r["warm"], r["rule_mask"]): r for r in rows}
    for algo in algos:
        for warm in (False, True):
            off = keyed.get((algo, warm, False))
            on = keyed.get((algo, warm, True))
            if off and on:
                lines.append(
                    f"  {algo}/{'ws' if warm else 'no_ws'}: "
                    f"{on['mean'] - off['mean']:+.2f} "
                    f"(ell2|sev {off['ell2_sev']:.0%}→{on['ell2_sev']:.0%})"
                )

    text = "\n".join(lines) + "\n"
    log_path = LOG_DIR / f"rule_buffer_mask_ablation_{args.profile}_{args.n_rep}rep.log"
    log_path.write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)
    print(f"log → {log_path}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
