#!/usr/bin/env python3
"""Compare coarse (ell,e,g) vs fine (ell,focus,e,g) DDQN under forbid_e1.

Plan A: global buffer + local focus. Fine space has 108 actions
(54 legal when e=1 is forbidden). focus=none recovers coarse policies.

Default: harsh_strong, risk encoder, 50 episodes, 30-rep eval.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
from rl.dqn import build_q_network
from rl.dqn_train import DQNTrainConfig, build_env, run_dqn_training

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
    p.add_argument("--encoder", default="risk", choices=["raw", "risk"])
    p.add_argument("--spaces", default="coarse,fine")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--log-path", default="")
    return p.parse_args(argv)


def make_cfg(space: str, args: argparse.Namespace) -> DQNTrainConfig:
    ckpt = str(
        CKPT_DIR
        / f"ddqn_{args.profile}_forbid_e1_{args.encoder}_{space}_best.pt"
    )
    return DQNTrainConfig(
        algorithm="ddqn",
        state_encoder=args.encoder,
        action_space=space,
        disruption_matrix_profile=args.profile,
        warmup_days=30,
        evaluation_days=365,
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
        checkpoint_path=ckpt,
    )


def load_online(cfg: DQNTrainConfig, device: torch.device):
    probe = build_env(cfg, seed=cfg.seed)
    probe.reset(seed=cfg.seed)
    enc = build_state_encoder_from_env(probe, encoder_type=cfg.state_encoder)
    n_act = num_actions_for(cfg.action_space)
    online = build_q_network(
        state_dim=enc.dim, num_actions=n_act, device=device
    )
    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    online.load_state_dict(ckpt["model_state_dict"])
    online.eval()
    return online, enc.dim, n_act, ckpt.get("best_eval_profit")


@torch.no_grad()
def focus_usage(
    online,
    cfg: DQNTrainConfig,
    *,
    reps: int,
    seed_base: int,
    device: torch.device,
) -> Dict[str, float]:
    from mdp.state_encoder import build_state_encoder_from_env
    from rl.dqn import states_to_tensor

    focus_counts: Counter = Counter()
    ell_sev = Counter()
    n_sev = 0
    n_steps = 0
    online.eval()
    for r in range(reps):
        seed = seed_base + r
        env = build_env(cfg, seed=seed)
        state = env.reset(seed=seed)
        encoder = build_state_encoder_from_env(
            env, encoder_type=cfg.state_encoder
        )
        while not env.done:
            mask = env.get_action_mask()
            vec = encoder.encode(state)
            action = online.select_greedy(
                states_to_tensor(vec, device=device),
                action_mask=torch.as_tensor(
                    mask, dtype=torch.bool, device=device
                ),
            )
            act = decode_action(action, space=cfg.action_space)
            focus_counts[act.focus] += 1
            n_steps += 1
            a = int(state.supplier_availability)
            u = int(state.transport_state)
            if a >= 2 or u >= 2:
                n_sev += 1
                ell_sev[act.ell] += 1
            state, _, _, _ = env.step(action)
    non_none = sum(c for f, c in focus_counts.items() if f != 0)
    return {
        "focus_non_none_rate": non_none / max(1, n_steps),
        "ell2_given_severe": (
            ell_sev[2] / n_sev if n_sev else float("nan")
        ),
        "n_severe": float(n_sev),
        "n_steps": float(n_steps),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    spaces = [s.strip().lower() for s in args.spaces.split(",") if s.strip()]
    device = torch.device(args.device)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = Path(
        args.log_path
        or (
            LOG_DIR
            / f"ddqn_action_coarse_vs_fine_{args.profile}_{args.n_rep}rep.log"
        )
    )

    cfgs: Dict[str, DQNTrainConfig] = {}
    for space in spaces:
        cfg = make_cfg(space, args)
        cfgs[space] = cfg
        if args.skip_train and Path(cfg.checkpoint_path).is_file():
            print(f"[skip-train] {space} → {cfg.checkpoint_path}", flush=True)
            continue
        print(f"\n===== TRAIN action_space={space} =====", flush=True)
        result = run_dqn_training(cfg)
        print(
            f"[{space}] best_eval={result['best_eval_profit']:.2f}",
            flush=True,
        )

    # Baselines on coarse (focus=none isomorphic)
    sim = SimulationConfig(
        disruption_mode="both",
        disruption_matrix_profile=args.profile,
        warmup_days=30,
        evaluation_days=365,
        num_replications=args.n_rep,
        demand_intensity_scale=1.2,
        random_seed_base=args.seed,
    )
    print(f"\n===== {args.n_rep}-rep baselines (coarse) =====", flush=True)
    baseline_profits: Dict[str, List[float]] = {}
    for policy in make_baseline_policies(args.profile, action_space="coarse"):
        _, profits = evaluate_baseline(sim, policy)
        baseline_profits[policy.name] = profits
        print(
            f"  {policy.name}: {float(np.mean(profits)):.2f} ± {_ci95(profits):.2f}",
            flush=True,
        )

    rows = []
    for space, cfg in cfgs.items():
        online, dim, n_act, train_best = load_online(cfg, device)
        print(
            f"\n===== EVAL action_space={space} dim={dim} n_act={n_act} =====",
            flush=True,
        )
        _, profits, _ = evaluate_ddqn_greedy(
            online,
            cfg,
            num_reps=args.n_rep,
            seed_base=args.seed,
            device=device,
        )
        usage = focus_usage(
            online,
            cfg,
            reps=min(10, args.n_rep),
            seed_base=args.seed,
            device=device,
        )
        m = float(np.mean(profits))
        ci = _ci95(profits)
        print(
            f"  DDQN({space}) mean={m:.2f} ± {ci:.2f} "
            f"focus≠none={usage['focus_non_none_rate']:.1%} "
            f"ell2|sev={usage['ell2_given_severe']:.1%}",
            flush=True,
        )
        rows.append(
            {
                "space": space,
                "dim": dim,
                "n_act": n_act,
                "mean": m,
                "ci": ci,
                "usage": usage,
                "train_best": train_best,
            }
        )

    rule_mean = float(np.mean(baseline_profits["rule_based"]))
    best_name = [k for k in baseline_profits if k.startswith("best_fixed")][0]
    fixed0_name = [k for k in baseline_profits if k.startswith("fixed_0")][0]
    best_mean = float(np.mean(baseline_profits[best_name]))
    fixed0_mean = float(np.mean(baseline_profits[fixed0_name]))

    lines = [
        "Coarse vs Fine (Plan A) action-space comparison",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"profile: {args.profile}",
        f"encoder: {args.encoder}",
        f"forbid_expedite: True",
        f"episodes: {args.episodes}",
        f"n_rep: {args.n_rep} seeds {args.seed}..{args.seed + args.n_rep - 1}",
        "",
        f"{'space':<8} {'nA':>4} {'DDQN':>12} {'±CI':>10} {'Δrule':>10} "
        f"{'Δbest':>10} {'focus≠0':>8} {'ell2|sev':>10}",
    ]
    for row in rows:
        u = row["usage"]
        lines.append(
            f"{row['space']:<8} {row['n_act']:4d} {row['mean']:12.2f} "
            f"{row['ci']:10.2f} {row['mean'] - rule_mean:+10.2f} "
            f"{row['mean'] - best_mean:+10.2f} "
            f"{u['focus_non_none_rate']:8.1%} {u['ell2_given_severe']:10.1%}"
        )
    lines.append("")
    lines.append(
        f"baselines: rule={rule_mean:.2f}±{_ci95(baseline_profits['rule_based']):.2f} | "
        f"{best_name}={best_mean:.2f}±{_ci95(baseline_profits[best_name]):.2f} | "
        f"{fixed0_name}={fixed0_mean:.2f}±{_ci95(baseline_profits[fixed0_name]):.2f}"
    )
    if len(rows) == 2:
        d = rows[1]["mean"] - rows[0]["mean"]
        lines.append(
            f"Fine − Coarse profit delta: {d:+.2f} "
            f"({100.0 * d / abs(rows[0]['mean']):+.2f}% of Coarse)"
        )
    lines.append(
        "Note: fine focus=none recovers coarse (ell,e,g); "
        "local focus can raise only tight SKUs/FWs."
    )

    text = "\n".join(lines) + "\n"
    log_path.write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)
    print(f"log → {log_path}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
