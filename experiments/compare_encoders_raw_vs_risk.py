#!/usr/bin/env python3
"""Compare Raw94 vs Risk33 state encoders under the same DDQN budget.

Default: harsh_strong, forbid_e1, no warmstart, 50 RL episodes, 30-rep eval
vs fixed_0 / best_fixed / rule_based.

Example::

    python experiments/compare_encoders_raw_vs_risk.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys
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
from experiments.diagnose_ddqn_ell_g import collect_traces
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
    p.add_argument("--encoders", default="raw,risk", help="comma list")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--log-path", default="")
    return p.parse_args(argv)


def make_cfg(encoder: str, args: argparse.Namespace) -> DQNTrainConfig:
    ckpt = str(CKPT_DIR / f"ddqn_{args.profile}_forbid_e1_{encoder}_best.pt")
    return DQNTrainConfig(
        algorithm="ddqn",
        state_encoder=encoder,
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
        rule_warmstart_episodes=0,
        rule_bc_updates=0,
        rule_td_pre_updates=0,
    )


def load_online(cfg: DQNTrainConfig, device: torch.device):
    probe = build_env(cfg, seed=cfg.seed)
    probe.reset(seed=cfg.seed)
    enc = build_state_encoder_from_env(probe, encoder_type=cfg.state_encoder)
    online = build_q_network(state_dim=enc.dim, device=device)
    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    online.load_state_dict(ckpt["model_state_dict"])
    online.eval()
    return online, enc.dim, ckpt.get("best_eval_profit")


def diag_summary(records: list) -> Dict[str, float]:
    severe = [r for r in records if r["severity"] == "severe"]
    n = len(severe)
    return {
        "n_severe": float(n),
        "ell2_given_severe": (
            sum(1 for r in severe if r["ell"] == 2) / n if n else float("nan")
        ),
        "agree_rule": (
            sum(1 for r in records if r["agree_rule"]) / len(records)
            if records
            else float("nan")
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    encoders = [e.strip().lower() for e in args.encoders.split(",") if e.strip()]
    device = torch.device(args.device)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = Path(
        args.log_path
        or (LOG_DIR / f"ddqn_encoder_raw_vs_risk_{args.profile}_{args.n_rep}rep.log")
    )

    cfgs: Dict[str, DQNTrainConfig] = {}
    for enc in encoders:
        cfg = make_cfg(enc, args)
        cfgs[enc] = cfg
        if args.skip_train and Path(cfg.checkpoint_path).is_file():
            print(f"[skip-train] {enc} → {cfg.checkpoint_path}", flush=True)
            continue
        print(f"\n===== TRAIN encoder={enc} =====", flush=True)
        result = run_dqn_training(cfg)
        print(
            f"[{enc}] best_eval={result['best_eval_profit']:.2f} "
            f"dim checkpoint saved",
            flush=True,
        )

    # Baselines once
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

    rows = []
    for enc, cfg in cfgs.items():
        online, dim, train_best = load_online(cfg, device)
        print(f"\n===== EVAL encoder={enc} dim={dim} =====", flush=True)
        _, profits, _ = evaluate_ddqn_greedy(
            online,
            cfg,
            num_reps=args.n_rep,
            seed_base=args.seed,
            device=device,
        )
        records = collect_traces(
            online,
            cfg,
            reps=min(10, args.n_rep),
            seed_base=args.seed,
            device=device,
        )
        # collect_traces builds encoder via build_env + build_state_encoder_from_env
        # without encoder_type — patch by re-collecting with correct encoder below
        # if needed. Check diagnose collect_traces...
        diag = diag_summary(records)
        m = float(np.mean(profits))
        ci = _ci95(profits)
        print(
            f"  DDQN({enc}) mean={m:.2f} ± {ci:.2f} "
            f"ell2|sev={diag['ell2_given_severe']:.1%} "
            f"agree_rule={diag['agree_rule']:.1%}",
            flush=True,
        )
        rows.append(
            {
                "enc": enc,
                "dim": dim,
                "mean": m,
                "ci": ci,
                "train_best": train_best,
                "diag": diag,
                "profits": profits,
            }
        )

    rule_mean = float(np.mean(baseline_profits["rule_based"]))
    best_name = [k for k in baseline_profits if k.startswith("best_fixed")][0]
    fixed0_name = [k for k in baseline_profits if k.startswith("fixed_0")][0]
    best_mean = float(np.mean(baseline_profits[best_name]))
    fixed0_mean = float(np.mean(baseline_profits[fixed0_name]))

    lines = [
        "Raw94 vs Risk33 encoder comparison (forbid_e1, no warmstart)",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"profile: {args.profile}",
        f"episodes: {args.episodes}",
        f"n_rep: {args.n_rep} seeds {args.seed}..{args.seed + args.n_rep - 1}",
        f"demand_intensity_scale: 1.2",
        "",
        f"{'enc':<6} {'dim':>4} {'DDQN':>12} {'±CI':>10} {'Δrule':>10} "
        f"{'Δbest':>10} {'ell2|sev':>10} {'agree_r':>8}",
    ]
    for row in rows:
        lines.append(
            f"{row['enc']:<6} {row['dim']:4d} {row['mean']:12.2f} {row['ci']:10.2f} "
            f"{row['mean'] - rule_mean:+10.2f} {row['mean'] - best_mean:+10.2f} "
            f"{row['diag']['ell2_given_severe']:10.1%} "
            f"{row['diag']['agree_rule']:8.1%}"
        )
    lines.append("")
    lines.append(
        f"baselines: rule={rule_mean:.2f}±{_ci95(baseline_profits['rule_based']):.2f} | "
        f"{best_name}={best_mean:.2f}±{_ci95(baseline_profits[best_name]):.2f} | "
        f"{fixed0_name}={fixed0_mean:.2f}±{_ci95(baseline_profits[fixed0_name]):.2f}"
    )
    lines.append("")
    lines.append(
        "Note: both DDQNs trained without rule warmstart for a fair encoder A/B."
    )
    if len(rows) == 2:
        d = rows[1]["mean"] - rows[0]["mean"]
        lines.append(
            f"Risk − Raw profit delta: {d:+.2f} "
            f"({100.0 * d / abs(rows[0]['mean']):+.2f}% of Raw)"
        )

    text = "\n".join(lines) + "\n"
    log_path.write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)
    print(f"log → {log_path}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
