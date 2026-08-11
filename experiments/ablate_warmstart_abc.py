#!/usr/bin/env python3
"""Warm-start strength ablation A/B/C (+ report D if checkpoint exists).

A: no warmstart
B: replay prefill only (no BC, no TD-pre)
C: prefill + light BC (200 updates) + TD-pre 500
D: prefill + heavy BC (2000) — reuse existing ckpt

Default: harsh_strong + forbid_e1, then 30-rep eval vs baselines.
Matches rule-warmstart hyperparams (50 RL ep, eps 0.2→0.01 for B/C/D;
A uses eps 1.0→0.01 like vanilla forbid-e1).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.simulation_config import SimulationConfig  # noqa: E402
from experiments.compare_ddqn_baselines import (  # noqa: E402
    evaluate_baseline,
    evaluate_ddqn_greedy,
)
from experiments.diagnose_ddqn_ell_g import (  # noqa: E402
    collect_traces,
    severity_bucket,
)
from metrics.performance_tracker import PerformanceTracker  # noqa: E402
from mdp.policies import make_baseline_policies  # noqa: E402
from mdp.state_encoder import build_state_encoder_from_env  # noqa: E402
from rl.dqn import QNetwork, build_q_network  # noqa: E402
from rl.dqn_train import DQNTrainConfig, build_env, run_dqn_training  # noqa: E402

LOG_DIR = ROOT / "logs"
CKPT_DIR = LOG_DIR / "checkpoints"

VARIANTS = {
    "A": {
        "label": "no_warmstart",
        "rule_warmstart_episodes": 0,
        "rule_bc_updates": 0,
        "rule_td_pre_updates": 0,
        "eps_start": 1.0,
        "eps_decay_steps": 15_000,
        "start_learning_after": 1_000,
    },
    "B": {
        "label": "prefill_only",
        "rule_warmstart_episodes": 20,
        "rule_bc_updates": 0,
        "rule_td_pre_updates": 0,
        "eps_start": 0.2,
        "eps_decay_steps": 10_000,
        "start_learning_after": 0,
    },
    "C": {
        "label": "light_bc_200",
        "rule_warmstart_episodes": 20,
        "rule_bc_updates": 200,
        "rule_td_pre_updates": 500,
        "eps_start": 0.2,
        "eps_decay_steps": 10_000,
        "start_learning_after": 0,
    },
}


def _ci95(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    return 1.96 * float(np.std(xs, ddof=1)) / math.sqrt(n)


def _g_entropy(gs: List[int]) -> float:
    if not gs:
        return float("nan")
    c = Counter(gs)
    n = len(gs)
    h = 0.0
    for v in c.values():
        p = v / n
        h -= p * math.log(p + 1e-12)
    return h


def _load_online(path: Path, train_cfg: DQNTrainConfig, device: torch.device) -> QNetwork:
    probe = build_env(train_cfg, seed=train_cfg.seed)
    probe.reset(seed=train_cfg.seed)
    encoder = build_state_encoder_from_env(probe)
    online = build_q_network(state_dim=encoder.dim, device=device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    online.load_state_dict(ckpt["model_state_dict"])
    online.eval()
    return online


def _make_train_cfg(
    key: str,
    *,
    profile: str,
    episodes: int,
    seed: int,
    device: str,
    out_dir: Path,
) -> DQNTrainConfig:
    spec = VARIANTS[key]
    ckpt = out_dir / f"ddqn_ws_ablate_{key}_{spec['label']}_best.pt"
    return DQNTrainConfig(
        algorithm="ddqn",
        episodes=episodes,
        evaluation_days=365,
        warmup_days=30,
        disruption_mode="both",
        disruption_matrix_profile=profile,
        demand_intensity_scale=1.2,
        forbid_expedite=True,
        enforce_action_mask=True,
        seed=seed,
        device=device,
        lr=3e-4,
        eps_start=float(spec["eps_start"]),
        eps_end=0.01,
        eps_decay_steps=int(spec["eps_decay_steps"]),
        start_learning_after=int(spec["start_learning_after"]),
        eval_every_episodes=5,
        eval_num_seeds=3,
        log_every_episode=10,
        checkpoint_path=str(ckpt),
        rule_warmstart_episodes=int(spec["rule_warmstart_episodes"]),
        rule_bc_updates=int(spec["rule_bc_updates"]),
        rule_td_pre_updates=int(spec["rule_td_pre_updates"]),
    )


def _train_variant(key: str, cfg: DQNTrainConfig) -> Path:
    spec = VARIANTS[key]
    print(f"\n===== TRAIN {key}: {spec['label']} =====", flush=True)
    print(
        f"  warmstart_ep={cfg.rule_warmstart_episodes} "
        f"bc={cfg.rule_bc_updates} td_pre={cfg.rule_td_pre_updates} "
        f"eps={cfg.eps_start}→{cfg.eps_end} rl_ep={cfg.episodes}",
        flush=True,
    )
    result = run_dqn_training(cfg)
    print(
        f"[{key}] best_eval={result['best_eval_profit']:.2f} "
        f"ws={result.get('warmstart')}",
        flush=True,
    )
    return Path(cfg.checkpoint_path)


def _diag_from_traces(records: List[dict]) -> Dict[str, float]:
    severe = [r for r in records if r["severity"] == "severe"]
    n_sev = len(severe)
    ell2 = sum(1 for r in severe if r["ell"] == 2)
    return {
        "n_severe": float(n_sev),
        "ell2_given_severe": (ell2 / n_sev) if n_sev else float("nan"),
        "g_entropy_severe": _g_entropy([r["g"] for r in severe]),
        "agree_rule": (
            sum(1 for r in records if r["agree_rule"]) / len(records)
            if records
            else float("nan")
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default="harsh_strong")
    p.add_argument("--variants", default="A,B,C")
    p.add_argument("--train-episodes", type=int, default=50)
    p.add_argument("--n-rep", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-seed0", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--d-checkpoint",
        default=str(CKPT_DIR / "ddqn_harsh_strong_forbid_e1_rule_ws_best.pt"),
    )
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--checkpoint-dir", default=str(CKPT_DIR))
    p.add_argument(
        "--log-path",
        default="",
    )
    args = p.parse_args()

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    keys = [k.strip().upper() for k in args.variants.split(",") if k.strip()]
    log_path = Path(
        args.log_path
        or (LOG_DIR / f"ddqn_ws_ablate_{args.profile}_{args.n_rep}rep.log")
    )

    ckpts: Dict[str, Path] = {}
    train_cfgs: Dict[str, DQNTrainConfig] = {}
    for key in keys:
        if key not in VARIANTS:
            raise SystemExit(f"unknown variant {key}; expected A/B/C")
        cfg = _make_train_cfg(
            key,
            profile=args.profile,
            episodes=args.train_episodes,
            seed=args.seed,
            device=args.device,
            out_dir=out_dir,
        )
        train_cfgs[key] = cfg
        expected = Path(cfg.checkpoint_path)
        if args.skip_train and expected.is_file():
            ckpts[key] = expected
        else:
            ckpts[key] = _train_variant(key, cfg)

    # Shared eval train_cfg (forbid_e1, same horizon)
    eval_cfg = DQNTrainConfig(
        algorithm="ddqn",
        disruption_matrix_profile=args.profile,
        warmup_days=30,
        evaluation_days=365,
        demand_intensity_scale=1.2,
        forbid_expedite=True,
        seed=args.eval_seed0,
        device=args.device,
    )
    sim = SimulationConfig(
        disruption_mode="both",
        disruption_matrix_profile=args.profile,
        warmup_days=30,
        evaluation_days=365,
        num_replications=args.n_rep,
        demand_intensity_scale=1.2,
        random_seed_base=args.eval_seed0,
    )

    print(
        f"\n===== {args.n_rep}-rep eval profile={args.profile} =====",
        flush=True,
    )

    # Baselines once
    baseline_profits: Dict[str, List[float]] = {}
    for policy in make_baseline_policies(args.profile):
        print(f"\n=== baseline {policy.name} ===", flush=True)
        _, profits = evaluate_baseline(sim, policy)
        baseline_profits[policy.name] = profits
        print(
            f"  {policy.name} mean={float(np.mean(profits)):.2f} "
            f"± {_ci95(profits):.2f}",
            flush=True,
        )

    all_rows: Dict[str, Dict[str, Any]] = {}

    def eval_ddqn(label: str, path: Path) -> Tuple[List[float], Dict[str, float]]:
        online = _load_online(path, eval_cfg, device)
        report, profits, _ = evaluate_ddqn_greedy(
            online,
            eval_cfg,
            num_reps=args.n_rep,
            seed_base=args.eval_seed0,
            device=device,
        )
        print(
            f"  [{label}] DDQN mean={float(np.mean(profits)):.2f} "
            f"± {_ci95(profits):.2f}  (report {report['total_profit']})",
            flush=True,
        )
        records = collect_traces(
            online,
            eval_cfg,
            reps=min(10, args.n_rep),
            seed_base=args.eval_seed0,
            device=device,
        )
        diag = _diag_from_traces(records)
        print(
            f"  [{label}] severe_days={int(diag['n_severe'])} "
            f"ell2|sev={diag['ell2_given_severe']:.1%} "
            f"gH_sev={diag['g_entropy_severe']:.3f} "
            f"agree_rule={diag['agree_rule']:.1%}",
            flush=True,
        )
        return profits, diag

    for key, path in ckpts.items():
        label = f"{key}:{VARIANTS[key]['label']}"
        print(f"\n=== DDQN {label} ===", flush=True)
        profits, diag = eval_ddqn(label, path)
        all_rows[key] = {"profits": profits, "diag": diag, "label": VARIANTS[key]["label"]}

    d_path = Path(args.d_checkpoint)
    if d_path.is_file():
        print("\n=== DDQN D:heavy_bc_2000 ===", flush=True)
        profits, diag = eval_ddqn("D:heavy_bc_2000", d_path)
        all_rows["D"] = {
            "profits": profits,
            "diag": diag,
            "label": "heavy_bc_2000",
        }

    rule_mean = float(np.mean(baseline_profits["rule_based"]))
    best_name = [k for k in baseline_profits if k.startswith("best_fixed")][0]
    fixed0_name = [k for k in baseline_profits if k.startswith("fixed_0")][0]
    best_mean = float(np.mean(baseline_profits[best_name]))
    fixed0_mean = float(np.mean(baseline_profits[fixed0_name]))

    lines = [
        "Warm-start strength ablation A/B/C (+D)",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"profile: {args.profile}",
        f"forbid_expedite: True",
        f"n_rep: {args.n_rep} seeds {args.eval_seed0}..{args.eval_seed0 + args.n_rep - 1}",
        f"train_episodes: {args.train_episodes}",
        "",
        f"{'var':<4} {'label':<14} {'DDQN':>12} {'±CI':>10} {'Δrule':>10} "
        f"{'Δbest':>10} {'ell2|sev':>10} {'agree_r':>8}",
    ]
    order = [k for k in ["A", "B", "C", "D"] if k in all_rows]
    for key in order:
        row = all_rows[key]
        xs = row["profits"]
        m = float(np.mean(xs))
        ci = _ci95(xs)
        diag = row["diag"]
        lines.append(
            f"{key:<4} {row['label']:<14} {m:12.2f} {ci:10.2f} "
            f"{m - rule_mean:+10.2f} {m - best_mean:+10.2f} "
            f"{diag['ell2_given_severe']:10.1%} {diag['agree_rule']:8.1%}"
        )
    lines.append("")
    lines.append(
        f"{'--':<4} {'rule_based':<14} {rule_mean:12.2f} "
        f"{_ci95(baseline_profits['rule_based']):10.2f}"
    )
    lines.append(
        f"{'--':<4} {best_name:<14} {best_mean:12.2f} "
        f"{_ci95(baseline_profits[best_name]):10.2f}"
    )
    lines.append(
        f"{'--':<4} {fixed0_name:<14} {fixed0_mean:12.2f} "
        f"{_ci95(baseline_profits[fixed0_name]):10.2f}"
    )
    lines.extend(
        [
            "",
            "Interpretation guide:",
            "  A = learn from scratch (no imitation)",
            "  B = rule data in replay only (no BC)",
            "  C = light BC (200)",
            "  D = heavy BC (2000) — prior run",
            "  Rising agree_rule / ell2|sev with BC → imitation lock-in",
        ]
    )

    text = "\n".join(lines) + "\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)
    print(f"log → {log_path}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
