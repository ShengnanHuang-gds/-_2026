#!/usr/bin/env python3
"""
Train/eval five RL algos (rule-ws) with allow_e1 on three both-mode matrices,
then plot fig_three_matrix_all_algos_allow_e1.pdf for report §7.1.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
FIG_DIR = ROOT / "figures"
PROFILES = ("pdf", "harsh_mild", "harsh_strong")
ALGOS = ("dqn", "ddqn", "dueling", "d3qn", "ppo")


def _ci95(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    return 1.96 * float(np.std(xs, ddof=1)) / math.sqrt(n)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--n-rep", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument(
        "--force-train",
        action="store_true",
        help="retrain even if checkpoint exists",
    )
    p.add_argument(
        "--algos",
        default=",".join(ALGOS),
    )
    p.add_argument(
        "--profiles",
        default=",".join(PROFILES),
    )
    return p.parse_args(argv)


def ckpt_path(profile: str, algo: str) -> Path:
    return CKPT_DIR / f"{algo}_{profile}_raw94_no_ws_allow_e1_best.pt"


def make_cfg(algo: str, profile: str, args: argparse.Namespace):
    common = dict(
        state_encoder="raw",
        action_space="coarse",
        disruption_mode="both",
        disruption_matrix_profile=profile,
        demand_intensity_scale=1.2,
        forbid_expedite=False,  # allow e=1
        rule_buffer_mask=False,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        eval_every_episodes=5,
        eval_num_seeds=3,
        log_every_episode=10,
        checkpoint_path=str(ckpt_path(profile, algo)),
        # No rule warm-start (user request for §7.1 allow_e1)
        rule_warmstart_episodes=0,
        rule_bc_updates=0,
        rule_td_pre_updates=0,
        lr=3e-4,
    )
    if algo == "ppo":
        return PPOTrainConfig(
            algorithm="ppo",
            entropy_coef=0.01,
            **common,
        )
    return DQNTrainConfig(
        algorithm=algo,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay_steps=15_000,
        start_learning_after=1_000,
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


def plot_all(rows: List[dict], basel: Dict[str, Dict[str, Tuple[float, float]]]) -> Path:
    """basel[profile][name] = (mean, ci)"""
    algo_order = ["dqn", "ddqn", "dueling", "d3qn", "ppo"]
    labels = ["DQN", "DDQN", "Dueling", "D3QN", "PPO"]
    colors = ["#4c78a8", "#2c7bb6", "#54a24b", "#e45756", "#b279a2"]
    profiles = list(PROFILES)

    keyed = {(r["profile"], r["algo"]): r for r in rows}

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    x = np.arange(len(algo_order))
    width = 0.62
    line_styles = {
        "fixed_0": ("#888888", "--", "fixed_0"),
        "rule": ("#d7191c", ":", "rule_based"),
        "best": ("#fdae61", "-.", "best_fixed"),
    }

    for ax, profile, title in zip(axes, profiles, profiles):
        means, cis = [], []
        for algo in algo_order:
            r = keyed[(profile, algo)]
            means.append(r["mean"] / 1e6)
            cis.append(r["ci"] / 1e6)
        ax.bar(
            x,
            means,
            width=width,
            yerr=cis,
            capsize=2.5,
            color=colors,
            edgecolor="white",
            linewidth=0.4,
            error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
        )
        b = basel[profile]
        for key, (col, ls, lab) in line_styles.items():
            ax.axhline(
                b[key][0] / 1e6,
                color=col,
                linestyle=ls,
                linewidth=1.4,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("Total profit (millions)")
    # auto ylim with padding
    all_m = [r["mean"] / 1e6 for r in rows]
    for p in profiles:
        for k in ("fixed_0", "rule", "best"):
            all_m.append(basel[p][k][0] / 1e6)
    lo, hi = min(all_m), max(all_m)
    pad = max(0.05, 0.08 * (hi - lo))
    axes[0].set_ylim(lo - pad, hi + pad)

    handles = [
        Patch(facecolor=c, edgecolor="white", label=l)
        for c, l in zip(colors, labels)
    ]
    handles += [
        Line2D([0], [0], color=col, linestyle=ls, linewidth=1.4, label=lab)
        for col, ls, lab in line_styles.values()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=8,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.suptitle(
        "allow_e1, no warm-start: RL vs baselines across Markov matrices",
        y=1.14,
        fontsize=12,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_three_matrix_all_algos_allow_e1.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    device = torch.device(args.device)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Train
    for profile in profiles:
        for algo in algos:
            cfg = make_cfg(algo, profile, args)
            path = Path(cfg.checkpoint_path)
            tag = f"{algo}/{profile}/allow_e1/no_ws"
            if path.is_file() and not getattr(args, "force_train", False):
                print(f"[reuse-ckpt] {tag} → {path.name}", flush=True)
                continue
            print(f"\n===== TRAIN {tag} =====", flush=True)
            if algo == "ppo":
                run_ppo_training(cfg)
            else:
                run_dqn_training(cfg)

    # Baselines + eval
    results: List[dict] = []
    basel: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for profile in profiles:
        sim = SimulationConfig(
            disruption_mode="both",
            disruption_matrix_profile=profile,
            warmup_days=30,
            evaluation_days=365,
            num_replications=args.n_rep,
            demand_intensity_scale=1.2,
            random_seed_base=args.seed,
        )
        print(f"\n===== baselines {profile} =====", flush=True)
        bmap: Dict[str, Tuple[float, float]] = {}
        for policy in make_baseline_policies(profile):
            _, profits = evaluate_baseline(sim, policy)
            m, ci = float(np.mean(profits)), _ci95(profits)
            print(f"  {policy.name}: {m:.2f} ± {ci:.2f}", flush=True)
            if policy.name.startswith("fixed_0"):
                bmap["fixed_0"] = (m, ci)
            elif policy.name.startswith("best_fixed"):
                bmap["best"] = (m, ci)
            elif policy.name == "rule_based":
                bmap["rule"] = (m, ci)
        basel[profile] = bmap

        for algo in algos:
            cfg = make_cfg(algo, profile, args)
            print(f"\n===== EVAL {algo}/{profile} =====", flush=True)
            net = load_net(algo, cfg, device)
            _, profits, _ = evaluate_ddqn_greedy(
                net, cfg, num_reps=args.n_rep, seed_base=args.seed, device=device
            )
            m, ci = float(np.mean(profits)), _ci95(profits)
            print(f"  {m:.2f} ± {ci:.2f}", flush=True)
            results.append(
                {
                    "profile": profile,
                    "algo": algo,
                    "mean": m,
                    "ci": ci,
                    "delta_rule": m - bmap["rule"][0],
                    "delta_best": m - bmap["best"][0],
                }
            )

    csv_path = LOG_DIR / "three_matrix_allow_e1_ws_30rep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "profile",
                "algo",
                "mean",
                "ci",
                "delta_rule",
                "delta_best",
            ],
        )
        w.writeheader()
        w.writerows(results)

    # text summary for table
    lines = [
        "Three-matrix allow_e1, NO warm-start (30-rep)",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"forbid_expedite=False rule_warmstart=0 episodes={args.episodes}",
        "",
    ]
    for profile in profiles:
        lines.append(f"=== {profile} ===")
        b = basel[profile]
        lines.append(
            f"  fixed_0={b['fixed_0'][0]:.2f}±{b['fixed_0'][1]:.2f} | "
            f"best={b['best'][0]:.2f}±{b['best'][1]:.2f} | "
            f"rule={b['rule'][0]:.2f}±{b['rule'][1]:.2f}"
        )
        for r in results:
            if r["profile"] != profile:
                continue
            lines.append(
                f"  {r['algo']:<8} {r['mean']:.2f}±{r['ci']:.2f} "
                f"Δrule={r['delta_rule']:+.0f} Δbest={r['delta_best']:+.0f}"
            )
        lines.append("")
    # DDQN table row helper
    lines.append("Table DDQN (no-ws, allow_e1) vs baselines:")
    for profile in profiles:
        dd = next(r for r in results if r["profile"] == profile and r["algo"] == "ddqn")
        b = basel[profile]
        lines.append(
            f"  {profile}: DDQN={dd['mean']/1e6:.3f}M "
            f"fixed_0={b['fixed_0'][0]/1e6:.3f}M "
            f"best={b['best'][0]/1e6:.3f}M "
            f"rule={b['rule'][0]/1e6:.3f}M"
        )

    log_path = LOG_DIR / "three_matrix_allow_e1_ws_30rep.log"
    text = "\n".join(lines) + "\n"
    log_path.write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)

    fig = plot_all(results, basel)
    print(f"csv → {csv_path}", flush=True)
    print(f"log → {log_path}", flush=True)
    print(f"fig → {fig}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
