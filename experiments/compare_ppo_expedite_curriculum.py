#!/usr/bin/env python3
"""
PPO expedite curriculum对照 (harsh_strong, raw94, coarse).

Arms:
  A) forbid_e1 + rule warmstart          (stage-1)
  B) load A → allow_e1 fine-tune         (curriculum)
  C) allow_e1 from scratch + rule ws     (no curriculum)

Eval: 30-rep profit + e=1 ratio vs rule / best_fixed.
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
from mdp.actions import decode_action, num_actions_for
from mdp.policies import make_baseline_policies
from mdp.state_encoder import build_state_encoder_from_env
from rl.dqn import states_to_tensor
from rl.dqn_train import build_env
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
    p.add_argument("--episodes-a", type=int, default=50, help="forbid_e1 episodes")
    p.add_argument(
        "--episodes-b", type=int, default=30, help="allow_e1 fine-tune episodes"
    )
    p.add_argument(
        "--episodes-c", type=int, default=50, help="scratch allow_e1 episodes"
    )
    p.add_argument("--n-rep", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-train-a", action="store_true")
    p.add_argument(
        "--init-a",
        default="",
        help="optional existing forbid_e1 PPO ckpt for stage A",
    )
    return p.parse_args(argv)


def ckpt_a(profile: str) -> Path:
    return CKPT_DIR / f"ppo_{profile}_raw94_ws_forbid_e1_stageA_best.pt"


def ckpt_b(profile: str) -> Path:
    return CKPT_DIR / f"ppo_{profile}_raw94_ws_curriculum_allow_e1_best.pt"


def ckpt_c(profile: str) -> Path:
    return CKPT_DIR / f"ppo_{profile}_raw94_ws_scratch_allow_e1_best.pt"


def make_forbid_cfg(args: argparse.Namespace) -> PPOTrainConfig:
    return PPOTrainConfig(
        algorithm="ppo",
        state_encoder="raw",
        action_space="coarse",
        disruption_mode="both",
        disruption_matrix_profile=args.profile,
        demand_intensity_scale=1.2,
        forbid_expedite=True,
        rule_buffer_mask=False,
        episodes=args.episodes_a,
        seed=args.seed,
        device=args.device,
        lr=3e-4,
        rule_warmstart_episodes=20,
        rule_bc_updates=2000,
        rule_td_pre_updates=0,
        entropy_coef=0.005,
        eval_every_episodes=5,
        eval_num_seeds=3,
        log_every_episode=10,
        checkpoint_path=str(ckpt_a(args.profile)),
    )


def make_curriculum_cfg(
    args: argparse.Namespace, init_path: str
) -> PPOTrainConfig:
    return PPOTrainConfig(
        algorithm="ppo",
        state_encoder="raw",
        action_space="coarse",
        disruption_mode="both",
        disruption_matrix_profile=args.profile,
        demand_intensity_scale=1.2,
        forbid_expedite=False,  # open e=1
        rule_buffer_mask=False,
        episodes=args.episodes_b,
        seed=args.seed + 1000,
        device=args.device,
        lr=1e-4,  # smaller LR for fine-tune
        rule_warmstart_episodes=0,
        rule_bc_updates=0,
        rule_td_pre_updates=0,
        entropy_coef=0.02,  # encourage trying e=1
        init_checkpoint=init_path,
        eval_every_episodes=5,
        eval_num_seeds=3,
        log_every_episode=5,
        checkpoint_path=str(ckpt_b(args.profile)),
    )


def make_scratch_allow_cfg(args: argparse.Namespace) -> PPOTrainConfig:
    return PPOTrainConfig(
        algorithm="ppo",
        state_encoder="raw",
        action_space="coarse",
        disruption_mode="both",
        disruption_matrix_profile=args.profile,
        demand_intensity_scale=1.2,
        forbid_expedite=False,
        rule_buffer_mask=False,
        episodes=args.episodes_c,
        seed=args.seed + 2000,
        device=args.device,
        lr=3e-4,
        rule_warmstart_episodes=20,
        rule_bc_updates=2000,
        rule_td_pre_updates=0,
        entropy_coef=0.01,
        eval_every_episodes=5,
        eval_num_seeds=3,
        log_every_episode=10,
        checkpoint_path=str(ckpt_c(args.profile)),
    )


def load_net(cfg: PPOTrainConfig, device: torch.device):
    probe = build_env(cfg, seed=cfg.seed)
    probe.reset(seed=cfg.seed)
    enc = build_state_encoder_from_env(probe, encoder_type=cfg.state_encoder)
    net = build_actor_critic(
        state_dim=enc.dim,
        num_actions=num_actions_for(cfg.action_space),
        device=device,
    )
    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net


@torch.no_grad()
def diagnose_e1(
    net,
    cfg: PPOTrainConfig,
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
            act = decode_action(action, space=cfg.action_space)
            if act.expedite == 1:
                e1 += 1
            n += 1
            state, _, _, info = env.step(action)
            cost += float(info.get("expedite_cost", 0.0) or 0.0)
    return {
        "e1_ratio": e1 / n if n else float("nan"),
        "expedite_cost_per_ep": cost / n_rep if n_rep else float("nan"),
        "n_steps": float(n),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Stage A: forbid e=1 -----
    cfg_a = make_forbid_cfg(args)
    path_a = Path(cfg_a.checkpoint_path)
    train_stats: Dict[str, Dict[str, float]] = {}
    if args.init_a:
        src = Path(args.init_a)
        if not src.is_file():
            raise FileNotFoundError(src)
        print(f"[A] reuse init-a → copy to {path_a}", flush=True)
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        torch.save(ckpt, path_a)
        train_stats["A_forbid"] = {"sample_e1": 0.0, "e1_legal": 0.0}
    elif args.skip_train_a and path_a.is_file():
        print(f"[A] skip-train, use {path_a}", flush=True)
        train_stats["A_forbid"] = {"sample_e1": float("nan"), "e1_legal": float("nan")}
    else:
        print("\n===== STAGE A: PPO forbid_e1 + rule ws =====", flush=True)
        result_a = run_ppo_training(cfg_a)
        print(f"[A] best_eval={result_a.get('best_eval_profit')}", flush=True)
        train_stats["A_forbid"] = {
            "sample_e1": float(result_a["mean_train_e1_ratio"]),
            "e1_legal": float(result_a["mean_train_e1_legal_ratio"]),
        }

    # ----- Stage B: curriculum allow e=1 -----
    cfg_b = make_curriculum_cfg(args, str(path_a))
    print("\n===== STAGE B: curriculum allow_e1 fine-tune =====", flush=True)
    result_b = run_ppo_training(cfg_b)
    print(f"[B] best_eval={result_b.get('best_eval_profit')}", flush=True)
    train_stats["B_curriculum"] = {
        "sample_e1": float(result_b["mean_train_e1_ratio"]),
        "e1_legal": float(result_b["mean_train_e1_legal_ratio"]),
    }

    # ----- Stage C: scratch allow e=1 -----
    cfg_c = make_scratch_allow_cfg(args)
    print("\n===== STAGE C: scratch allow_e1 + rule ws =====", flush=True)
    result_c = run_ppo_training(cfg_c)
    print(f"[C] best_eval={result_c.get('best_eval_profit')}", flush=True)
    train_stats["C_scratch"] = {
        "sample_e1": float(result_c["mean_train_e1_ratio"]),
        "e1_legal": float(result_c["mean_train_e1_legal_ratio"]),
    }

    # ----- Baselines -----
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
    for tag, cfg in (("A_forbid", cfg_a), ("B_curriculum", cfg_b), ("C_scratch", cfg_c)):
        print(f"\n===== EVAL {tag} =====", flush=True)
        net = load_net(cfg, device)
        _, profits, _ = evaluate_ddqn_greedy(
            net, cfg, num_reps=args.n_rep, seed_base=args.seed, device=device
        )
        diag = diagnose_e1(
            net, cfg, n_rep=min(10, args.n_rep), seed0=args.seed, device=device
        )
        m = float(np.mean(profits))
        ci = _ci95(profits)
        print(
            f"  {tag}: {m:.2f} ± {ci:.2f} | greedy_e1={diag['e1_ratio']:.1%} "
            f"| train_e1={train_stats[tag]['sample_e1']:.1%} "
            f"| e1_legal={train_stats[tag]['e1_legal']:.1%} "
            f"| exp_cost/ep={diag['expedite_cost_per_ep']:.0f} "
            f"| Δrule={m - rule_mean:+.0f} Δbest={m - best_mean:+.0f}",
            flush=True,
        )
        rows.append(
            {
                "tag": tag,
                "forbid": cfg.forbid_expedite,
                "mean": m,
                "ci": ci,
                "e1": diag["e1_ratio"],
                "exp_cost": diag["expedite_cost_per_ep"],
                "train_e1": train_stats[tag]["sample_e1"],
                "train_e1_legal": train_stats[tag]["e1_legal"],
            }
        )

    lines = [
        "PPO expedite curriculum (harsh_strong, raw94, coarse)",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"A: forbid_e1 + ws ({args.episodes_a} ep)",
        f"B: load A → allow_e1 fine-tune ({args.episodes_b} ep, lr=1e-4, ent=0.02)",
        f"C: scratch allow_e1 + ws ({args.episodes_c} ep)",
        f"n_rep={args.n_rep}",
        "",
        "train_sample_e1 = stochastic rollout e=1 rate during training",
        "train_e1_legal = fraction of days where mask allows at least one e=1",
        "greedy_e1 = e=1 rate under select_greedy eval",
        "",
        f"{'arm':<14} {'forbid':>6} {'profit':>12} {'±CI':>10} "
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
            f"{r['train_e1_legal']:10.1%}"
            if r["train_e1_legal"] == r["train_e1_legal"]
            else f"{'n/a':>10}"
        )
        lines.append(
            f"{r['tag']:<14} {'Y' if r['forbid'] else 'N':>6} "
            f"{r['mean']:12.2f} {r['ci']:10.2f} {r['e1']:10.1%} "
            f"{te} {tl} "
            f"{r['mean'] - rule_mean:+10.2f} {r['mean'] - best_mean:+10.2f}"
        )
    lines.append("")
    lines.append(
        f"baselines: rule={rule_mean:.2f}±{_ci95(basel['rule_based']):.2f} | "
        f"{best_name}={best_mean:.2f}±{_ci95(basel[best_name]):.2f}"
    )
    keyed = {r["tag"]: r for r in rows}
    if "A_forbid" in keyed and "B_curriculum" in keyed:
        a, b = keyed["A_forbid"], keyed["B_curriculum"]
        lines.append(
            f"curriculum lift B−A: {b['mean'] - a['mean']:+.2f} "
            f"(greedy_e1 {a['e1']:.1%}→{b['e1']:.1%}; "
            f"train_e1 {a['train_e1']:.1%}→{b['train_e1']:.1%})"
        )
    if "C_scratch" in keyed and "B_curriculum" in keyed:
        b, c = keyed["B_curriculum"], keyed["C_scratch"]
        lines.append(
            f"curriculum vs scratch B−C: {b['mean'] - c['mean']:+.2f} "
            f"(greedy_e1 {b['e1']:.1%} vs {c['e1']:.1%}; "
            f"train_e1 {b['train_e1']:.1%} vs {c['train_e1']:.1%})"
        )

    text = "\n".join(lines) + "\n"
    log_path = LOG_DIR / f"ppo_expedite_curriculum_{args.profile}_{args.n_rep}rep.log"
    log_path.write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)
    print(f"log → {log_path}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
