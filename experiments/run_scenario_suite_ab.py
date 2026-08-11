#!/usr/bin/env python3
"""
Part A (mechanism ablation) + Part B (long-horizon Markov) suite.

Algorithms: DQN / DDQN / Dueling / D3QN / PPO
× warmstart on/off
× raw 94-d encoder, coarse actions, forbid_e1

Part A scenarios:
  none
  supply_only × {pdf, harsh_mild, harsh_strong}
  transport_only × {pdf, harsh_mild, harsh_strong}
  fixed_U3 (short horizon)
  fixed_A2, fixed_A3 (severe supply)

Part B scenarios:
  both × {pdf, harsh_mild, harsh_strong}
  horizon 30+365, multi-rep

Example::

    python experiments/run_scenario_suite_ab.py --part all
    python experiments/run_scenario_suite_ab.py --part B --n-rep-b 30
    python experiments/run_scenario_suite_ab.py --part A --algos dqn,ddqn,ppo
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_ddqn_baselines import evaluate_ddqn_greedy
from mdp.actions import num_actions_for
from mdp.policies import BEST_FIXED_BY_PROFILE, make_baseline_policies
from mdp.state_encoder import build_state_encoder_from_env
from rl.dqn import build_q_network
from rl.dqn_train import DQNTrainConfig, build_env, run_dqn_training
from rl.ppo import build_actor_critic
from rl.ppo_train import PPOTrainConfig, run_ppo_training

LOG_DIR = ROOT / "logs"
CKPT_DIR = LOG_DIR / "checkpoints" / "suite_ab"
ALL_ALGOS = ("dqn", "ddqn", "dueling", "d3qn", "ppo")
PROFILES = ("pdf", "harsh_mild", "harsh_strong")


@dataclass(frozen=True)
class Scenario:
    part: str  # "A" | "B"
    name: str
    disruption_mode: str
    profile: str
    fixed_a: Optional[int] = None
    fixed_u: Optional[int] = None
    warmup_days: int = 30
    evaluation_days: int = 365
    note: str = ""


def build_scenarios() -> List[Scenario]:
    out: List[Scenario] = []
    out.append(
        Scenario("A", "none", "none", "pdf", note="no disruption")
    )
    for p in PROFILES:
        out.append(
            Scenario("A", f"supply_only_{p}", "supply_only", p, note="A-chain only")
        )
    for p in PROFILES:
        out.append(
            Scenario(
                "A", f"transport_only_{p}", "transport_only", p, note="U-chain only"
            )
        )
    out.append(
        Scenario(
            "A",
            "fixed_U3",
            "none",
            "pdf",
            fixed_a=0,
            fixed_u=3,
            warmup_days=15,
            evaluation_days=90,
            note="locked transport block",
        )
    )
    out.append(
        Scenario(
            "A",
            "fixed_A2",
            "none",
            "pdf",
            fixed_a=2,
            fixed_u=0,
            warmup_days=15,
            evaluation_days=120,
            note="locked severe supply 40%",
        )
    )
    out.append(
        Scenario(
            "A",
            "fixed_A3",
            "none",
            "pdf",
            fixed_a=3,
            fixed_u=0,
            warmup_days=15,
            evaluation_days=120,
            note="locked supply shutdown",
        )
    )
    for p in PROFILES:
        out.append(
            Scenario(
                "B",
                f"both_{p}",
                "both",
                p,
                warmup_days=30,
                evaluation_days=365,
                note="long-horizon compound Markov",
            )
        )
    return out


def _ci95(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    return 1.96 * float(np.std(xs, ddof=1)) / math.sqrt(n)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part", default="all", choices=["all", "A", "B"])
    p.add_argument("--algos", default=",".join(ALL_ALGOS))
    p.add_argument("--warmstarts", default="0,1")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--n-rep-a", type=int, default=10)
    p.add_argument("--n-rep-b", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument(
        "--scenarios",
        default="",
        help="optional comma filter of scenario names",
    )
    return p.parse_args(argv)


def ckpt_for(scen: Scenario, algo: str, warm: bool) -> Path:
    tag = "ws" if warm else "no_ws"
    return CKPT_DIR / f"{scen.part}_{scen.name}_{algo}_{tag}_best.pt"


def make_value_cfg(
    algo: str,
    warm: bool,
    scen: Scenario,
    args: argparse.Namespace,
) -> DQNTrainConfig:
    common = dict(
        algorithm=algo,
        state_encoder="raw",
        action_space="coarse",
        disruption_mode=scen.disruption_mode,
        disruption_matrix_profile=scen.profile,
        fixed_supplier_availability=scen.fixed_a,
        fixed_transport_state=scen.fixed_u,
        warmup_days=scen.warmup_days,
        evaluation_days=scen.evaluation_days,
        eval_warmup_days=scen.warmup_days,
        eval_evaluation_days=scen.evaluation_days,
        demand_intensity_scale=1.2,
        forbid_expedite=True,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        eval_every_episodes=max(5, args.episodes // 8),
        eval_num_seeds=2,
        log_every_episode=max(5, args.episodes // 8),
        checkpoint_path=str(ckpt_for(scen, algo, warm)),
    )
    if warm:
        return DQNTrainConfig(
            lr=3e-4,
            eps_start=0.2,
            eps_end=0.01,
            eps_decay_steps=10_000,
            start_learning_after=0,
            rule_warmstart_episodes=15,
            rule_bc_updates=1500,
            rule_td_pre_updates=500,
            **common,
        )
    return DQNTrainConfig(
        lr=3e-4,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay_steps=12_000,
        start_learning_after=800,
        rule_warmstart_episodes=0,
        rule_bc_updates=0,
        rule_td_pre_updates=0,
        **common,
    )


def make_ppo_cfg(
    warm: bool, scen: Scenario, args: argparse.Namespace
) -> PPOTrainConfig:
    base = dict(
        state_encoder="raw",
        action_space="coarse",
        disruption_mode=scen.disruption_mode,
        disruption_matrix_profile=scen.profile,
        fixed_supplier_availability=scen.fixed_a,
        fixed_transport_state=scen.fixed_u,
        warmup_days=scen.warmup_days,
        evaluation_days=scen.evaluation_days,
        eval_warmup_days=scen.warmup_days,
        eval_evaluation_days=scen.evaluation_days,
        demand_intensity_scale=1.2,
        forbid_expedite=True,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        eval_every_episodes=max(5, args.episodes // 8),
        eval_num_seeds=2,
        log_every_episode=max(5, args.episodes // 8),
        checkpoint_path=str(ckpt_for(scen, "ppo", warm)),
        ppo_epochs=4,
        ppo_minibatch_size=256,
        entropy_coef=0.01 if not warm else 0.005,
    )
    if warm:
        return PPOTrainConfig(
            algorithm="ppo",
            rule_warmstart_episodes=15,
            rule_bc_updates=1500,
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


def eval_baselines(scen: Scenario, n_rep: int, seed0: int) -> Dict[str, List[float]]:
    profile_key = scen.profile if scen.profile in BEST_FIXED_BY_PROFILE else "pdf"
    out: Dict[str, List[float]] = {}
    for policy in make_baseline_policies(profile_key, action_space="coarse"):
        profits: List[float] = []
        for r in range(n_rep):
            seed = seed0 + r
            cfg = DQNTrainConfig(
                algorithm="ddqn",
                disruption_mode=scen.disruption_mode,
                disruption_matrix_profile=scen.profile,
                fixed_supplier_availability=scen.fixed_a,
                fixed_transport_state=scen.fixed_u,
                warmup_days=scen.warmup_days,
                evaluation_days=scen.evaluation_days,
                forbid_expedite=False,
                seed=seed,
            )
            env = build_env(cfg, seed=seed)
            state = env.reset(seed=seed)
            while not env.done:
                state, _, _, _ = env.step(policy.select(state))
            profits.append(
                float(env.engine.performance_tracker.summarize()["total_profit"])
            )
        out[policy.name] = profits
        print(
            f"  baseline {policy.name}: {float(np.mean(profits)):.1f} ± {_ci95(profits):.1f}",
            flush=True,
        )
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    algos = [a.strip().lower() for a in args.algos.split(",") if a.strip()]
    warms = []
    for w in args.warmstarts.split(","):
        w = w.strip()
        if w in ("0", "no", "false"):
            warms.append(False)
        elif w in ("1", "yes", "true"):
            warms.append(True)
        else:
            raise SystemExit(f"bad warmstart {w}")

    scenarios = build_scenarios()
    if args.part != "all":
        scenarios = [s for s in scenarios if s.part == args.part]
    if args.scenarios.strip():
        want = {x.strip() for x in args.scenarios.split(",") if x.strip()}
        scenarios = [s for s in scenarios if s.name in want]

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = LOG_DIR / f"suite_ab_results_{stamp}.csv"
    md_path = LOG_DIR / f"suite_ab_summary_{stamp}.md"

    rows: List[dict] = []
    t_suite = time.time()
    print(
        f"Suite AB | scenarios={len(scenarios)} algos={algos} warms={warms} "
        f"episodes={args.episodes}",
        flush=True,
    )

    for scen in scenarios:
        n_rep = args.n_rep_a if scen.part == "A" else args.n_rep_b
        print(
            f"\n######## {scen.part}/{scen.name} mode={scen.disruption_mode} "
            f"profile={scen.profile} fixed=({scen.fixed_a},{scen.fixed_u}) "
            f"H={scen.warmup_days}+{scen.evaluation_days} n_rep={n_rep} ########",
            flush=True,
        )

        print("--- baselines ---", flush=True)
        basel = eval_baselines(scen, n_rep=n_rep, seed0=args.seed)
        rule_mean = float(np.mean(basel["rule_based"]))
        best_key = [k for k in basel if k.startswith("best_fixed")][0]
        fixed0_key = [k for k in basel if k.startswith("fixed_0")][0]
        best_mean = float(np.mean(basel[best_key]))
        fixed0_mean = float(np.mean(basel[fixed0_key]))

        for name, profits in basel.items():
            rows.append(
                {
                    "part": scen.part,
                    "scenario": scen.name,
                    "algo": name,
                    "warmstart": "",
                    "mean_profit": float(np.mean(profits)),
                    "ci95": _ci95(profits),
                    "n_rep": n_rep,
                    "delta_rule": float(np.mean(profits)) - rule_mean,
                    "delta_best": float(np.mean(profits)) - best_mean,
                    "horizon": f"{scen.warmup_days}+{scen.evaluation_days}",
                    "note": scen.note,
                }
            )

        for algo in algos:
            for warm in warms:
                tag = f"{algo}/{'ws' if warm else 'no_ws'}"
                cfg = (
                    make_ppo_cfg(warm, scen, args)
                    if algo == "ppo"
                    else make_value_cfg(algo, warm, scen, args)
                )
                path = Path(cfg.checkpoint_path)
                if not (args.skip_train and path.is_file()):
                    print(f"\n=== TRAIN {scen.name} | {tag} ===", flush=True)
                    t0 = time.time()
                    if algo == "ppo":
                        result = run_ppo_training(cfg)
                    else:
                        result = run_dqn_training(cfg)
                    print(
                        f"[{tag}] best_eval={result.get('best_eval_profit')} "
                        f"({time.time() - t0:.0f}s)",
                        flush=True,
                    )
                else:
                    print(f"[skip-train] {scen.name} | {tag}", flush=True)

                net = load_net(algo, cfg, device)
                _, profits, _ = evaluate_ddqn_greedy(
                    net,
                    cfg,
                    num_reps=n_rep,
                    seed_base=args.seed,
                    device=device,
                )
                m = float(np.mean(profits))
                ci = _ci95(profits)
                print(
                    f"  EVAL {tag}: {m:.1f} ± {ci:.1f} "
                    f"(Δrule={m - rule_mean:+.0f}, Δbest={m - best_mean:+.0f})",
                    flush=True,
                )
                rows.append(
                    {
                        "part": scen.part,
                        "scenario": scen.name,
                        "algo": algo,
                        "warmstart": "Y" if warm else "N",
                        "mean_profit": m,
                        "ci95": ci,
                        "n_rep": n_rep,
                        "delta_rule": m - rule_mean,
                        "delta_best": m - best_mean,
                        "horizon": f"{scen.warmup_days}+{scen.evaluation_days}",
                        "note": scen.note,
                    }
                )

    # Write CSV
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Markdown summary pivots
    lines = [
        f"# Suite A/B summary",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"episodes={args.episodes} n_rep_A={args.n_rep_a} n_rep_B={args.n_rep_b}",
        f"algos={algos} warmstarts={warms}",
        f"elapsed_sec={time.time() - t_suite:.0f}",
        "",
        "## Results (RL + baselines)",
        "",
        "| part | scenario | algo | ws | profit | ±CI | Δrule | Δbest |",
        "|------|----------|------|----|--------|-----|-------|-------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['part']} | {r['scenario']} | {r['algo']} | {r['warmstart'] or '-'} | "
            f"{r['mean_profit']:.1f} | {r['ci95']:.1f} | "
            f"{r['delta_rule']:+.1f} | {r['delta_best']:+.1f} |"
        )

    # Best RL per scenario
    lines.extend(["", "## Best RL per scenario", ""])
    by_scen: Dict[str, List[dict]] = {}
    for r in rows:
        if r["warmstart"] in ("Y", "N"):
            by_scen.setdefault(r["scenario"], []).append(r)
    for scen_name, lst in by_scen.items():
        best = max(lst, key=lambda x: x["mean_profit"])
        lines.append(
            f"- **{scen_name}**: {best['algo']} ws={best['warmstart']} "
            f"→ {best['mean_profit']:.1f} (Δrule={best['delta_rule']:+.1f})"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nCSV → {csv_path}", flush=True)
    print(f"MD  → {md_path}", flush=True)
    print("\n".join(lines[-30:]), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
