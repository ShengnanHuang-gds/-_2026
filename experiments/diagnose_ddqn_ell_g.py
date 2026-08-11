#!/usr/bin/env python3
"""
Diagnose forbid-e1 DDQN action choices: focus on (ell, g).

Default: harsh_strong + logs/checkpoints/ddqn_harsh_strong_forbid_e1_best.pt

Reports:
  - overall action / ell / g distribution
  - breakdown by disruption severity (normal / mild / severe)
  - agreement with best_fixed_12=(2,0,0) and rule_based
  - whether ell=2 is used enough on severe days

Example::

    python experiments/diagnose_ddqn_ell_g.py
    python experiments/diagnose_ddqn_ell_g.py --profile harsh_mild --reps 10
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdp.actions import decode_action, encode_action
from mdp.policies import BEST_FIXED_BY_PROFILE, RuleBasedPolicy
from mdp.state_encoder import build_state_encoder_from_env
from rl.dqn import build_q_network, states_to_tensor
from rl.dqn_train import DQNTrainConfig, build_env

LOG_DIR = ROOT / "logs"
CKPT_DIR = LOG_DIR / "checkpoints"


def severity_bucket(a: int, u: int) -> str:
    if a == 0 and u == 0:
        return "normal"
    if a >= 2 or u >= 2:
        return "severe"
    return "mild"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose DDQN (ell,g) under forbid_e1")
    p.add_argument("--profile", default="harsh_strong")
    p.add_argument(
        "--checkpoint",
        default="",
        help="default: logs/checkpoints/ddqn_<profile>_forbid_e1_best.pt",
    )
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-days", type=int, default=30)
    p.add_argument("--evaluation-days", type=int, default=365)
    p.add_argument("--demand-intensity-scale", type=float, default=1.2)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--log-path",
        default="",
        help="default: logs/ddqn_<profile>_forbid_e1_ell_g_diag.log",
    )
    return p.parse_args(argv)


@torch.no_grad()
def collect_traces(
    online,
    cfg: DQNTrainConfig,
    *,
    reps: int,
    seed_base: int,
    device: torch.device,
) -> List[dict]:
    """One record per decision day."""
    rule = RuleBasedPolicy()
    best_id = BEST_FIXED_BY_PROFILE.get(cfg.disruption_matrix_profile, 12)
    records: List[dict] = []

    online.eval()
    for r in range(reps):
        seed = seed_base + r
        env = build_env(
            cfg,
            seed=seed,
            evaluation_days=cfg.evaluation_days,
            warmup_days=cfg.warmup_days,
        )
        assert env.forbid_expedite is True
        state = env.reset(seed=seed)
        encoder = build_state_encoder_from_env(
            env, encoder_type=cfg.state_encoder
        )
        while not env.done:
            mask = env.get_action_mask()
            vec = encoder.encode(state)
            action = online.select_greedy(
                states_to_tensor(vec, device=device),
                action_mask=torch.as_tensor(mask, dtype=torch.bool, device=device),
            )
            act = decode_action(action)
            a = int(state.supplier_availability)
            u = int(state.transport_state)
            rule_id = rule.select(state)
            records.append(
                {
                    "seed": seed,
                    "day": state.day,
                    "A": a,
                    "U": u,
                    "severity": severity_bucket(a, u),
                    "action_id": action,
                    "ell": act.ell,
                    "e": act.expedite,
                    "g": act.allocation,
                    "tuple": act.as_tuple(),
                    "agree_best": action == best_id,
                    "agree_rule": action == rule_id,
                    "rule_id": rule_id,
                    "best_id": best_id,
                }
            )
            state, _, _, _ = env.step(action)
    return records


def pct(n: int, d: int) -> str:
    if d <= 0:
        return "n/a"
    return f"{100.0 * n / d:.1f}%"


def summarize(records: List[dict], profile: str) -> List[str]:
    best_id = BEST_FIXED_BY_PROFILE.get(profile, 12)
    best_act = decode_action(best_id)
    n = len(records)
    lines = [
        f"DDQN forbid_e1 (ell,g) diagnosis",
        f"profile: {profile}",
        f"steps: {n}",
        f"best_fixed: id={best_id} {best_act.as_tuple()}",
        f"rule severe target: (2,0,1)  | rule normal: (0,0,1) | mild: (1,0,1)",
        "",
    ]

    # Sanity: no e=1
    e1 = sum(1 for r in records if r["e"] == 1)
    lines.append(f"e=1 count: {e1} ({pct(e1, n)})  [expect 0 under forbid_e1]")
    lines.append("")

    # Overall action distribution
    action_counts = Counter(r["action_id"] for r in records)
    lines.append("=== overall action distribution (9 e=0 actions) ===")
    for aid in sorted(action_counts):
        act = decode_action(aid)
        c = action_counts[aid]
        mark = ""
        if aid == best_id:
            mark = "  ← best_fixed"
        elif act.as_tuple() == (2, 0, 1):
            mark = "  ← rule severe"
        lines.append(
            f"  id={aid:2d} {act.as_tuple()}  {c:5d}  {pct(c, n)}{mark}"
        )
    lines.append("")

    # ell / g marginals
    ell_counts = Counter(r["ell"] for r in records)
    g_counts = Counter(r["g"] for r in records)
    lines.append("=== overall ell / g ===")
    for ell in (0, 1, 2):
        lines.append(f"  ell={ell}: {ell_counts[ell]:5d}  {pct(ell_counts[ell], n)}")
    for g in (0, 1, 2):
        tag = {0: "proportional", 1: "lowest_dos", 2: "value_penalty"}[g]
        lines.append(f"  g={g} ({tag}): {g_counts[g]:5d}  {pct(g_counts[g], n)}")
    lines.append("")

    agree_best = sum(1 for r in records if r["agree_best"])
    agree_rule = sum(1 for r in records if r["agree_rule"])
    lines.append("=== agreement ===")
    lines.append(f"  vs best_fixed_{best_id}: {agree_best}/{n}  {pct(agree_best, n)}")
    lines.append(f"  vs rule_based:         {agree_rule}/{n}  {pct(agree_rule, n)}")
    lines.append("")

    # By severity
    by_sev: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_sev[r["severity"]].append(r)

    lines.append("=== by disruption severity ===")
    for sev in ("normal", "mild", "severe"):
        rs = by_sev.get(sev, [])
        m = len(rs)
        lines.append(f"-- {sev} (n={m}, {pct(m, n)} of days) --")
        if m == 0:
            lines.append("  (none)")
            lines.append("")
            continue
        ell_c = Counter(r["ell"] for r in rs)
        g_c = Counter(r["g"] for r in rs)
        act_c = Counter(r["action_id"] for r in rs)
        lines.append(
            "  ell: "
            + ", ".join(f"{ell}={pct(ell_c[ell], m)}" for ell in (0, 1, 2))
        )
        lines.append(
            "  g:   "
            + ", ".join(f"{g}={pct(g_c[g], m)}" for g in (0, 1, 2))
        )
        lines.append("  top actions:")
        for aid, c in act_c.most_common(5):
            lines.append(
                f"    id={aid:2d} {decode_action(aid).as_tuple()}  {pct(c, m)}"
            )
        if sev == "severe":
            ell2 = ell_c[2]
            g1 = g_c[1]
            match_best = sum(1 for r in rs if r["action_id"] == best_id)
            match_rule_sev = sum(
                1 for r in rs if r["tuple"] == (2, 0, 1)
            )
            lines.append(
                f"  FOCUS severe: ell=2 rate={pct(ell2, m)} "
                f"(buffer high enough?) | g=1 rate={pct(g1, m)} "
                f"(lowest_dos stable?)"
            )
            lines.append(
                f"  FOCUS severe: best_fixed match={pct(match_best, m)} | "
                f"(2,0,1) match={pct(match_rule_sev, m)}"
            )
            under_buffer = m - ell2
            lines.append(
                f"  FOCUS severe: under-buffered days (ell<2)="
                f"{under_buffer} ({pct(under_buffer, m)})"
            )
        lines.append("")

    # Quick verdict hints
    severe = by_sev.get("severe", [])
    lines.append("=== quick verdict hints ===")
    if not severe:
        lines.append("  no severe days in sample")
    else:
        m = len(severe)
        ell2_rate = sum(1 for r in severe if r["ell"] == 2) / m
        g_spread = len({r["g"] for r in severe})
        g2_rate = sum(1 for r in severe if r["g"] == 2) / m
        if ell2_rate < 0.5:
            lines.append(
                f"  ell=2 only {ell2_rate:.0%} on severe → buffer likely TOO LOW"
            )
        else:
            lines.append(
                f"  ell=2 is {ell2_rate:.0%} on severe → buffer direction OK"
            )
        if g2_rate > 0.25 or g_spread == 3:
            lines.append(
                f"  g=2 rate={g2_rate:.0%}, distinct g values={g_spread} "
                f"on severe → allocation UNSTABLE / noisy"
            )
        else:
            lines.append(
                f"  g distribution on severe looks relatively concentrated "
                f"(g=2={g2_rate:.0%})"
            )
    return lines


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    profile = args.profile
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (
        CKPT_DIR / f"ddqn_{profile}_forbid_e1_best.pt"
    )
    log_path = Path(args.log_path) if args.log_path else (
        LOG_DIR / f"ddqn_{profile}_forbid_e1_ell_g_diag.log"
    )

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt_path}\n"
            f"Train first: python experiments/compare_ddqn_baselines.py "
            f"--forbid-expedite --profiles {profile}"
        )

    cfg = DQNTrainConfig(
        algorithm="ddqn",
        disruption_matrix_profile=profile,
        warmup_days=args.warmup_days,
        evaluation_days=args.evaluation_days,
        demand_intensity_scale=args.demand_intensity_scale,
        forbid_expedite=True,
        seed=args.seed,
        device=args.device,
    )
    device = torch.device(args.device)
    # Infer state dim from a throwaway env reset
    probe = build_env(cfg, seed=args.seed)
    state = probe.reset(seed=args.seed)
    encoder = build_state_encoder_from_env(probe)
    state_dim = encoder.dim

    online = build_q_network(state_dim=state_dim, device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    online.load_state_dict(ckpt["model_state_dict"])
    print(
        f"loaded {ckpt_path} (best_eval={ckpt.get('best_eval_profit')})",
        flush=True,
    )

    records = collect_traces(
        online,
        cfg,
        reps=args.reps,
        seed_base=args.seed,
        device=device,
    )
    lines = [
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
        f"checkpoint: {ckpt_path}",
        f"reps: {args.reps}  seed: {args.seed}  "
        f"demand_intensity_scale: {args.demand_intensity_scale}",
        "",
    ] + summarize(records, profile)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    log_path.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"Wrote {log_path}", flush=True)


if __name__ == "__main__":
    main()
