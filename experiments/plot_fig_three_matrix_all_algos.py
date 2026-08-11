#!/usr/bin/env python3
"""
One figure: 1x3 panels (pdf / harsh_mild / harsh_strong).
Bars = five RL algos (rule-ws); horizontal lines = baselines.
Data: Suite B CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "logs" / "suite_ab_results_20260729_133915.csv"
OUT = ROOT / "figures" / "fig_three_matrix_all_algos.pdf"

SCENARIOS = ["both_pdf", "both_harsh_mild", "both_harsh_strong"]
SCENARIO_TITLES = ["pdf", "harsh_mild", "harsh_strong"]
ALGOS = ["dqn", "ddqn", "dueling", "d3qn", "ppo"]
ALGO_LABELS = ["DQN", "DDQN", "Dueling", "D3QN", "PPO"]
COLORS = ["#4c78a8", "#2c7bb6", "#54a24b", "#e45756", "#b279a2"]


def load_rows() -> List[dict]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_rl(rows: List[dict], scenario: str, algo: str) -> Tuple[float, float]:
    for row in rows:
        if (
            row["scenario"] == scenario
            and row["algo"] == algo
            and row["warmstart"] == "Y"
        ):
            return float(row["mean_profit"]), float(row["ci95"])
    raise KeyError(f"{scenario} {algo} Y")


def get_baseline(rows: List[dict], scenario: str, kind: str) -> Tuple[float, float]:
    for row in rows:
        if row["scenario"] != scenario or row["warmstart"] not in ("",):
            continue
        algo = row["algo"]
        if kind == "fixed_0" and algo.startswith("fixed_0"):
            return float(row["mean_profit"]), float(row["ci95"])
        if kind == "best_fixed" and algo.startswith("best_fixed"):
            return float(row["mean_profit"]), float(row["ci95"])
        if kind == "rule" and algo == "rule_based":
            return float(row["mean_profit"]), float(row["ci95"])
    raise KeyError(f"{scenario} {kind}")


def main() -> None:
    rows = load_rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)

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
    x = np.arange(len(ALGOS))
    width = 0.62

    line_styles = {
        "fixed_0": ("#888888", "--", "fixed_0"),
        "rule": ("#d7191c", ":", "rule_based"),
        "best_fixed": ("#fdae61", "-.", "best_fixed"),
    }

    for ax, scen, title in zip(axes, SCENARIOS, SCENARIO_TITLES):
        means, cis = [], []
        for algo in ALGOS:
            m, c = get_rl(rows, scen, algo)
            means.append(m / 1e6)
            cis.append(c / 1e6)
        ax.bar(
            x,
            means,
            width=width,
            yerr=cis,
            capsize=2.5,
            color=COLORS,
            edgecolor="white",
            linewidth=0.4,
            error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
        )
        for kind, (color, ls, label) in line_styles.items():
            bm, _ = get_baseline(rows, scen, kind)
            ax.axhline(
                bm / 1e6,
                color=color,
                linestyle=ls,
                linewidth=1.4,
                label=label if ax is axes[0] else None,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(ALGO_LABELS, rotation=25, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("Total profit (millions)")
    axes[0].set_ylim(1.90, 2.62)
    # legend: algo colors + baseline lines
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    handles = [
        Patch(facecolor=c, edgecolor="white", label=l)
        for c, l in zip(COLORS, ALGO_LABELS)
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
        "Suite B: RL (rule-ws) vs baselines across Markov matrices",
        y=1.14,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"saved → {OUT}")
    print(f"saved → {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
