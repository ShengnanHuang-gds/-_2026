#!/usr/bin/env python3
"""Generate figures/fig_three_matrix_profit.pdf for the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "fig_three_matrix_profit.pdf"

# Formal 30-rep (seeds 42–71), demand_scale=1.2, both mode.
# RL column = DDQN rule-warmstart (forbid_e1), same protocol across matrices.
# Source: logs/ddqn_*_forbid_e1_rule_ws_vs_baselines_30rep.log
#         logs/ddqn_rule_ws_three_matrix_30rep_summary.log
SCENARIOS = ["pdf", "harsh_mild", "harsh_strong"]
SCENARIO_LABELS = ["pdf", "harsh_mild", "harsh_strong"]

# order: RL, fixed_0, best_fixed, rule
SERIES = {
    "DDQN (rule-ws)": {
        "mean": [2574863.8, 2486155.4, 2136920.3],
        "ci": [5639.7, 25917.7, 59921.8],
        "color": "#2c7bb6",
    },
    "fixed_0": {
        "mean": [2573988.6, 2453409.2, 1987242.8],
        "ci": [6246.9, 30976.5, 67121.2],
        "color": "#bdbdbd",
    },
    "best_fixed": {
        "mean": [2574056.1, 2466956.2, 2155239.6],
        "ci": [6180.4, 25417.1, 57518.1],
        "color": "#fdae61",
    },
    "rule_based": {
        "mean": [2574862.7, 2486155.4, 2143907.0],
        "ci": [5639.5, 25917.7, 59791.7],
        "color": "#d7191c",
    },
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    n_scen = len(SCENARIOS)
    n_series = len(SERIES)
    x = np.arange(n_scen)
    width = 0.2
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for i, (name, data) in enumerate(SERIES.items()):
        means = np.asarray(data["mean"], dtype=float) / 1e6
        cis = np.asarray(data["ci"], dtype=float) / 1e6
        ax.bar(
            x + offsets[i],
            means,
            width=width * 0.92,
            yerr=cis,
            capsize=3,
            label=name,
            color=data["color"],
            edgecolor="white",
            linewidth=0.4,
            error_kw={"elinewidth": 0.9, "ecolor": "#333333"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_LABELS)
    ax.set_ylabel("Total profit (millions)")
    ax.set_xlabel("Markov disruption matrix (both mode)")
    ax.set_title("Long-horizon both-mode profit by matrix profile (30-rep)")
    ax.set_ylim(1.85, 2.65)
    ax.axhline(0, color="none")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    # also keep a copy under experiments/figures for convenience
    alt = ROOT / "experiments" / "figures" / "fig_three_matrix_profit.pdf"
    alt.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(alt, bbox_inches="tight")
    png = OUT.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"saved → {OUT}")
    print(f"saved → {alt}")
    print(f"saved → {png}")


if __name__ == "__main__":
    main()
