#!/usr/bin/env python3
"""
Plot three-matrix (both_pdf / both_harsh_mild / both_harsh_strong) profit
bars for each RL algo with rule warm-start, vs baselines.

Data: logs/suite_ab_results_20260729_133915.csv (Suite B).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "logs" / "suite_ab_results_20260729_133915.csv"
OUT_DIR = ROOT / "figures"

SCENARIOS = ["both_pdf", "both_harsh_mild", "both_harsh_strong"]
SCENARIO_LABELS = ["pdf", "harsh_mild", "harsh_strong"]

# (algo_key_in_csv, output_stem, legend_name)
ALGOS = [
    ("dqn", "fig_three_matrix_dqn", "DQN (rule-ws)"),
    ("dueling", "fig_three_matrix_dueling", "Dueling (rule-ws)"),
    ("d3qn", "fig_three_matrix_d3qn", "D3QN (rule-ws)"),
    ("ppo", "fig_three_matrix_ppo", "PPO (rule-ws)"),
]


def load_rows() -> List[dict]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick(
    rows: List[dict], scenario: str, algo_prefix: str, warm: str = ""
) -> Tuple[float, float]:
    """Return (mean, ci) for matching row. algo_prefix matches start of algo name."""
    for row in rows:
        if row["scenario"] != scenario:
            continue
        algo = row["algo"]
        ws = row["warmstart"]
        if warm == "":
            if ws not in ("",):
                continue
            if algo.startswith(algo_prefix) or algo == algo_prefix:
                return float(row["mean_profit"]), float(row["ci95"])
        else:
            if ws != warm:
                continue
            if algo == algo_prefix:
                return float(row["mean_profit"]), float(row["ci95"])
    # baselines: fixed_0_(s,S), best_fixed_*, rule_based
    for row in rows:
        if row["scenario"] != scenario:
            continue
        if row["warmstart"] not in ("",):
            continue
        if algo_prefix == "fixed_0" and row["algo"].startswith("fixed_0"):
            return float(row["mean_profit"]), float(row["ci95"])
        if algo_prefix == "best_fixed" and row["algo"].startswith("best_fixed"):
            return float(row["mean_profit"]), float(row["ci95"])
        if algo_prefix == "rule_based" and row["algo"] == "rule_based":
            return float(row["mean_profit"]), float(row["ci95"])
    raise KeyError(f"missing {scenario=} {algo_prefix=} {warm=}")


def plot_one(
    rows: List[dict],
    algo: str,
    out_stem: str,
    rl_label: str,
) -> Path:
    series = {
        rl_label: {
            "mean": [],
            "ci": [],
            "color": "#2c7bb6",
        },
        "fixed_0": {
            "mean": [],
            "ci": [],
            "color": "#bdbdbd",
        },
        "best_fixed": {
            "mean": [],
            "ci": [],
            "color": "#fdae61",
        },
        "rule_based": {
            "mean": [],
            "ci": [],
            "color": "#d7191c",
        },
    }
    for scen in SCENARIOS:
        m, c = pick(rows, scen, algo, warm="Y")
        series[rl_label]["mean"].append(m)
        series[rl_label]["ci"].append(c)
        m, c = pick(rows, scen, "fixed_0")
        series["fixed_0"]["mean"].append(m)
        series["fixed_0"]["ci"].append(c)
        m, c = pick(rows, scen, "best_fixed")
        series["best_fixed"]["mean"].append(m)
        series["best_fixed"]["ci"].append(c)
        m, c = pick(rows, scen, "rule_based")
        series["rule_based"]["mean"].append(m)
        series["rule_based"]["ci"].append(c)

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
    n_series = len(series)
    x = np.arange(n_scen)
    width = 0.2
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for i, (name, data) in enumerate(series.items()):
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
    ax.set_title(
        f"{rl_label} vs baselines by matrix (Suite B, 30-rep)"
    )
    ax.set_ylim(1.85, 2.65)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{out_stem}.pdf"
    png = OUT_DIR / f"{out_stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return pdf


def main() -> None:
    rows = load_rows()
    for algo, stem, label in ALGOS:
        path = plot_one(rows, algo, stem, label)
        print(f"saved → {path}")


if __name__ == "__main__":
    main()
