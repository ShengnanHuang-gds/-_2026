#!/usr/bin/env python3
"""Step 3 + 5: simulate 1000-day A_t / U_t paths and plot them."""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.disruption_chain import (
    SupplierAvailability,
    TransportState,
    get_transition_matrices,
    sample_next_supplier_availability,
    sample_next_transport_state,
)


def simulate_paths(
    num_days: int = 1000,
    seed: int = 42,
    matrix_profile: str = "pdf",
):
    rng = random.Random(seed)
    availability = SupplierAvailability.A0
    transport = TransportState.U0
    m_a, m_u = get_transition_matrices(matrix_profile)

    path_a: list[int] = []
    path_u: list[int] = []
    for _ in range(num_days):
        path_a.append(int(availability))
        path_u.append(int(transport))
        availability = sample_next_supplier_availability(
            availability, rng, matrix=m_a
        )
        transport = sample_next_transport_state(transport, rng, matrix=m_u)

    return path_a, path_u


def print_frequencies(path_a: list[int], path_u: list[int]) -> None:
    n = len(path_a)
    count_a = Counter(path_a)
    count_u = Counter(path_u)

    print(f"Empirical frequencies over {n} days (seed=42):")
    print("  A:", {k: f"{count_a.get(k, 0) / n:.4f}" for k in range(4)})
    print("  U:", {k: f"{count_u.get(k, 0) / n:.4f}" for k in range(4)})
    print("  A day counts:", dict(sorted(count_a.items())))
    print("  U day counts:", dict(sorted(count_u.items())))


def plot_paths(path_a: list[int], path_u: list[int], out_path: Path) -> None:
    days = list(range(1, len(path_a) + 1))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].step(days, path_a, where="post", color="steelblue")
    axes[0].set_ylabel("A_t")
    axes[0].set_yticks([0, 1, 2, 3])
    axes[0].set_yticklabels(["A0", "A1", "A2", "A3"])
    axes[0].set_title("Supplier availability path (1000 days)")
    axes[0].grid(True, alpha=0.3)

    axes[1].step(days, path_u, where="post", color="darkorange")
    axes[1].set_ylabel("U_t")
    axes[1].set_xlabel("Day")
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(["U0", "U1", "U2", "U3"])
    axes[1].set_title("Transport state path (1000 days)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def main() -> None:
    path_a, path_u = simulate_paths(num_days=1000, seed=42)
    print_frequencies(path_a, path_u)

    out = Path(__file__).resolve().parent / "figures" / "disruption_path_1000.png"
    plot_paths(path_a, path_u, out)


if __name__ == "__main__":
    main()
