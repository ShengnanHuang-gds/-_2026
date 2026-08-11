from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Sequence

# PDF MDP Action Design Guide Section 3: A_t / U_t Markov chains.


class SupplierAvailability(IntEnum):
    """Supplier availability state A_t."""

    A0 = 0  # normal: release 100%
    A1 = 1  # mild shortage: release 80%
    A2 = 2  # severe shortage: release 40%
    A3 = 3  # full shutdown: release 0%


class TransportState(IntEnum):
    """Supplier-to-CW transport state U_t."""

    U0 = 0  # normal: extra delay 0
    U1 = 1  # mild delay: +1 day
    U2 = 2  # severe delay: +2 days
    U3 = 3  # blocked: shipments wait, do not enter pipeline


# Release fraction alpha(A_t)
RELEASE_RATIO: dict[SupplierAvailability, float] = {
    SupplierAvailability.A0: 1.00,
    SupplierAvailability.A1: 0.80,
    SupplierAvailability.A2: 0.40,
    SupplierAvailability.A3: 0.00,
}

# Extra delay delta(U_t) for U0–U2. U3 is handled separately (waiting queue).
EXTRA_DELAY: dict[TransportState, int] = {
    TransportState.U0: 0,
    TransportState.U1: 1,
    TransportState.U2: 2,
}

# ---------------------------------------------------------------------------
# Transition-matrix profiles (swappable for stress tests)
# ---------------------------------------------------------------------------
#
# "pdf"         : course PDF default (mild; ~97% time in A0/U0)
# "harsh_mild"  : leave normal more often; abnormal states stickier
# "harsh_strong": frequent / persistent disruption (aggressive stress)

TransitionMatrix = List[List[float]]

M_A_PDF: TransitionMatrix = [
    [0.9950, 0.0040, 0.0008, 0.0002],
    [0.3000, 0.6000, 0.0800, 0.0200],
    [0.0800, 0.2500, 0.5500, 0.1200],
    [0.0200, 0.0800, 0.2500, 0.6500],
]

M_U_PDF: TransitionMatrix = [
    [0.9920, 0.0060, 0.0015, 0.0005],
    [0.4000, 0.5000, 0.0800, 0.0200],
    [0.1500, 0.3000, 0.4500, 0.1000],
    [0.0500, 0.1500, 0.3000, 0.5000],
]

M_A_HARSH_MILD: TransitionMatrix = [
    [0.9700, 0.0200, 0.0070, 0.0030],
    [0.2500, 0.5500, 0.1500, 0.0500],
    [0.0500, 0.2000, 0.5500, 0.2000],
    [0.0200, 0.0800, 0.2000, 0.7000],
]

M_U_HARSH_MILD: TransitionMatrix = [
    [0.9600, 0.0250, 0.0100, 0.0050],
    [0.3000, 0.5000, 0.1500, 0.0500],
    [0.1000, 0.2500, 0.4500, 0.2000],
    [0.0500, 0.1000, 0.2500, 0.6000],
]

M_A_HARSH_STRONG: TransitionMatrix = [
    [0.9000, 0.0600, 0.0300, 0.0100],
    [0.2000, 0.5000, 0.2000, 0.1000],
    [0.0500, 0.1500, 0.5000, 0.3000],
    [0.0200, 0.0800, 0.2000, 0.7000],
]

M_U_HARSH_STRONG: TransitionMatrix = [
    [0.8800, 0.0700, 0.0300, 0.0200],
    [0.2500, 0.4500, 0.2000, 0.1000],
    [0.1000, 0.2000, 0.4500, 0.2500],
    [0.0500, 0.1000, 0.2500, 0.6000],
]

VALID_MATRIX_PROFILES = frozenset({"pdf", "harsh_mild", "harsh_strong"})

MATRIX_PROFILES: dict[str, dict[str, object]] = {
    "pdf": {
        "M_A": M_A_PDF,
        "M_U": M_U_PDF,
        "description": "PDF default (mild; ~97% time in A0/U0)",
    },
    "harsh_mild": {
        "M_A": M_A_HARSH_MILD,
        "M_U": M_U_HARSH_MILD,
        "description": "Stress: leave A0/U0 more often; ~20% time disrupted",
    },
    "harsh_strong": {
        "M_A": M_A_HARSH_STRONG,
        "M_U": M_U_HARSH_STRONG,
        "description": "Aggressive stress: ~50%+ time disrupted",
    },
}

# Module-level defaults remain the PDF matrices (backward compatible).
M_A: TransitionMatrix = M_A_PDF
M_U: TransitionMatrix = M_U_PDF


def get_transition_matrices(
    profile: str = "pdf",
) -> tuple[TransitionMatrix, TransitionMatrix]:
    """Return (M_A, M_U) copies for a named profile."""
    if profile not in VALID_MATRIX_PROFILES:
        raise ValueError(
            f"unknown disruption_matrix_profile: {profile!r}; "
            f"expected one of {sorted(VALID_MATRIX_PROFILES)}"
        )
    entry = MATRIX_PROFILES[profile]
    m_a = [list(row) for row in entry["M_A"]]  # type: ignore[index]
    m_u = [list(row) for row in entry["M_U"]]  # type: ignore[index]
    return m_a, m_u


@dataclass(frozen=True)
class DisruptionState:
    """Joint disruption state Z_t = (A_t, U_t)."""

    supplier_availability: SupplierAvailability
    transport_state: TransportState

    @property
    def release_ratio(self) -> float:
        return RELEASE_RATIO[self.supplier_availability]

    @property
    def is_transport_blocked(self) -> bool:
        return self.transport_state == TransportState.U3

    @property
    def extra_delay(self) -> int:
        """Extra lead-time days for U0–U2. Raises if transport is blocked."""
        if self.is_transport_blocked:
            raise ValueError("U3 has no finite extra delay; use waiting queue")
        return EXTRA_DELAY[self.transport_state]

    def code(self) -> str:
        """Short label like Z_21."""
        return (
            f"Z_{self.supplier_availability.value}"
            f"{self.transport_state.value}"
        )


def default_disruption_state() -> DisruptionState:
    """Start from fully normal operations."""
    return DisruptionState(
        supplier_availability=SupplierAvailability.A0,
        transport_state=TransportState.U0,
    )


def _validate_row_stochastic(matrix: Sequence[Sequence[float]], name: str) -> None:
    for row_index, row in enumerate(matrix):
        if len(row) != 4:
            raise ValueError(f"{name} row {row_index} must have length 4")
        row_sum = sum(row)
        if abs(row_sum - 1.0) > 1e-9:
            raise ValueError(
                f"{name} row {row_index} sums to {row_sum}, expected 1.0"
            )


def validate_transition_matrices(
    m_a: Optional[Sequence[Sequence[float]]] = None,
    m_u: Optional[Sequence[Sequence[float]]] = None,
) -> None:
    """Check that given (or default PDF) matrices are 4x4 row-stochastic."""
    _validate_row_stochastic(m_a if m_a is not None else M_A, "M_A")
    _validate_row_stochastic(m_u if m_u is not None else M_U, "M_U")


def validate_all_matrix_profiles() -> None:
    """Validate every named matrix profile."""
    for profile_name in sorted(VALID_MATRIX_PROFILES):
        m_a, m_u = get_transition_matrices(profile_name)
        _validate_row_stochastic(m_a, f"M_A[{profile_name}]")
        _validate_row_stochastic(m_u, f"M_U[{profile_name}]")


def mean_sojourn_time(self_transition_probability: float) -> float:
    """E[T] = 1 / (1 - p) for geometric sojourn with stay probability p."""
    if not 0.0 <= self_transition_probability < 1.0:
        raise ValueError("self-transition probability must be in [0, 1)")
    return 1.0 / (1.0 - self_transition_probability)


def _sample_from_row(
    row: Sequence[float],
    random_generator: random.Random,
) -> int:
    draw = random_generator.random()
    cumulative = 0.0
    for index, probability in enumerate(row):
        cumulative += probability
        if draw <= cumulative:
            return index
    return len(row) - 1


def sample_next_supplier_availability(
    current: SupplierAvailability,
    random_generator: random.Random,
    matrix: Optional[Sequence[Sequence[float]]] = None,
) -> SupplierAvailability:
    """Sample A_{t+1} from M^A (or a provided matrix) given A_t."""
    active = matrix if matrix is not None else M_A
    next_index = _sample_from_row(active[current.value], random_generator)
    return SupplierAvailability(next_index)


def sample_next_transport_state(
    current: TransportState,
    random_generator: random.Random,
    matrix: Optional[Sequence[Sequence[float]]] = None,
) -> TransportState:
    """Sample U_{t+1} from M^U (or a provided matrix) given U_t."""
    active = matrix if matrix is not None else M_U
    next_index = _sample_from_row(active[current.value], random_generator)
    return TransportState(next_index)


def sample_next_disruption_state(
    current: DisruptionState,
    random_generator: random.Random,
    m_a: Optional[Sequence[Sequence[float]]] = None,
    m_u: Optional[Sequence[Sequence[float]]] = None,
) -> DisruptionState:
    """Sample independent A_{t+1} and U_{t+1}, forming Z_{t+1}."""
    return DisruptionState(
        supplier_availability=sample_next_supplier_availability(
            current.supplier_availability,
            random_generator,
            matrix=m_a,
        ),
        transport_state=sample_next_transport_state(
            current.transport_state,
            random_generator,
            matrix=m_u,
        ),
    )


def compute_realized_lead_time(
    base_lead_time: int,
    transport_state: TransportState,
    expedite: int = 0,
) -> int:
    """
    Realized lead time L = max{1, L0 + delta(U) - e} for U0–U2.

    U3 must not call this; blocked shipments use the waiting queue.
    expedite e ∈ {0,1} shortens lead time by at most one day (PDF eq. 47).
    """
    if transport_state == TransportState.U3:
        raise ValueError("Cannot compute finite lead time under U3")
    if base_lead_time < 1:
        raise ValueError("base_lead_time must be >= 1")
    if expedite not in (0, 1):
        raise ValueError("expedite must be 0 or 1")
    return max(1, base_lead_time + EXTRA_DELAY[transport_state] - expedite)


def released_quantity(order_quantity: int, availability: SupplierAvailability) -> int:
    """Integer units released today from an outstanding quantity."""
    if order_quantity < 0:
        raise ValueError("order_quantity must be non-negative")
    ratio = RELEASE_RATIO[availability]
    return int(order_quantity * ratio)
