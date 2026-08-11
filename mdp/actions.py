from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# PDF MDP Action Design Guide Section 7 (coarse):
#   a_t = (ell_t, e_t, g_t), |A| = 3 x 2 x 3 = 18
#
# Plan A (fine): global buffer + local focus
#   a_t = (ell_t, f_t, e_t, g_t), |A| = 3 x 6 x 2 x 3 = 108
#   f_t=none recovers the coarse policy when (ell,e,g) match.


NUM_BUFFER_LEVELS = 3
NUM_EXPEDITE_CHOICES = 2
NUM_ALLOCATION_RULES = 3
NUM_FOCUS = 6

ACTION_SPACE_COARSE = "coarse"
ACTION_SPACE_FINE = "fine"
VALID_ACTION_SPACES = frozenset({ACTION_SPACE_COARSE, ACTION_SPACE_FINE})

NUM_ACTIONS_COARSE = (
    NUM_BUFFER_LEVELS * NUM_EXPEDITE_CHOICES * NUM_ALLOCATION_RULES
)  # 18
NUM_ACTIONS_FINE = (
    NUM_BUFFER_LEVELS
    * NUM_FOCUS
    * NUM_EXPEDITE_CHOICES
    * NUM_ALLOCATION_RULES
)  # 108

# Backward-compatible alias (coarse / PDF default).
NUM_ACTIONS = NUM_ACTIONS_COARSE

# Table: ell_t -> (eta^{C,s}, eta^{C,S}, eta^{F})
BUFFER_MULTIPLIERS: Dict[int, Tuple[float, float, float]] = {
    0: (1.00, 1.00, 1.00),
    1: (1.10, 1.10, 1.05),
    2: (1.25, 1.25, 1.10),
}

ALLOCATION_RULE_NAMES: Dict[int, str] = {
    0: "proportional",
    1: "lowest_dos",
    2: "value_penalty",
}

FOCUS_NONE = 0
FOCUS_TIGHTEST_SKU = 1
FOCUS_ALL_TIGHT_SKUS = 2
FOCUS_CW_ONLY = 3
FOCUS_FW_ONLY = 4
FOCUS_PROTECT_LOW_FW = 5

FOCUS_NAMES: Dict[int, str] = {
    FOCUS_NONE: "none",
    FOCUS_TIGHTEST_SKU: "boost_tightest_sku",
    FOCUS_ALL_TIGHT_SKUS: "boost_all_tight_skus",
    FOCUS_CW_ONLY: "boost_cw_only",
    FOCUS_FW_ONLY: "boost_fw_only",
    FOCUS_PROTECT_LOW_FW: "protect_low_fw",
}

TIGHT_DOS_THRESHOLD = 1.0


def num_actions_for(space: str) -> int:
    space = _normalize_space(space)
    if space == ACTION_SPACE_FINE:
        return NUM_ACTIONS_FINE
    return NUM_ACTIONS_COARSE


def _normalize_space(space: str) -> str:
    key = str(space).lower()
    if key not in VALID_ACTION_SPACES:
        raise ValueError(
            f"unknown action_space {space!r}; "
            f"expected one of {sorted(VALID_ACTION_SPACES)}"
        )
    return key


@dataclass(frozen=True)
class MDPAction:
    """
    Discrete risk-response action.

    Coarse: (ell, e, g) with focus=0.
    Fine:   (ell, focus, e, g) — local buffer focus on top of global ell.
    """

    ell: int
    expedite: int
    allocation: int
    focus: int = FOCUS_NONE

    def __post_init__(self) -> None:
        if self.ell not in BUFFER_MULTIPLIERS:
            raise ValueError(f"ell must be in {{0,1,2}}, got {self.ell}")
        if self.expedite not in (0, 1):
            raise ValueError(f"expedite must be in {{0,1}}, got {self.expedite}")
        if self.allocation not in ALLOCATION_RULE_NAMES:
            raise ValueError(
                f"allocation must be in {{0,1,2}}, got {self.allocation}"
            )
        if self.focus not in FOCUS_NAMES:
            raise ValueError(
                f"focus must be in {{0..{NUM_FOCUS - 1}}}, got {self.focus}"
            )

    def action_id_for(self, space: str = ACTION_SPACE_COARSE) -> int:
        return encode_action(
            self.ell,
            self.expedite,
            self.allocation,
            focus=self.focus,
            space=space,
        )

    @property
    def action_id(self) -> int:
        """Coarse encoding; requires focus=0."""
        if self.focus != FOCUS_NONE:
            raise ValueError(
                "action_id (coarse) requires focus=0; use action_id_for('fine')"
            )
        return self.action_id_for(ACTION_SPACE_COARSE)

    @property
    def eta_central_reorder(self) -> float:
        return BUFFER_MULTIPLIERS[self.ell][0]

    @property
    def eta_central_order_up_to(self) -> float:
        return BUFFER_MULTIPLIERS[self.ell][1]

    @property
    def eta_forward_reorder(self) -> float:
        return BUFFER_MULTIPLIERS[self.ell][2]

    @property
    def allocation_rule_name(self) -> str:
        return ALLOCATION_RULE_NAMES[self.allocation]

    @property
    def focus_name(self) -> str:
        return FOCUS_NAMES[self.focus]

    @property
    def is_baseline(self) -> bool:
        """Phase-2 path only for exact (0,0,0) with no local focus."""
        return (
            self.ell == 0
            and self.expedite == 0
            and self.allocation == 0
            and self.focus == FOCUS_NONE
        )

    def as_tuple(self) -> Tuple[int, ...]:
        """(ell, e, g) if focus=0 else (ell, focus, e, g)."""
        if self.focus == FOCUS_NONE:
            return (self.ell, self.expedite, self.allocation)
        return (self.ell, self.focus, self.expedite, self.allocation)

    def as_full_tuple(self) -> Tuple[int, int, int, int]:
        return (self.ell, self.focus, self.expedite, self.allocation)


def decode_action(
    action_id: int,
    *,
    space: str = ACTION_SPACE_COARSE,
) -> MDPAction:
    """Map action_id to MDPAction under the given action space."""
    if not isinstance(action_id, int) or isinstance(action_id, bool):
        raise TypeError(f"action_id must be int, got {type(action_id).__name__}")
    space = _normalize_space(space)
    n = num_actions_for(space)
    if action_id < 0 or action_id >= n:
        raise ValueError(f"action_id must be in 0..{n - 1}, got {action_id}")

    if space == ACTION_SPACE_COARSE:
        ell = action_id // (NUM_EXPEDITE_CHOICES * NUM_ALLOCATION_RULES)
        remainder = action_id % (NUM_EXPEDITE_CHOICES * NUM_ALLOCATION_RULES)
        expedite = remainder // NUM_ALLOCATION_RULES
        allocation = remainder % NUM_ALLOCATION_RULES
        return MDPAction(
            ell=ell, expedite=expedite, allocation=allocation, focus=FOCUS_NONE
        )

    # fine: ell, focus, e, g
    stride_eg = NUM_EXPEDITE_CHOICES * NUM_ALLOCATION_RULES  # 6
    stride_feg = NUM_FOCUS * stride_eg  # 36
    ell = action_id // stride_feg
    rem = action_id % stride_feg
    focus = rem // stride_eg
    rem2 = rem % stride_eg
    expedite = rem2 // NUM_ALLOCATION_RULES
    allocation = rem2 % NUM_ALLOCATION_RULES
    return MDPAction(
        ell=ell, expedite=expedite, allocation=allocation, focus=focus
    )


def encode_action(
    ell: int,
    expedite: int,
    allocation: int,
    focus: int = FOCUS_NONE,
    *,
    space: str = ACTION_SPACE_COARSE,
) -> int:
    """Map components to action_id under the given action space."""
    space = _normalize_space(space)
    action = MDPAction(
        ell=ell, expedite=expedite, allocation=allocation, focus=focus
    )
    if space == ACTION_SPACE_COARSE:
        if focus != FOCUS_NONE:
            raise ValueError("coarse action space requires focus=0")
        return (
            action.ell * (NUM_EXPEDITE_CHOICES * NUM_ALLOCATION_RULES)
            + action.expedite * NUM_ALLOCATION_RULES
            + action.allocation
        )
    stride_eg = NUM_EXPEDITE_CHOICES * NUM_ALLOCATION_RULES
    stride_feg = NUM_FOCUS * stride_eg
    return (
        action.ell * stride_feg
        + action.focus * stride_eg
        + action.expedite * NUM_ALLOCATION_RULES
        + action.allocation
    )


def all_actions(*, space: str = ACTION_SPACE_COARSE) -> List[MDPAction]:
    n = num_actions_for(space)
    return [decode_action(i, space=space) for i in range(n)]


def baseline_action() -> MDPAction:
    """Baseline a=(0,0,0) with action_id 0."""
    return decode_action(0, space=ACTION_SPACE_COARSE)


def coarse_id_to_fine_id(coarse_id: int) -> int:
    """Embed a coarse (ell,e,g) action as fine with focus=none."""
    a = decode_action(coarse_id, space=ACTION_SPACE_COARSE)
    return encode_action(
        a.ell, a.expedite, a.allocation, focus=FOCUS_NONE, space=ACTION_SPACE_FINE
    )


def resolve_product_buffer_levels(
    action: MDPAction,
    *,
    central_inventory: Sequence[int],
    forward_inventory: Sequence[Sequence[int]],
    fw_daily_demand_means: Sequence[Sequence[float]],
    eps: float = 1e-6,
    tight_dos_threshold: float = TIGHT_DOS_THRESHOLD,
) -> Tuple[List[int], List[List[int]]]:
    """
    Resolve per-SKU CW ell and per-FW-per-SKU FW ell from (ell_global, focus).

    Returns:
      cw_ell: length K
      fw_ell: N x K
    """
    k_products = len(central_inventory)
    n_fw = len(forward_inventory)
    if n_fw == 0:
        raise ValueError("forward_inventory must be non-empty")
    if len(fw_daily_demand_means) != n_fw:
        raise ValueError("fw_daily_demand_means length must match forward_inventory")

    ell0 = int(action.ell)
    focus = int(action.focus)

    def bump(level: int) -> int:
        return min(2, level + 1)

    # System DOS per SKU (CW + all FW on-hand) / total demand
    sku_dos: List[float] = []
    for k in range(k_products):
        inv = float(central_inventory[k])
        mu = 0.0
        for i in range(n_fw):
            inv += float(forward_inventory[i][k])
            mu += float(fw_daily_demand_means[i][k])
        sku_dos.append(inv / (mu + eps))

    cw_ell = [ell0] * k_products
    fw_ell = [[ell0 for _ in range(k_products)] for _ in range(n_fw)]

    if focus == FOCUS_NONE:
        return cw_ell, fw_ell

    if focus == FOCUS_CW_ONLY:
        fw_ell = [[0 for _ in range(k_products)] for _ in range(n_fw)]
        return cw_ell, fw_ell

    if focus == FOCUS_FW_ONLY:
        cw_ell = [0] * k_products
        return cw_ell, fw_ell

    if focus == FOCUS_TIGHTEST_SKU:
        k_star = min(range(k_products), key=lambda k: sku_dos[k])
        cw_ell[k_star] = bump(ell0)
        for i in range(n_fw):
            fw_ell[i][k_star] = bump(ell0)
        return cw_ell, fw_ell

    if focus == FOCUS_ALL_TIGHT_SKUS:
        for k in range(k_products):
            if sku_dos[k] < tight_dos_threshold:
                cw_ell[k] = bump(ell0)
                for i in range(n_fw):
                    fw_ell[i][k] = bump(ell0)
        return cw_ell, fw_ell

    if focus == FOCUS_PROTECT_LOW_FW:
        for i in range(n_fw):
            dos_vals = [
                float(forward_inventory[i][k])
                / (float(fw_daily_demand_means[i][k]) + eps)
                for k in range(k_products)
            ]
            if (sum(dos_vals) / k_products) < tight_dos_threshold:
                for k in range(k_products):
                    fw_ell[i][k] = bump(ell0)
        return cw_ell, fw_ell

    raise ValueError(f"unhandled focus={focus}")
