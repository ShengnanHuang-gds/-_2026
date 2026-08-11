"""
Action masks for illegal / ineffective risk-response actions.

- U3 (transport fully blocked): expedite cannot move goods → mask e=1
- Optional forbid_expedite: always mask e=1 (RL training ablation)
- Optional rule_buffer_mask: if severe or low DOS, only allow ell=2
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from mdp.actions import (
    ACTION_SPACE_COARSE,
    NUM_ACTIONS,
    all_actions,
    decode_action,
    encode_action,
    num_actions_for,
)

DEFAULT_DOS_THRESHOLD = 1.0


def inventory_min_dos(
    state: object,
    fw_daily_demand_means: Sequence[Sequence[float]],
    *,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """
    Return (CW min product DOS, FW min cell DOS) using mean daily demand.
    """
    cw = getattr(state, "central_inventory")
    fw = getattr(state, "forward_inventory")
    n_fw = len(fw)
    k_products = len(cw)
    if len(fw_daily_demand_means) != n_fw:
        raise ValueError("fw_daily_demand_means length must match forward_inventory")

    cw_demand = [
        sum(float(fw_daily_demand_means[i][k]) for i in range(n_fw))
        for k in range(k_products)
    ]
    cw_min = min(
        float(cw[k]) / (cw_demand[k] + eps) for k in range(k_products)
    )
    fw_min = float("inf")
    for i in range(n_fw):
        for k in range(k_products):
            dos = float(fw[i][k]) / (float(fw_daily_demand_means[i][k]) + eps)
            if dos < fw_min:
                fw_min = dos
    return cw_min, fw_min


def is_severe_disruption(state: object) -> bool:
    a = int(getattr(state, "supplier_availability"))
    u = int(getattr(state, "transport_state"))
    return a >= 2 or u >= 2


def requires_high_buffer(
    state: object,
    fw_daily_demand_means: Optional[Sequence[Sequence[float]]] = None,
    *,
    dos_threshold: float = DEFAULT_DOS_THRESHOLD,
) -> bool:
    """
    True when business rules force ell=2:
      severe (A>=2 or U>=2) OR CW/FW min DOS below threshold.
    """
    if is_severe_disruption(state):
        return True
    if fw_daily_demand_means is None:
        return False
    cw_min, fw_min = inventory_min_dos(state, fw_daily_demand_means)
    return cw_min < dos_threshold or fw_min < dos_threshold


def action_mask_for_transport(
    transport_state: int,
    num_actions: int = NUM_ACTIONS,
    *,
    forbid_expedite: bool = False,
    space: str = ACTION_SPACE_COARSE,
) -> np.ndarray:
    """
    Return boolean mask of shape (num_actions,); True = legal.

    U3 => forbid every action with expedite=1.
    forbid_expedite=True => forbid e=1 regardless of U.
    """
    expected = num_actions_for(space)
    if num_actions != expected:
        num_actions = expected
    mask = np.ones(num_actions, dtype=bool)
    block_expedite = bool(forbid_expedite) or int(transport_state) == 3
    if not block_expedite:
        return mask
    for action in all_actions(space=space):
        if action.expedite == 1:
            mask[action.action_id_for(space)] = False
    return mask


def apply_rule_buffer_mask(
    mask: np.ndarray,
    *,
    force_ell2: bool,
    space: str = ACTION_SPACE_COARSE,
) -> np.ndarray:
    """If force_ell2, zero out every action with ell != 2."""
    if not force_ell2:
        return mask
    out = mask.copy()
    for action in all_actions(space=space):
        if action.ell != 2:
            out[action.action_id_for(space)] = False
    return out


def action_mask(
    state: Union[object, int],
    *,
    forbid_expedite: bool = False,
    space: str = ACTION_SPACE_COARSE,
    rule_buffer_mask: bool = False,
    fw_daily_demand_means: Optional[Sequence[Sequence[float]]] = None,
    dos_threshold: float = DEFAULT_DOS_THRESHOLD,
) -> np.ndarray:
    """Mask from an MDPState (or raw transport int for legacy callers)."""
    if hasattr(state, "transport_state"):
        transport_state = getattr(state, "transport_state")
    else:
        transport_state = int(state)
        # Raw int cannot evaluate DOS / severity beyond U.
        rule_buffer_mask = False

    mask = action_mask_for_transport(
        transport_state,
        num_actions=num_actions_for(space),
        forbid_expedite=forbid_expedite,
        space=space,
    )
    if rule_buffer_mask:
        force = requires_high_buffer(
            state,
            fw_daily_demand_means,
            dos_threshold=dos_threshold,
        )
        mask = apply_rule_buffer_mask(mask, force_ell2=force, space=space)
    return mask


def is_action_legal(
    action_id: int,
    state: Union[object, int],
    *,
    forbid_expedite: bool = False,
    space: str = ACTION_SPACE_COARSE,
    rule_buffer_mask: bool = False,
    fw_daily_demand_means: Optional[Sequence[Sequence[float]]] = None,
    dos_threshold: float = DEFAULT_DOS_THRESHOLD,
) -> bool:
    mask = action_mask(
        state,
        forbid_expedite=forbid_expedite,
        space=space,
        rule_buffer_mask=rule_buffer_mask,
        fw_daily_demand_means=fw_daily_demand_means,
        dos_threshold=dos_threshold,
    )
    if not (0 <= action_id < len(mask)):
        return False
    return bool(mask[action_id])


def legal_action_ids(
    state: Union[object, int],
    *,
    forbid_expedite: bool = False,
    space: str = ACTION_SPACE_COARSE,
    rule_buffer_mask: bool = False,
    fw_daily_demand_means: Optional[Sequence[Sequence[float]]] = None,
    dos_threshold: float = DEFAULT_DOS_THRESHOLD,
) -> List[int]:
    mask = action_mask(
        state,
        forbid_expedite=forbid_expedite,
        space=space,
        rule_buffer_mask=rule_buffer_mask,
        fw_daily_demand_means=fw_daily_demand_means,
        dos_threshold=dos_threshold,
    )
    return [i for i, ok in enumerate(mask) if ok]


def remap_illegal_action(
    action_id: int,
    state: Union[object, int],
    *,
    forbid_expedite: bool = False,
    space: str = ACTION_SPACE_COARSE,
    rule_buffer_mask: bool = False,
    fw_daily_demand_means: Optional[Sequence[Sequence[float]]] = None,
    dos_threshold: float = DEFAULT_DOS_THRESHOLD,
) -> int:
    """
    Repair illegal actions:
      1) drop expedite under U3 / forbid_expedite
      2) bump ell→2 when rule_buffer_mask requires high buffer
    """
    kwargs = dict(
        forbid_expedite=forbid_expedite,
        space=space,
        rule_buffer_mask=rule_buffer_mask,
        fw_daily_demand_means=fw_daily_demand_means,
        dos_threshold=dos_threshold,
    )
    if is_action_legal(action_id, state, **kwargs):
        return action_id

    action = decode_action(action_id, space=space)
    ell = action.ell
    expedite = action.expedite
    allocation = action.allocation
    focus = action.focus

    # Expedite repair
    block_expedite = bool(forbid_expedite)
    if hasattr(state, "transport_state"):
        block_expedite = block_expedite or int(state.transport_state) == 3
    else:
        block_expedite = block_expedite or int(state) == 3
    if block_expedite and expedite == 1:
        expedite = 0

    # Buffer repair
    if rule_buffer_mask and hasattr(state, "central_inventory"):
        if requires_high_buffer(
            state, fw_daily_demand_means, dos_threshold=dos_threshold
        ):
            ell = 2

    repaired = encode_action(
        ell=ell,
        expedite=expedite,
        allocation=allocation,
        focus=focus,
        space=space,
    )
    if is_action_legal(repaired, state, **kwargs):
        return repaired
    # Last resort: first legal action
    legal = legal_action_ids(state, **kwargs)
    if not legal:
        raise RuntimeError("no legal actions after remapping")
    return int(legal[0])
