from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from mdp.actions import BUFFER_MULTIPLIERS, MDPAction, resolve_product_buffer_levels
from models.central_warehouse import CentralWarehouse
from models.forward_warehouse import ForwardWarehouse


def _eta_triple(ell: int) -> Tuple[float, float, float]:
    return BUFFER_MULTIPLIERS[int(ell)]


def temporary_forward_thresholds(
    forward_warehouse: ForwardWarehouse,
    action: MDPAction,
    *,
    ell_per_product: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[int]]:
    """
    PDF (45): temporary FW thresholds from ell_t (or per-product ell).

    s̃^F = min{ceil(η^F * s^F), S^F - 1},  S̃^F = S^F.
    """
    reorder_points: List[int] = []
    order_up_to_levels: List[int] = []

    for k, (reorder_point, order_up_to) in enumerate(
        zip(
            forward_warehouse.reorder_points,
            forward_warehouse.order_up_to_levels,
        )
    ):
        if ell_per_product is None:
            eta = action.eta_forward_reorder
        else:
            eta = _eta_triple(ell_per_product[k])[2]
        raised = math.ceil(eta * reorder_point)
        reorder_points.append(min(raised, order_up_to - 1))
        order_up_to_levels.append(order_up_to)

    return reorder_points, order_up_to_levels


def temporary_central_thresholds(
    central_warehouse: CentralWarehouse,
    action: MDPAction,
    *,
    ell_per_product: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[int]]:
    """
    PDF (44): temporary CW thresholds from ell_t (or per-product ell).

    s̃^C = ceil(η^{C,s} * s^C),  S̃^C = ceil(η^{C,S} * S^C).
    """
    reorder_points: List[int] = []
    order_up_to_levels: List[int] = []

    for k, (reorder_point, order_up_to) in enumerate(
        zip(
            central_warehouse.reorder_points,
            central_warehouse.order_up_to_levels,
        )
    ):
        if ell_per_product is None:
            eta_s = action.eta_central_reorder
            eta_S = action.eta_central_order_up_to
        else:
            eta_s, eta_S, _ = _eta_triple(ell_per_product[k])
        reorder_points.append(math.ceil(eta_s * reorder_point))
        order_up_to_levels.append(math.ceil(eta_S * order_up_to))

    return reorder_points, order_up_to_levels


def resolve_thresholds_for_action(
    action: MDPAction,
    central_warehouse: CentralWarehouse,
    forward_warehouses: Sequence[ForwardWarehouse],
    *,
    central_inventory: Sequence[int],
    forward_inventory: Sequence[Sequence[int]],
) -> Tuple[List[int], List[int], List[Tuple[List[int], List[int]]]]:
    """
    Compute CW (s,S) and per-FW (s,S) under focus-aware buffer levels.

    Returns:
      cw_s, cw_S, list of (fw_s, fw_S) aligned with forward_warehouses order.
    """
    fw_mu = [list(fw.get_daily_demand_means()) for fw in forward_warehouses]
    cw_ell, fw_ell = resolve_product_buffer_levels(
        action,
        central_inventory=central_inventory,
        forward_inventory=forward_inventory,
        fw_daily_demand_means=fw_mu,
    )
    cw_s, cw_S = temporary_central_thresholds(
        central_warehouse, action, ell_per_product=cw_ell
    )
    fw_thresholds: List[Tuple[List[int], List[int]]] = []
    for i, fw in enumerate(forward_warehouses):
        fw_thresholds.append(
            temporary_forward_thresholds(
                fw, action, ell_per_product=fw_ell[i]
            )
        )
    return cw_s, cw_S, fw_thresholds
