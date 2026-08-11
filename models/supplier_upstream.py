from __future__ import annotations

from typing import List, Optional

from config.disruption_chain import (
    DisruptionState,
    compute_realized_lead_time,
    released_quantity,
)
from models.supplier_order import SupplierOrder

NUM_PRODUCTS = 5


def _zero_vector() -> List[int]:
    return [0] * NUM_PRODUCTS


def _validate_quantity_vector(quantities: List[int], name: str) -> None:
    if len(quantities) != NUM_PRODUCTS:
        raise ValueError(f"{name} must contain exactly {NUM_PRODUCTS} values")
    if any(quantity < 0 for quantity in quantities):
        raise ValueError(f"{name} must be non-negative")


class SupplierUpstreamBuffers:
    """
    Phase 2 upstream chain before goods enter the CW pipeline:

    supplier_backlog -> transport_waiting -> (SupplierOrder pipeline)
    """

    def __init__(self) -> None:
        self.supplier_backlog: List[int] = _zero_vector()
        self.transport_waiting: List[int] = _zero_vector()

    def add_order(self, quantities: List[int]) -> None:
        """Place new CW order quantities into supplier backlog."""
        _validate_quantity_vector(quantities, "order quantities")
        for product_index in range(NUM_PRODUCTS):
            self.supplier_backlog[product_index] += quantities[product_index]

    def open_quantities(self) -> List[int]:
        """Backlog + waiting still owed to the CW (not yet in pipeline)."""
        return [
            self.supplier_backlog[product_index]
            + self.transport_waiting[product_index]
            for product_index in range(NUM_PRODUCTS)
        ]

    def release_from_backlog(self, disruption_state: DisruptionState) -> List[int]:
        """
        Release floor(backlog * alpha(A)) from backlog into transport_waiting.

        Returns the released vector.
        """
        released = _zero_vector()
        availability = disruption_state.supplier_availability

        for product_index in range(NUM_PRODUCTS):
            backlog = self.supplier_backlog[product_index]
            released_units = released_quantity(backlog, availability)
            released[product_index] = released_units
            self.supplier_backlog[product_index] = backlog - released_units
            self.transport_waiting[product_index] += released_units

        return released

    def try_dispatch_waiting(
        self,
        disruption_state: DisruptionState,
        base_lead_time: int,
        expedite: int = 0,
    ) -> Optional[SupplierOrder]:
        """
        If transport is not blocked, move all transport_waiting into one pipeline order.

        Lead time is max(1, L0 + delta(U) - e). Under U3, waiting stays and this returns None.
        """
        if disruption_state.is_transport_blocked:
            return None

        if sum(self.transport_waiting) == 0:
            return None

        lead_time = compute_realized_lead_time(
            base_lead_time,
            disruption_state.transport_state,
            expedite=expedite,
        )
        order = SupplierOrder(
            quantities=list(self.transport_waiting),
            remaining_lead_time=lead_time,
        )
        self.transport_waiting = _zero_vector()
        return order

    def process_evening(
        self,
        new_order_quantities: List[int],
        disruption_state: DisruptionState,
        base_lead_time: int,
        expedite: int = 0,
    ) -> Optional[SupplierOrder]:
        """
        Evening sequence:
        1) add Q to backlog
        2) release by A_t into transport_waiting
        3) dispatch waiting by U_t into a pipeline order (if allowed)
        """
        self.add_order(new_order_quantities)
        self.release_from_backlog(disruption_state)
        return self.try_dispatch_waiting(
            disruption_state,
            base_lead_time,
            expedite=expedite,
        )
