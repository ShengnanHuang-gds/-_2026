from __future__ import annotations

from typing import List


class SupplierOrder:
    """represent a order from cw to external warehouse section8.1"""

    def __init__(
        self,
        quantities: List[int],
        remaining_lead_time: int,
    ) -> None:
        if len(quantities) != 5:
            raise ValueError("quantities must contain exactly 5 product values")

        if remaining_lead_time < 1:
            raise ValueError("remaining_lead_time must be >= 1 when order is created")

        if any(quantity < 0 for quantity in quantities):
            raise ValueError("quantities must be non-negative")

        self.quantities = list(quantities)  
        self.remaining_lead_time = remaining_lead_time 

    def __repr__(self) -> str:
        return (
            f"SupplierOrder(quantities={self.quantities}, "
            f"remaining_lead_time={self.remaining_lead_time})"
        )