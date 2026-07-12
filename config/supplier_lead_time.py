from __future__ import annotations

import random
from enum import Enum


class SupplierLeadTimeMode(str, Enum):
    BASELINE = "baseline"
    VOLATILE = "volatile"
    DISRUPTION_MIXTURE = "disruption_mixture"


def sample_supplier_lead_time(
    mode: SupplierLeadTimeMode,
    random_generator: random.Random,
) -> int:
    """Sample supplier-to-CW lead time L^C_t (PDF Section 17.1)."""
    if mode == SupplierLeadTimeMode.BASELINE:
        return random_generator.randint(1, 3)

    if mode == SupplierLeadTimeMode.VOLATILE:
        return random_generator.randint(1, 5)

    if mode == SupplierLeadTimeMode.DISRUPTION_MIXTURE:
        if random_generator.random() < 0.9:
            return random_generator.randint(1, 3)
        return random_generator.randint(4, 6)

    raise ValueError(f"Unknown supplier lead time mode: {mode}")
