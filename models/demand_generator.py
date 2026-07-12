from __future__ import annotations

from typing import Dict, List

import numpy as np

from config.policy import compute_daily_demand_mean
from models.forward_warehouse import ForwardWarehouse
from models.product import Product


class DemandGenerator:
    """Generate independent Poisson customer demand (PDF Section 5)."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.random_generator = np.random.default_rng(seed)

    def generate_demand_for_product(
        self,
        forward_warehouse: ForwardWarehouse,
        product: Product,
    ) -> int:
        """Generate demand for one FW and one product."""
        daily_demand_mean = compute_daily_demand_mean(
            forward_warehouse.demand_intensity_multiplier,
            product,
        )
        return int(self.random_generator.poisson(daily_demand_mean))

    def generate_for_forward_warehouse(
        self,
        forward_warehouse: ForwardWarehouse,
        products: List[Product],
    ) -> List[int]:
        """Generate demand vector for one FW across all products."""
        return [
            self.generate_demand_for_product(forward_warehouse, product)
            for product in products
        ]

    def generate_all(
        self,
        forward_warehouses: List[ForwardWarehouse],
        products: List[Product],
    ) -> Dict[int, List[int]]:
        """
        Generate demand for all FWs and all products.

        Returns:
            {forward_warehouse_id: [demand_p1, ..., demand_p5]}
        """
        demands: Dict[int, List[int]] = {}

        for forward_warehouse in forward_warehouses:
            demands[forward_warehouse.forward_warehouse_id] = (
                self.generate_for_forward_warehouse(forward_warehouse, products)
            )

        return demands