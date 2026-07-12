from __future__ import annotations

from typing import List

from config.policy import (
    compute_daily_demand_mean,
    compute_forward_warehouse_policy,
)
from models.product import Product


class ForwardWarehouse:
    """"""

    def __init__(
        self,
        forward_warehouse_id: int,
        demand_intensity_multiplier: float,
        products: List[Product],
        forward_warehouse_capacity: int,
    ) -> None:
        self.forward_warehouse_id = forward_warehouse_id
        self.demand_intensity_multiplier = demand_intensity_multiplier
        self.products = products
        self.forward_warehouse_capacity = forward_warehouse_capacity

        reorder_points, order_up_to_levels = compute_forward_warehouse_policy(
            demand_intensity_multiplier, products
        )#fw (s,S) policy
        self.reorder_points = reorder_points          
        self.order_up_to_levels = order_up_to_levels

        # Section 15 initial condition: the fw contain max i
        self.inventory = list(order_up_to_levels)

        # today's flow variables (Step 3-4), reset each day by engine or methods
        self.today_demand: List[int] = [0, 0, 0, 0, 0]
        self.today_sales: List[int] = [0, 0, 0, 0, 0]
        self.today_lost_sales: List[int] = [0, 0, 0, 0, 0]
        self.today_replenishment_requests: List[int] = [0, 0, 0, 0, 0]

    def receive_shipments(self, shipment_quantities: List[int]) -> None:
        """Step 2: overnight arrivals from central warehouse."""
        if len(shipment_quantities) != 5:
            raise ValueError("shipment_quantities must contain exactly 5 values")

        for product_index in range(5):
            self.inventory[product_index] += shipment_quantities[product_index]

    def execute_daily_sales(self, demand: List[int]) -> None:
        """Step 3: section 4 step 3"""
        if len(demand) != 5:
            raise ValueError("demand must contain exactly 5 values")

        self.today_demand = list(demand)
        self.today_sales = [0, 0, 0, 0, 0]
        self.today_lost_sales = [0, 0, 0, 0, 0]

        for product_index in range(5):
            current_inventory = self.inventory[product_index]
            customer_demand = demand[product_index]#the customer demand for a product

            sales = min(current_inventory, customer_demand)# section7 
           #sales quantity s equal to the minimum of the available inventory and customer demand 
            lost_sales = customer_demand - sales

            self.today_sales[product_index] = sales
            self.today_lost_sales[product_index] = lost_sales
            self.inventory[product_index] = current_inventory - sales

    def check_replenishment_request(self) -> List[int]:
        """section7 如果低于s补到S else 不变 step4 in section 4"""
        self.today_replenishment_requests = [0, 0, 0, 0, 0]

        for product_index in range(5):
            current_inventory = self.inventory[product_index]
            reorder_point = self.reorder_points[product_index]
            order_up_to_level = self.order_up_to_levels[product_index]

            if current_inventory <= reorder_point:
                self.today_replenishment_requests[product_index] = (
                    order_up_to_level - current_inventory
                )

        return list(self.today_replenishment_requests)

    def get_inventory_copy(self) -> List[int]:
        """Return a copy of current on-hand inventory."""
        return list(self.inventory)

    def get_daily_demand_means(self) -> List[float]:
        """Return lambda_i,k for each product (used later by CW allocation DOS)."""
        return [
            compute_daily_demand_mean(self.demand_intensity_multiplier, product)
            for product in self.products
        ]

    def __repr__(self) -> str:
        return (
            f"ForwardWarehouse(forward_warehouse_id={self.forward_warehouse_id}, "
            f"inventory={self.inventory})"
        )