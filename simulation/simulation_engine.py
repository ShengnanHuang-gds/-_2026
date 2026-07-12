from __future__ import annotations

import random
from typing import Dict, List, Optional

from config.input_product import (
    get_default_forward_warehouse_demand_intensity_multipliers,
    get_default_products,
)
from config.simulation_config import SimulationConfig
from metrics.performance_tracker import PerformanceTracker
from models.central_warehouse import CentralWarehouse
from models.demand_generator import DemandGenerator
from models.forward_warehouse import ForwardWarehouse
from models.product import Product


class SimulationEngine:
    """Orchestrates the 7-step daily event sequence (PDF Section 4)."""

    def __init__(
        self,
        config: SimulationConfig,
        products: Optional[List[Product]] = None,
        demand_intensity_multipliers: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.products = products or get_default_products(
            config.forward_holding_cost_ratio
        )
        self.demand_intensity_multipliers = (
            demand_intensity_multipliers
            or [
                multiplier * config.demand_intensity_scale
                for multiplier in get_default_forward_warehouse_demand_intensity_multipliers()
            ]
        )
        actual_seed = seed if seed is not None else config.random_seed_base
        self.random_generator = random.Random(actual_seed)

        self.central_warehouse, self.forward_warehouses = self._build_network()
        self.demand_generator = DemandGenerator(actual_seed)
        self.performance_tracker = PerformanceTracker(config, self.products)

    def _build_network(self):
        forward_warehouse_ids = list(
            range(1, self.config.num_forward_warehouses + 1)
        )

        forward_warehouses = [
            ForwardWarehouse(
                forward_warehouse_id=forward_warehouse_id,
                demand_intensity_multiplier=self.demand_intensity_multipliers[
                    forward_warehouse_id - 1
                ],
                products=self.products,
                forward_warehouse_capacity=self.config.forward_warehouse_capacity,
            )
            for forward_warehouse_id in forward_warehouse_ids
        ]

        central_warehouse = CentralWarehouse(
            products=self.products,
            central_warehouse_capacity=self.config.central_warehouse_capacity,
            days_of_supply_epsilon=self.config.days_of_supply_epsilon,
            forward_warehouse_ids=forward_warehouse_ids,
            config=self.config,
        )

        return central_warehouse, forward_warehouses

    def run_one_day(self, day: int) -> dict:
        """Run one day and return a snapshot dict."""
        cw = self.central_warehouse
        fws = self.forward_warehouses

        # Step 1: supplier arrivals
        cw.advance_morning_pipeline()
        central_inventory_begin = cw.get_inventory_copy()

        # Step 2: overnight CW -> FW arrivals
        for fw in fws:
            fw.receive_shipments(
                cw.pop_pending_shipments(fw.forward_warehouse_id)
            )
        forward_inventory_begin = {
            fw.forward_warehouse_id: fw.get_inventory_copy() for fw in fws
        }

        # Step 3: demand and sales
        demands = self.demand_generator.generate_all(fws, self.products)
        for fw in fws:
            fw.execute_daily_sales(demands[fw.forward_warehouse_id])
        forward_inventory_end = {
            fw.forward_warehouse_id: fw.get_inventory_copy() for fw in fws
        }

        # Step 4: FW replenishment requests
        requests = {
            fw.forward_warehouse_id: fw.check_replenishment_request()
            for fw in fws
        }

        # Step 5: CW allocation
        shipments, shortage_flags = cw.allocate_to_fws(
            requests, fws, forward_inventory_end
        )
        central_inventory_end = cw.get_inventory_copy()

        # Step 6: CW supplier ordering
        cw.check_supplier_ordering(self.random_generator)

        # Step 7: accounting
        snapshot = {
            "day": day,
            "is_warmup": day <= self.config.warmup_days,
            "central_inventory_begin": central_inventory_begin,
            "central_inventory_end": central_inventory_end,
            "forward_inventory_begin": forward_inventory_begin,
            "forward_inventory_end": forward_inventory_end,
            "demands": demands,
            "requests": requests,
            "shipments": shipments,
            "shortage_flags": shortage_flags,
        }
        self.performance_tracker.log_daily_snapshot(snapshot, fws)
        return snapshot

    def run(self) -> dict:
        """Run warmup + evaluation days; return one replication's KPIs."""
        for day in range(1, self.config.total_simulation_days + 1):
            self.run_one_day(day)

        return self.performance_tracker.summarize()
