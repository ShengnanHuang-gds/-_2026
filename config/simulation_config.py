from __future__ import annotations

import random

from config.supplier_lead_time import SupplierLeadTimeMode, sample_supplier_lead_time


class SimulationConfig:
    """Global simulation settings (PDF Section 15 + default capacities)."""

    def __init__(
        self,
        warmup_days: int = 30,              # T_warm
        evaluation_days: int = 365,         # T
        num_replications: int = 30,       # R
        central_warehouse_capacity: int = 3600,   # CC
        forward_warehouse_capacity: int = 160,      # CF
        days_of_supply_epsilon: float = 1e-6,       # epsilon
        random_seed_base: int = 42,
        num_products: int = 5,
        num_forward_warehouses: int = 10,
        supplier_lead_time_mode: SupplierLeadTimeMode = SupplierLeadTimeMode.BASELINE,
        demand_intensity_scale: float = 1.2,
        forward_holding_cost_ratio: float = 3.0,
    ) -> None:
        self.warmup_days = warmup_days
        self.evaluation_days = evaluation_days
        self.num_replications = num_replications
        self.central_warehouse_capacity = central_warehouse_capacity
        self.forward_warehouse_capacity = forward_warehouse_capacity
        self.days_of_supply_epsilon = days_of_supply_epsilon
        self.random_seed_base = random_seed_base
        self.num_products = num_products
        self.num_forward_warehouses = num_forward_warehouses
        self.supplier_lead_time_mode = supplier_lead_time_mode
        self.demand_intensity_scale = demand_intensity_scale
        self.forward_holding_cost_ratio = forward_holding_cost_ratio

    @property
    def total_simulation_days(self) -> int:
        """Warm-up + evaluation horizon (T_warm + T)."""
        return self.warmup_days + self.evaluation_days

    def sample_supplier_lead_time(self, random_generator: random.Random) -> int:
        """Sample one supplier order lead time according to the configured mode."""
        return sample_supplier_lead_time(
            self.supplier_lead_time_mode,
            random_generator,
        )

    def __repr__(self) -> str:
        return (
            f"SimulationConfig("
            f"warmup_days={self.warmup_days}, "
            f"evaluation_days={self.evaluation_days}, "
            f"num_replications={self.num_replications}, "
            f"central_warehouse_capacity={self.central_warehouse_capacity}, "
            f"forward_warehouse_capacity={self.forward_warehouse_capacity}, "
            f"supplier_lead_time_mode={self.supplier_lead_time_mode.value}, "
            f"demand_intensity_scale={self.demand_intensity_scale}, "
            f"forward_holding_cost_ratio={self.forward_holding_cost_ratio})"
        )
