from __future__ import annotations

import random
from typing import List, Optional, Sequence

from config.disruption_chain import (
    VALID_MATRIX_PROFILES,
    get_transition_matrices,
    validate_transition_matrices,
)
from config.supplier_lead_time import SupplierLeadTimeMode, sample_supplier_lead_time

VALID_DISRUPTION_MODES = frozenset(
    {"none", "both", "supply_only", "transport_only"}
)


class SimulationConfig:
    """Global simulation settings (PDF Section 15 + default capacities)."""

    def __init__(
        self,
        warmup_days: int = 30,              # T_warm
        evaluation_days: int = 365,         # T
        num_replications: int = 30,       # R
        central_warehouse_capacity: int = 3600,   # CC
        forward_warehouse_capacity: int = 180,      # CF
        days_of_supply_epsilon: float = 1e-6,       # epsilon
        random_seed_base: int = 42,
        num_products: int = 5,
        num_forward_warehouses: int = 10,
        supplier_lead_time_mode: SupplierLeadTimeMode = SupplierLeadTimeMode.BASELINE,
        demand_intensity_scale: float = 1.0,
        forward_holding_cost_ratio: float = 3.0,
        enable_disruption: bool = False,
        disruption_mode: Optional[str] = None,
        disruption_matrix_profile: str = "pdf",
        custom_m_a: Optional[Sequence[Sequence[float]]] = None,
        custom_m_u: Optional[Sequence[Sequence[float]]] = None,
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
        # Phase 2: disruption_mode is the primary switch.
        # enable_disruption remains for compatibility (False→none, True→both).
        if disruption_mode is None:
            disruption_mode = "both" if enable_disruption else "none"
        if disruption_mode not in VALID_DISRUPTION_MODES:
            raise ValueError(f"unknown disruption_mode: {disruption_mode}")
        self.disruption_mode = disruption_mode
        self.enable_disruption = disruption_mode != "none"

        # Transition-matrix selection (PDF default or stress profiles).
        # If custom_m_a / custom_m_u are both provided, they override the profile.
        if (custom_m_a is None) ^ (custom_m_u is None):
            raise ValueError(
                "custom_m_a and custom_m_u must both be provided or both omitted"
            )
        if custom_m_a is not None and custom_m_u is not None:
            validate_transition_matrices(custom_m_a, custom_m_u)
            self.disruption_matrix_profile = "custom"
            self.m_a: List[List[float]] = [list(row) for row in custom_m_a]
            self.m_u: List[List[float]] = [list(row) for row in custom_m_u]
        else:
            if disruption_matrix_profile not in VALID_MATRIX_PROFILES:
                raise ValueError(
                    f"unknown disruption_matrix_profile: "
                    f"{disruption_matrix_profile!r}; "
                    f"expected one of {sorted(VALID_MATRIX_PROFILES)}"
                )
            self.disruption_matrix_profile = disruption_matrix_profile
            self.m_a, self.m_u = get_transition_matrices(disruption_matrix_profile)

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
            f"forward_holding_cost_ratio={self.forward_holding_cost_ratio}, "
            f"disruption_mode={self.disruption_mode!r}, "
            f"disruption_matrix_profile={self.disruption_matrix_profile!r}, "
            f"enable_disruption={self.enable_disruption})"
        )
