from __future__ import annotations

import random
from typing import Dict, List, Optional

from config.disruption_chain import (
    DisruptionState,
    SupplierAvailability,
    TransportState,
    default_disruption_state,
    sample_next_disruption_state,
    sample_next_supplier_availability,
    sample_next_transport_state,
)
from config.input_product import (
    get_default_forward_warehouse_demand_intensity_multipliers,
    get_default_products,
)
from config.simulation_config import SimulationConfig
from metrics.performance_tracker import PerformanceTracker
from mdp.actions import ACTION_SPACE_COARSE, MDPAction, decode_action
from mdp.thresholds import (
    resolve_thresholds_for_action,
    temporary_central_thresholds,
    temporary_forward_thresholds,
)
from models.central_warehouse import CentralWarehouse
from models.demand_generator import DemandGenerator
from models.forward_warehouse import ForwardWarehouse
from models.product import Product

ALLOCATION_RULE_BY_G = {
    0: "proportional",
    1: "lowest_dos",
    2: "value_penalty",
}

class SimulationEngine:
    """Orchestrates the 7-step daily event sequence (PDF Section 4)."""

    def __init__(
        self,
        config: SimulationConfig,
        products: Optional[List[Product]] = None,
        demand_intensity_multipliers: Optional[List[float]] = None,
        seed: Optional[int] = None,
        fixed_disruption_state: Optional[DisruptionState] = None,
        action_space: str = ACTION_SPACE_COARSE,
    ) -> None:
        self.config = config
        self.action_space = str(action_space).lower()
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

        # Phase 2 disruption state Z_t = (A_t, U_t)
        self.fixed_disruption_state = fixed_disruption_state
        if fixed_disruption_state is not None:
            self.disruption_state = fixed_disruption_state
        else:
            self.disruption_state = default_disruption_state()

        self.central_warehouse, self.forward_warehouses = self._build_network()
        self.demand_generator = DemandGenerator(actual_seed)
        self.performance_tracker = PerformanceTracker(config, self.products)

        # Post-demand observation (step 3 done, step 4 not yet started).
        # Updated every day by run_morning(); read by RetailMDPEnv.get_state().
        self.post_demand_obs: Optional[Dict] = None
        self._day_context: Optional[Dict] = None

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
            disruption_state=self.disruption_state,
        )

        return central_warehouse, forward_warehouses

    def _advance_disruption_state(self) -> None:
        """Sample Z_{t+1} according to disruption_mode, unless force-fixed."""
        if self.fixed_disruption_state is not None:
            self.disruption_state = self.fixed_disruption_state
            return

        mode = self.config.disruption_mode
        current = self.disruption_state
        rng = self.random_generator

        if mode == "none":
            self.disruption_state = default_disruption_state()
        elif mode == "both":
            self.disruption_state = sample_next_disruption_state(
                current,
                rng,
                m_a=self.config.m_a,
                m_u=self.config.m_u,
            )
        elif mode == "supply_only":
            self.disruption_state = DisruptionState(
                supplier_availability=sample_next_supplier_availability(
                    current.supplier_availability,
                    rng,
                    matrix=self.config.m_a,
                ),
                transport_state=TransportState.U0,
            )
        elif mode == "transport_only":
            self.disruption_state = DisruptionState(
                supplier_availability=SupplierAvailability.A0,
                transport_state=sample_next_transport_state(
                    current.transport_state,
                    rng,
                    matrix=self.config.m_u,
                ),
            )
        else:
            raise ValueError(f"unknown disruption_mode: {mode}")

    def _normalize_action(
        self,
        action: Optional[object],
    ) -> Optional[MDPAction]:
        if action is None:
            return None
        if isinstance(action, MDPAction):
            return action
        if isinstance(action, int) and not isinstance(action, bool):
            return decode_action(action, space=self.action_space)
        raise TypeError(
            "action must be None, MDPAction, or action_id int, "
            f"got {type(action).__name__}"
        )

    def _capture_post_demand_obs(
        self,
        day: int,
        cw: CentralWarehouse,
        fws: List[ForwardWarehouse],
        disruption_today: DisruptionState,
    ) -> None:
        """Store post-demand (step 3) observation for MDP env."""
        self.post_demand_obs = {
            "day": day,
            "is_warmup": day <= self.config.warmup_days,
            "central_inventory": cw.get_inventory_copy(),
            "forward_inventory": {
                fw.forward_warehouse_id: fw.get_inventory_copy() for fw in fws
            },
            "supplier_pipeline": [
                {
                    "quantities": list(order.quantities),
                    "remaining_lead_time": order.remaining_lead_time,
                }
                for order in cw.supplier_pipeline
            ],
            "supplier_backlog": list(cw.upstream.supplier_backlog),
            "transport_waiting": list(cw.upstream.transport_waiting),
            "supplier_availability": disruption_today.supplier_availability.value,
            "transport_state": disruption_today.transport_state.value,
            "disruption_state": disruption_today.code(),
            "demands": {
                fw.forward_warehouse_id: list(fw.today_demand) for fw in fws
            },
            "lost_sales": {
                fw.forward_warehouse_id: list(fw.today_lost_sales) for fw in fws
            },
        }

    def run_morning(self, day: int) -> dict:
        """
        Steps 1–3: supplier arrivals, FW shipments, demand/sales.

        Captures post_demand_obs (decision-time state) and stores day context
        for the subsequent run_evening() call.
        """
        cw = self.central_warehouse
        fws = self.forward_warehouses

        cw.set_disruption_state(self.disruption_state)
        disruption_today = self.disruption_state

        cw.advance_morning_pipeline()
        central_inventory_begin = cw.get_inventory_copy()

        for fw in fws:
            fw.receive_shipments(
                cw.pop_pending_shipments(fw.forward_warehouse_id)
            )
        forward_inventory_begin = {
            fw.forward_warehouse_id: fw.get_inventory_copy() for fw in fws
        }

        demands = self.demand_generator.generate_all(fws, self.products)
        for fw in fws:
            fw.execute_daily_sales(demands[fw.forward_warehouse_id])
        forward_inventory_end = {
            fw.forward_warehouse_id: fw.get_inventory_copy() for fw in fws
        }

        self._capture_post_demand_obs(day, cw, fws, disruption_today)

        self._day_context = {
            "day": day,
            "central_inventory_begin": central_inventory_begin,
            "forward_inventory_begin": forward_inventory_begin,
            "forward_inventory_end": forward_inventory_end,
            "demands": demands,
            "disruption_today": disruption_today,
        }
        return dict(self.post_demand_obs)

    def run_evening(self, action: Optional[object] = None) -> dict:
        """
        Steps 4–7: replenishment, allocation, supplier ordering, accounting.

        Must be preceded by run_morning() for the same day.
        """
        if not hasattr(self, "_day_context") or self._day_context is None:
            raise RuntimeError("run_morning() must be called before run_evening()")

        ctx = self._day_context
        day = ctx["day"]
        cw = self.central_warehouse
        fws = self.forward_warehouses
        mdp_action = self._normalize_action(action)

        central_inventory_begin = ctx["central_inventory_begin"]
        forward_inventory_begin = ctx["forward_inventory_begin"]
        forward_inventory_end = ctx["forward_inventory_end"]
        demands = ctx["demands"]
        disruption_today = ctx["disruption_today"]

        use_phase2_path = mdp_action is None or mdp_action.is_baseline

        if use_phase2_path:
            requests = {
                fw.forward_warehouse_id: fw.check_replenishment_request()
                for fw in fws
            }
            shipments, shortage_flags = cw.allocate_to_fws(
                requests, fws, forward_inventory_end
            )
            central_inventory_end = cw.get_inventory_copy()
            order_quantities = cw.check_supplier_ordering(self.random_generator)
            expedite_cost = 0.0
        else:
            # Decision-time inventories (post-demand, pre-allocation).
            cw_inv_decision = list(cw.get_inventory_copy())
            fw_inv_list = [
                list(forward_inventory_end[fw.forward_warehouse_id])
                for fw in fws
            ]
            cw_s, cw_S, fw_thresholds = resolve_thresholds_for_action(
                mdp_action,
                cw,
                fws,
                central_inventory=cw_inv_decision,
                forward_inventory=fw_inv_list,
            )

            requests = {}
            for fw, (s_tilde, S_tilde) in zip(fws, fw_thresholds):
                requests[fw.forward_warehouse_id] = fw.check_replenishment_request(
                    reorder_points=s_tilde,
                    order_up_to_levels=S_tilde,
                )

            allocation_rule = ALLOCATION_RULE_BY_G[mdp_action.allocation]
            shipments, shortage_flags = cw.allocate_to_fws(
                requests,
                fws,
                forward_inventory_end,
                allocation_rule=allocation_rule,
            )
            central_inventory_end = cw.get_inventory_copy()

            order_quantities = cw.check_supplier_ordering(
                self.random_generator,
                reorder_points=cw_s,
                order_up_to_levels=cw_S,
                expedite=mdp_action.expedite,
            )
            expedite_cost = 0.0
            if mdp_action.expedite == 1:
                for product_index, quantity in enumerate(order_quantities):
                    expedite_cost += (
                        0.05
                        * self.products[product_index].selling_price
                        * quantity
                    )

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
            "disruption_state": disruption_today.code(),
            "supplier_availability": (
                disruption_today.supplier_availability.value
            ),
            "transport_state": disruption_today.transport_state.value,
            "supplier_backlog": list(cw.upstream.supplier_backlog),
            "transport_waiting": list(cw.upstream.transport_waiting),
            "action_id": (
                None
                if mdp_action is None
                else mdp_action.action_id_for(self.action_space)
            ),
            "action": (
                None if mdp_action is None else mdp_action.as_full_tuple()
            ),
            "supplier_order_quantities": list(order_quantities),
            "expedite_cost": expedite_cost,
        }
        self.performance_tracker.log_daily_snapshot(snapshot, fws)

        self._advance_disruption_state()
        self._day_context = None

        return snapshot

    def run_one_day(
        self,
        day: int,
        action: Optional[object] = None,
    ) -> dict:
        """Run one full day (morning + evening). Backward-compatible wrapper."""
        self.run_morning(day)
        return self.run_evening(action)

    def run(self, action: Optional[object] = None) -> dict:
        """Run warmup + evaluation days; return one replication's KPIs.

        If action is provided, the same action is applied every day.
        """
        for day in range(1, self.config.total_simulation_days + 1):
            self.run_one_day(day, action=action)

        return self.performance_tracker.summarize()