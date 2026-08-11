from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

from config.disruption_chain import (
    DisruptionState,
    default_disruption_state,
)
from config.policy import compute_central_warehouse_policy
from config.simulation_config import SimulationConfig
from models.forward_warehouse import ForwardWarehouse
from models.product import Product
from models.supplier_order import SupplierOrder
from models.supplier_upstream import SupplierUpstreamBuffers


class CentralWarehouse:
    """Central warehouse (PDF Section 8 + Section 9 + Algorithm 1/2)."""

    def __init__(
        self,
        products: List[Product],
        central_warehouse_capacity: int,
        days_of_supply_epsilon: float,
        forward_warehouse_ids: List[int],
        config: SimulationConfig,
        disruption_state: Optional[DisruptionState] = None,
    ) -> None:
        self.config = config
        self.products = products
        self.central_warehouse_capacity = central_warehouse_capacity
        self.days_of_supply_epsilon = days_of_supply_epsilon
        self.disruption_state = (
            disruption_state
            if disruption_state is not None
            else default_disruption_state()
        )

        reorder_points, order_up_to_levels = compute_central_warehouse_policy(products)
        self.reorder_points = reorder_points
        self.order_up_to_levels = order_up_to_levels

        # Section 15: initial state 等于 upper bound
        self.inventory = list(order_up_to_levels)

        self.supplier_pipeline: List[SupplierOrder] = []
        self.upstream = SupplierUpstreamBuffers()

        # shipments dispatched at end of day t, arrive next morning
        self.pending_forward_warehouse_shipments: Dict[int, List[int]] = {
            forward_warehouse_id: [0, 0, 0, 0, 0]
            for forward_warehouse_id in forward_warehouse_ids
        }

    def set_disruption_state(self, disruption_state: DisruptionState) -> None:
        """Update today's Z_t = (A_t, U_t)."""
        self.disruption_state = disruption_state

    def get_inventory_position(self, product_index: int) -> int:
        """
        CW inventory position for product k.

        Phase 2: on-hand + pipeline + supplier backlog + transport waiting.
        """
        inventory_position = self.inventory[product_index]

        for order in self.supplier_pipeline:
            inventory_position += order.quantities[product_index]

        inventory_position += self.upstream.supplier_backlog[product_index]
        inventory_position += self.upstream.transport_waiting[product_index]
        return inventory_position


    def advance_morning_pipeline(self) -> None:
        """Algorithm 1 / Step 1: supplier arrivals."""
        remaining_orders: List[SupplierOrder] = []

        for order in self.supplier_pipeline:
            order.remaining_lead_time -= 1#day --

            if order.remaining_lead_time == 0:#if reach zero add to inventory
                for product_index in range(5):
                    self.inventory[product_index] += order.quantities[product_index]
            else:
                remaining_orders.append(order)

        self.supplier_pipeline = remaining_orders

    def pop_pending_shipments(self, forward_warehouse_id: int) -> List[int]:
        """Step 2: give overnight shipments to one FW and clear buffer."""
        shipment_quantities = list(
            self.pending_forward_warehouse_shipments[forward_warehouse_id]
        )
        self.pending_forward_warehouse_shipments[forward_warehouse_id] = [
            0, 0, 0, 0, 0
        ]#clear pending area
        return shipment_quantities

    def allocate_to_fws(
        self,
        requests: Dict[int, List[int]],
        forward_warehouses: List[ForwardWarehouse],
        post_demand_inventories: Dict[int, List[int]],
        allocation_rule: str = "phase2_default",
    ) -> tuple[Dict[int, List[int]], List[bool]]:
        """
        Algorithm 2 / Step 5.

        allocation_rule:
          - phase2_default: proportional floor + lowest-DOS remainder (Phase 1/2)
          - proportional: PDF g=0
          - lowest_dos: PDF g=1
          - value_penalty: PDF g=2
        """
        shipments = {
            forward_warehouse.forward_warehouse_id: [0, 0, 0, 0, 0]
            for forward_warehouse in forward_warehouses
        }
        shortage_flags = [False, False, False, False, False]

        for product_index in range(5):
            product_shipments, product_shortage = self._allocate_one_product(
                product_index=product_index,
                requests=requests,
                forward_warehouses=forward_warehouses,
                post_demand_inventories=post_demand_inventories,
                allocation_rule=allocation_rule,
            )

            for forward_warehouse in forward_warehouses:
                fw_id = forward_warehouse.forward_warehouse_id
                shipped_quantity = product_shipments[fw_id]
                shipments[fw_id][product_index] = shipped_quantity
                self.pending_forward_warehouse_shipments[fw_id][product_index] += (
                    shipped_quantity
                )
                self.inventory[product_index] -= shipped_quantity

            shortage_flags[product_index] = product_shortage

        return shipments, shortage_flags

    def _allocate_one_product(
        self,
        product_index: int,
        requests: Dict[int, List[int]],
        forward_warehouses: List[ForwardWarehouse],
        post_demand_inventories: Dict[int, List[int]],
        allocation_rule: str = "phase2_default",
    ) -> tuple[Dict[int, int], bool]:
        """Allocate one product under the selected shortage rule."""
        central_inventory = self.inventory[product_index]

        forward_requests = {
            forward_warehouse.forward_warehouse_id: requests[
                forward_warehouse.forward_warehouse_id
            ][product_index]
            for forward_warehouse in forward_warehouses
        }

        total_request = sum(forward_requests.values())
        shipments = {
            forward_warehouse.forward_warehouse_id: 0
            for forward_warehouse in forward_warehouses
        }

        if total_request == 0:
            return shipments, False

        if total_request <= central_inventory:
            for fw_id, request_quantity in forward_requests.items():
                shipments[fw_id] = request_quantity
            return shipments, False

        if allocation_rule == "phase2_default":
            return self._allocate_phase2_default(
                product_index,
                forward_warehouses,
                forward_requests,
                post_demand_inventories,
                central_inventory,
            )
        if allocation_rule == "proportional":
            return self._allocate_proportional(
                forward_warehouses,
                forward_requests,
                central_inventory,
                total_request,
            )
        if allocation_rule == "lowest_dos":
            return self._allocate_by_priority(
                product_index,
                forward_warehouses,
                forward_requests,
                post_demand_inventories,
                central_inventory,
                mode="lowest_dos",
            )
        if allocation_rule == "value_penalty":
            return self._allocate_by_priority(
                product_index,
                forward_warehouses,
                forward_requests,
                post_demand_inventories,
                central_inventory,
                mode="value_penalty",
            )
        raise ValueError(f"unknown allocation_rule: {allocation_rule}")

    def _allocate_phase2_default(
        self,
        product_index: int,
        forward_warehouses: List[ForwardWarehouse],
        forward_requests: Dict[int, int],
        post_demand_inventories: Dict[int, List[int]],
        central_inventory: int,
    ) -> tuple[Dict[int, int], bool]:
        """Proportional floor + lowest-DOS remainder (existing Phase 1/2 rule)."""
        total_request = sum(forward_requests.values())
        initial_shipments = {}

        for forward_warehouse in forward_warehouses:
            fw_id = forward_warehouse.forward_warehouse_id
            request_quantity = forward_requests[fw_id]
            initial_shipments[fw_id] = math.floor(
                central_inventory * request_quantity / total_request
            )

        remainder = central_inventory - sum(initial_shipments.values())

        while remainder > 0:
            best_fw_id = None
            best_days_of_supply = None

            for forward_warehouse in forward_warehouses:
                fw_id = forward_warehouse.forward_warehouse_id
                request_quantity = forward_requests[fw_id]
                current_shipment = initial_shipments[fw_id]

                if current_shipment >= request_quantity:
                    continue

                daily_demand_mean = forward_warehouse.get_daily_demand_means()[
                    product_index
                ]
                post_demand_inventory = post_demand_inventories[fw_id][product_index]
                days_of_supply = (
                    post_demand_inventory + current_shipment
                ) / (daily_demand_mean + self.days_of_supply_epsilon)

                if best_fw_id is None or days_of_supply < best_days_of_supply:
                    best_fw_id = fw_id
                    best_days_of_supply = days_of_supply
                elif (
                    days_of_supply == best_days_of_supply
                    and fw_id < best_fw_id
                ):
                    best_fw_id = fw_id
                    best_days_of_supply = days_of_supply

            if best_fw_id is None:
                break

            initial_shipments[best_fw_id] += 1
            remainder -= 1

        return initial_shipments, True

    def _allocate_proportional(
        self,
        forward_warehouses: List[ForwardWarehouse],
        forward_requests: Dict[int, int],
        central_inventory: int,
        total_request: int,
    ) -> tuple[Dict[int, int], bool]:
        """PDF g=0: proportional floor; remainder by ascending FW id."""
        shipments = {
            fw.forward_warehouse_id: math.floor(
                central_inventory
                * forward_requests[fw.forward_warehouse_id]
                / total_request
            )
            for fw in forward_warehouses
        }
        remainder = central_inventory - sum(shipments.values())
        for fw in sorted(forward_warehouses, key=lambda item: item.forward_warehouse_id):
            if remainder <= 0:
                break
            fw_id = fw.forward_warehouse_id
            room = forward_requests[fw_id] - shipments[fw_id]
            if room <= 0:
                continue
            take = min(room, remainder)
            shipments[fw_id] += take
            remainder -= take
        return shipments, True

    def _allocate_by_priority(
        self,
        product_index: int,
        forward_warehouses: List[ForwardWarehouse],
        forward_requests: Dict[int, int],
        post_demand_inventories: Dict[int, List[int]],
        central_inventory: int,
        mode: str,
    ) -> tuple[Dict[int, int], bool]:
        """Unit-by-unit allocation by lowest DOS or highest value/penalty priority."""
        shipments = {
            fw.forward_warehouse_id: 0 for fw in forward_warehouses
        }
        remaining = central_inventory

        while remaining > 0:
            best_fw_id = None
            best_score = None

            for forward_warehouse in forward_warehouses:
                fw_id = forward_warehouse.forward_warehouse_id
                if shipments[fw_id] >= forward_requests[fw_id]:
                    continue

                daily_demand_mean = forward_warehouse.get_daily_demand_means()[
                    product_index
                ]
                post_demand_inventory = post_demand_inventories[fw_id][product_index]

                if mode == "lowest_dos":
                    score = (post_demand_inventory + shipments[fw_id]) / (
                        daily_demand_mean + self.days_of_supply_epsilon
                    )
                    better = (
                        best_fw_id is None
                        or score < best_score
                        or (score == best_score and fw_id < best_fw_id)
                    )
                else:
                    penalty = self.products[product_index].lost_sales_penalty
                    score = penalty * max(
                        daily_demand_mean - (post_demand_inventory + shipments[fw_id]),
                        0.0,
                    )
                    better = (
                        best_fw_id is None
                        or score > best_score
                        or (score == best_score and fw_id < best_fw_id)
                    )

                if better:
                    best_fw_id = fw_id
                    best_score = score

            if best_fw_id is None:
                break

            shipments[best_fw_id] += 1
            remaining -= 1

        return shipments, True

    def check_supplier_ordering(
        self,
        random_generator: random.Random,
        reorder_points: Optional[List[int]] = None,
        order_up_to_levels: Optional[List[int]] = None,
        expedite: int = 0,
    ) -> List[int]:
        """
        Step 6: (s, S) ordering with Phase 2 upstream release/dispatch.

        Optional temporary (s,S) and expedite flag for MDP actions.
        """
        active_reorder_points = (
            reorder_points if reorder_points is not None else self.reorder_points
        )
        active_order_up_to_levels = (
            order_up_to_levels
            if order_up_to_levels is not None
            else self.order_up_to_levels
        )

        supplier_order_quantities = [0, 0, 0, 0, 0]

        for product_index in range(5):
            inventory_position = self.get_inventory_position(product_index)

            if inventory_position <= active_reorder_points[product_index]:
                supplier_order_quantities[product_index] = (
                    active_order_up_to_levels[product_index] - inventory_position
                )

        base_lead_time = self.config.sample_supplier_lead_time(random_generator)
        dispatched_order = self.upstream.process_evening(
            new_order_quantities=supplier_order_quantities,
            disruption_state=self.disruption_state,
            base_lead_time=base_lead_time,
            expedite=expedite,
        )

        if dispatched_order is not None:
            self.supplier_pipeline.append(dispatched_order)

        return supplier_order_quantities

    def get_inventory_copy(self) -> List[int]:
        return list(self.inventory)

    def __repr__(self) -> str:
        return f"CentralWarehouse(inventory={self.inventory})"