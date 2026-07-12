from __future__ import annotations

import math
import random
from typing import Dict, List

from config.policy import compute_central_warehouse_policy
from config.simulation_config import SimulationConfig
from models.forward_warehouse import ForwardWarehouse
from models.product import Product
from models.supplier_order import SupplierOrder


class CentralWarehouse:
    """Central warehouse (PDF Section 8 + Section 9 + Algorithm 1/2)."""

    def __init__(
        self,
        products: List[Product],
        central_warehouse_capacity: int,
        days_of_supply_epsilon: float,
        forward_warehouse_ids: List[int],
        config: SimulationConfig,
    ) -> None:
        self.config = config
        self.products = products
        self.central_warehouse_capacity = central_warehouse_capacity
        self.days_of_supply_epsilon = days_of_supply_epsilon

        reorder_points, order_up_to_levels = compute_central_warehouse_policy(products)
        self.reorder_points = reorder_points
        self.order_up_to_levels = order_up_to_levels

        # Section 15: initial state 等于 upper bound
        self.inventory = list(order_up_to_levels)

        self.supplier_pipeline: List[SupplierOrder] = []

        # shipments dispatched at end of day t, arrive next morning
        self.pending_forward_warehouse_shipments: Dict[int, List[int]] = {
            forward_warehouse_id: [0, 0, 0, 0, 0]
            for forward_warehouse_id in forward_warehouse_ids
        }

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
    ) -> tuple[Dict[int, List[int]], List[bool]]:
        """
        Algorithm 2 / Step 5.

        requests: {forward_warehouse_id: RF vector}
        post_demand_inventories: inventory after sales, before allocation
        """
        shipments = {
            forward_warehouse.forward_warehouse_id: [0, 0, 0, 0, 0]
            for forward_warehouse in forward_warehouses
        }
        shortage_flags = [False, False, False, False, False]

        for product_index in range(5):
            #allocate for each product
            product_shipments, product_shortage = self._allocate_one_product(
                product_index=product_index,
                requests=requests,
                forward_warehouses=forward_warehouses,
                post_demand_inventories=post_demand_inventories,
            )

            for forward_warehouse in forward_warehouses:
                fw_id = forward_warehouse.forward_warehouse_id
                shipped_quantity = product_shipments[fw_id]
#记入今晚总发货回执
                shipments[fw_id][product_index] = shipped_quantity
                #加入缓冲区
                self.pending_forward_warehouse_shipments[fw_id][product_index] += (
                    shipped_quantity
                )
                #deduct product quantity in cw
                self.inventory[product_index] -= shipped_quantity

            shortage_flags[product_index] = product_shortage

        return shipments, shortage_flags

    def _allocate_one_product(
        self,
        product_index: int,
        requests: Dict[int, List[int]],
        forward_warehouses: List[ForwardWarehouse],
        post_demand_inventories: Dict[int, List[int]],
    ) -> tuple[Dict[int, int], bool]:#return shipment and a bool if cw can 
    #satisify fw return false else return true
        """Algorithm 2 for one product k."""
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

        #if cw satisify all fw request
        if total_request <= central_inventory:
            for fw_id, request_quantity in forward_requests.items():
                shipments[fw_id] = request_quantity
            return shipments, False

        # shortage case
        shortage = True
        initial_shipments = {}


        #section9 proportional allocation followed by a remainder allocation
        for forward_warehouse in forward_warehouses:
            fw_id = forward_warehouse.forward_warehouse_id
            request_quantity = forward_requests[fw_id]
            initial_shipments[fw_id] = math.floor(
                central_inventory * request_quantity / total_request
            )


#section9 unallocated remainder
        remainder = central_inventory - sum(initial_shipments.values())

#section 9 compute dos of all fw
# remainder is assigned one unit at a time to FWs with the lowest days of supply
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
                #compute dos 
                days_of_supply = (
                    post_demand_inventory + current_shipment
                ) / (daily_demand_mean + self.days_of_supply_epsilon)


                #find out the repository with minimal dos
                if best_fw_id is None or days_of_supply < best_days_of_supply:
                    best_fw_id = fw_id
                    best_days_of_supply = days_of_supply
                    
                    #tie break same dos assign to fw with smaller index
                elif (
                    days_of_supply == best_days_of_supply
                    and fw_id < best_fw_id
                ):
                    # tie-break: smaller FW id first
                    best_fw_id = fw_id
                    best_days_of_supply = days_of_supply

            if best_fw_id is None:
                break

            #assian as unit
            initial_shipments[best_fw_id] += 1
            remainder -= 1

        for fw_id, shipped_quantity in initial_shipments.items():
            shipments[fw_id] = shipped_quantity

        return shipments, shortage

    def check_supplier_ordering(self, random_generator: random.Random) -> List[int]:
        """Step 6: place one supplier order if needed."""
        supplier_order_quantities = [0, 0, 0, 0, 0]

        for product_index in range(5):
           
            inventory_position = self.inventory[product_index]

             #section 8.1 inventory position
            for order in self.supplier_pipeline:
                inventory_position += order.quantities[product_index]
   
             # if trigger policy s add to upper bound 
            if inventory_position <= self.reorder_points[product_index]:
                supplier_order_quantities[product_index] = (
                    self.order_up_to_levels[product_index] - inventory_position
                )

        if sum(supplier_order_quantities) > 0:
            lead_time = self.config.sample_supplier_lead_time(random_generator)
            self.supplier_pipeline.append(
                SupplierOrder(
                    quantities=supplier_order_quantities,
                    remaining_lead_time=lead_time,
                )
            )

        return supplier_order_quantities

    def get_inventory_copy(self) -> List[int]:
        return list(self.inventory)

    def __repr__(self) -> str:
        return f"CentralWarehouse(inventory={self.inventory})"