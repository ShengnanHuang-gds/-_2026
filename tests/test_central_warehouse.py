from __future__ import annotations

import random

import pytest

from config.input_product import get_default_products
from config.simulation_config import SimulationConfig
from models.central_warehouse import CentralWarehouse
from models.forward_warehouse import ForwardWarehouse
from models.supplier_order import SupplierOrder


def make_config() -> SimulationConfig:
    return SimulationConfig()


def make_central_warehouse(forward_warehouse_ids=None) -> CentralWarehouse:
    config = make_config()
    if forward_warehouse_ids is None:
        forward_warehouse_ids = [1, 2]
    return CentralWarehouse(
        products=get_default_products(),
        central_warehouse_capacity=config.central_warehouse_capacity,
        days_of_supply_epsilon=config.days_of_supply_epsilon,
        forward_warehouse_ids=forward_warehouse_ids,
        config=config,
    )


def make_forward_warehouse(
    forward_warehouse_id: int,
    demand_intensity_multiplier: float,
) -> ForwardWarehouse:
    config = make_config()
    return ForwardWarehouse(
        forward_warehouse_id=forward_warehouse_id,
        demand_intensity_multiplier=demand_intensity_multiplier,
        products=get_default_products(),
        forward_warehouse_capacity=config.forward_warehouse_capacity,
    )


# --- 构造 / 初始状态 ---

def test_initial_inventory_equals_order_up_to_level():
    cw = make_central_warehouse()

    assert cw.inventory == cw.order_up_to_levels
    assert cw.inventory[0] == 974   # CW P1, Table 5


def test_cw_p1_policy_matches_table5():
    cw = make_central_warehouse()

    assert cw.reorder_points[0] == 429
    assert cw.order_up_to_levels[0] == 974


def test_pending_shipments_initialized_to_zero():
    cw = make_central_warehouse(forward_warehouse_ids=[1, 2, 3])

    assert cw.pending_forward_warehouse_shipments[1] == [0, 0, 0, 0, 0]
    assert cw.pending_forward_warehouse_shipments[2] == [0, 0, 0, 0, 0]
    assert cw.pending_forward_warehouse_shipments[3] == [0, 0, 0, 0, 0]


# --- advance_morning_pipeline (Algorithm 1) ---

def test_supplier_order_arrives_after_lead_time():
    cw = make_central_warehouse()
    initial_p1 = cw.inventory[0]

    cw.supplier_pipeline.append(
        SupplierOrder(quantities=[10, 0, 0, 0, 0], remaining_lead_time=2)
    )

    cw.advance_morning_pipeline()   # 2 -> 1, not arrived
    assert cw.inventory[0] == initial_p1
    assert len(cw.supplier_pipeline) == 1

    cw.advance_morning_pipeline()   # 1 -> 0, arrived
    assert cw.inventory[0] == initial_p1 + 10
    assert len(cw.supplier_pipeline) == 0


def test_multiple_supplier_orders_in_pipeline():
    cw = make_central_warehouse()
    initial_p1 = cw.inventory[0]

    cw.supplier_pipeline.append(
        SupplierOrder(quantities=[5, 0, 0, 0, 0], remaining_lead_time=1)
    )
    cw.supplier_pipeline.append(
        SupplierOrder(quantities=[7, 0, 0, 0, 0], remaining_lead_time=2)
    )

    cw.advance_morning_pipeline()
    assert cw.inventory[0] == initial_p1 + 5
    assert len(cw.supplier_pipeline) == 1

    cw.advance_morning_pipeline()
    assert cw.inventory[0] == initial_p1 + 5 + 7
    assert len(cw.supplier_pipeline) == 0


# --- pop_pending_shipments ---

def test_pop_pending_shipments_returns_and_clears():
    cw = make_central_warehouse(forward_warehouse_ids=[1, 2])
    cw.pending_forward_warehouse_shipments[1] = [3, 0, 0, 0, 0]

    shipments = cw.pop_pending_shipments(1)

    assert shipments == [3, 0, 0, 0, 0]
    assert cw.pending_forward_warehouse_shipments[1] == [0, 0, 0, 0, 0]


# --- allocate_to_fws (Algorithm 2) ---

def test_allocate_sufficient_inventory_gives_full_shipments():
    cw = make_central_warehouse(forward_warehouse_ids=[1, 2])
    fw1 = make_forward_warehouse(1, 0.75)
    fw2 = make_forward_warehouse(2, 1.00)
    forward_warehouses = [fw1, fw2]

    cw.inventory[0] = 100

    requests = {1: [30, 0, 0, 0, 0], 2: [20, 0, 0, 0, 0]}
    post_demand_inventories = {1: [5, 0, 0, 0, 0], 2: [8, 0, 0, 0, 0]}

    shipments, shortage_flags = cw.allocate_to_fws(
        requests, forward_warehouses, post_demand_inventories
    )

    assert shipments[1][0] == 30
    assert shipments[2][0] == 20
    assert shortage_flags[0] is False
    assert cw.inventory[0] == 50
    assert cw.pending_forward_warehouse_shipments[1][0] == 30
    assert cw.pending_forward_warehouse_shipments[2][0] == 20


def test_allocate_shortage_uses_proportional_floor():
    cw = make_central_warehouse(forward_warehouse_ids=[1, 2])
    fw1 = make_forward_warehouse(1, 0.75)
    fw2 = make_forward_warehouse(2, 1.00)
    forward_warehouses = [fw1, fw2]

    cw.inventory[0] = 10

    requests = {1: [8, 0, 0, 0, 0], 2: [8, 0, 0, 0, 0]}
    post_demand_inventories = {1: [2, 0, 0, 0, 0], 2: [2, 0, 0, 0, 0]}

    shipments, shortage_flags = cw.allocate_to_fws(
        requests, forward_warehouses, post_demand_inventories
    )

    # floor(10*8/16)=5 each, remainder=0
    assert shipments[1][0] == 5
    assert shipments[2][0] == 5
    assert shortage_flags[0] is True
    assert cw.inventory[0] == 0


def test_allocate_shortage_assigns_remainder_by_days_of_supply():
    cw = make_central_warehouse(forward_warehouse_ids=[1, 2])
    fw1 = make_forward_warehouse(1, 0.75)
    fw2 = make_forward_warehouse(2, 1.00)
    forward_warehouses = [fw1, fw2]

    cw.inventory[0] = 11

    requests = {1: [8, 0, 0, 0, 0], 2: [8, 0, 0, 0, 0]}
    post_demand_inventories = {
        1: [1, 0, 0, 0, 0],   # lower DOS -> gets +1 remainder
        2: [5, 0, 0, 0, 0],
    }

    shipments, shortage_flags = cw.allocate_to_fws(
        requests, forward_warehouses, post_demand_inventories
    )

    assert shipments[1][0] + shipments[2][0] == 11
    assert shipments[1][0] == 6
    assert shipments[2][0] == 5
    assert shortage_flags[0] is True


def test_shipments_never_exceed_requests_or_inventory():
    cw = make_central_warehouse(forward_warehouse_ids=[1, 2])
    fw1 = make_forward_warehouse(1, 0.75)
    fw2 = make_forward_warehouse(2, 1.50)
    forward_warehouses = [fw1, fw2]

    cw.inventory = [17, 0, 0, 0, 0]

    requests = {1: [12, 0, 0, 0, 0], 2: [10, 0, 0, 0, 0]}
    post_demand_inventories = {1: [3, 0, 0, 0, 0], 2: [4, 0, 0, 0, 0]}

    shipments, _ = cw.allocate_to_fws(
        requests, forward_warehouses, post_demand_inventories
    )

    assert shipments[1][0] <= requests[1][0]
    assert shipments[2][0] <= requests[2][0]
    assert shipments[1][0] + shipments[2][0] <= 17
    assert cw.inventory[0] == 0


def test_allocate_with_zero_total_request():
    cw = make_central_warehouse(forward_warehouse_ids=[1, 2])
    fw1 = make_forward_warehouse(1, 0.75)
    fw2 = make_forward_warehouse(2, 1.00)
    forward_warehouses = [fw1, fw2]

    before_inventory = cw.get_inventory_copy()

    shipments, shortage_flags = cw.allocate_to_fws(
        requests={1: [0, 0, 0, 0, 0], 2: [0, 0, 0, 0, 0]},
        forward_warehouses=forward_warehouses,
        post_demand_inventories={1: [0, 0, 0, 0, 0], 2: [0, 0, 0, 0, 0]},
    )

    assert shipments[1] == [0, 0, 0, 0, 0]
    assert shipments[2] == [0, 0, 0, 0, 0]
    assert shortage_flags == [False, False, False, False, False]
    assert cw.inventory == before_inventory


# --- check_supplier_ordering (Step 6) ---

def test_check_supplier_ordering_creates_pipeline_order():
    cw = make_central_warehouse()
    cw.inventory = [0, 0, 0, 0, 0]
    cw.supplier_pipeline = []

    rng = random.Random(42)
    order_quantities = cw.check_supplier_ordering(rng)

    assert sum(order_quantities) > 0
    assert len(cw.supplier_pipeline) == 1
    assert cw.supplier_pipeline[0].remaining_lead_time in [1, 2, 3]


def test_check_supplier_ordering_counts_pipeline_in_inventory_position():
    cw = make_central_warehouse()
    cw.inventory[0] = 0
    cw.supplier_pipeline = [
        SupplierOrder(quantities=[100, 0, 0, 0, 0], remaining_lead_time=2)
    ]

    rng = random.Random(42)
    order_quantities = cw.check_supplier_ordering(rng)

    # on-hand 0 + pipeline 100 should reduce order size vs empty pipeline case
    assert order_quantities[0] < cw.order_up_to_levels[0]


def test_check_supplier_ordering_skips_when_no_reorder_needed():
    cw = make_central_warehouse()
    before_pipeline_len = len(cw.supplier_pipeline)

    rng = random.Random(42)
    order_quantities = cw.check_supplier_ordering(rng)

    assert order_quantities == [0, 0, 0, 0, 0]
    assert len(cw.supplier_pipeline) == before_pipeline_len


# --- 手算组合一例 ---

def test_hand_trace_pipeline_allocate_and_order():
    """
    Simplified CW trace for product P1:

    1. Start with low P1 inventory
    2. FW requests exceed CW stock -> shortage allocation
    3. Pending buffer updated
    4. Supplier order created
    """
    cw = make_central_warehouse(forward_warehouse_ids=[1, 2])
    fw1 = make_forward_warehouse(1, 0.75)
    fw2 = make_forward_warehouse(2, 1.00)
    forward_warehouses = [fw1, fw2]

    cw.inventory[0] = 10

    requests = {1: [8, 0, 0, 0, 0], 2: [8, 0, 0, 0, 0]}
    post_demand_inventories = {1: [2, 0, 0, 0, 0], 2: [2, 0, 0, 0, 0]}

    shipments, shortage_flags = cw.allocate_to_fws(
        requests, forward_warehouses, post_demand_inventories
    )

    assert shipments[1][0] == 5
    assert shipments[2][0] == 5
    assert shortage_flags[0] is True
    assert cw.pending_forward_warehouse_shipments[1][0] == 5

    rng = random.Random(1)
    order_quantities = cw.check_supplier_ordering(rng)

    assert sum(order_quantities) > 0
    assert len(cw.supplier_pipeline) == 1