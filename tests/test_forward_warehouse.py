from __future__ import annotations

import pytest

from config.input_product import get_default_products
from config.simulation_config import SimulationConfig
from models.forward_warehouse import ForwardWarehouse


def make_forward_warehouse(
    forward_warehouse_id: int = 1,
    demand_intensity_multiplier: float = 0.75,
) -> ForwardWarehouse:
    config = SimulationConfig()
    return ForwardWarehouse(
        forward_warehouse_id=forward_warehouse_id,
        demand_intensity_multiplier=demand_intensity_multiplier,
        products=get_default_products(),
        forward_warehouse_capacity=config.forward_warehouse_capacity,
    )


def test_initial_inventory_equals_order_up_to_level():
    fw = make_forward_warehouse()

    assert fw.inventory == fw.order_up_to_levels
    assert fw.inventory[0] == 36   # FW1 P1, Table 4


def test_fw1_p1_policy_matches_table4():
    fw = make_forward_warehouse()

    assert fw.reorder_points[0] == 13
    assert fw.order_up_to_levels[0] == 36


def test_get_daily_demand_means_fw1_p1():
    fw = make_forward_warehouse()

    assert fw.get_daily_demand_means()[0] == 9.0   # 0.75 * 12


# --- receive_shipments ---

def test_receive_shipments_increases_inventory():
    fw = make_forward_warehouse()
    before = fw.get_inventory_copy()

    fw.receive_shipments([5, 0, 0, 0, 0])

    assert fw.inventory[0] == before[0] + 5
    assert fw.inventory[1:] == before[1:]


def test_receive_shipments_rejects_wrong_length():
    fw = make_forward_warehouse()

    with pytest.raises(ValueError, match="exactly 5"):
        fw.receive_shipments([1, 2, 3])


# --- execute_daily_sales ---

def test_execute_daily_sales_no_lost_sales_when_inventory_enough():
    fw = make_forward_warehouse()
    fw.inventory[0] = 41

    fw.execute_daily_sales([20, 0, 0, 0, 0])

    assert fw.today_sales[0] == 20
    assert fw.today_lost_sales[0] == 0
    assert fw.inventory[0] == 21
    assert fw.today_demand[0] == 20


def test_execute_daily_sales_with_lost_sales():
    fw = make_forward_warehouse()
    fw.inventory[0] = 10

    fw.execute_daily_sales([20, 0, 0, 0, 0])

    assert fw.today_sales[0] == 10
    assert fw.today_lost_sales[0] == 10
    assert fw.inventory[0] == 0


def test_execute_daily_sales_rejects_wrong_length():
    fw = make_forward_warehouse()

    with pytest.raises(ValueError, match="exactly 5"):
        fw.execute_daily_sales([1, 2, 3])


# --- check_replenishment_request ---

def test_check_replenishment_request_when_inventory_above_reorder_point():
    fw = make_forward_warehouse()
    fw.inventory[0] = 21   # s=13, still above reorder point

    requests = fw.check_replenishment_request()

    assert requests[0] == 0


def test_check_replenishment_request_when_inventory_at_or_below_reorder_point():
    fw = make_forward_warehouse()
    fw.inventory[0] = 6   # s=13, S=36

    requests = fw.check_replenishment_request()

    assert requests[0] == 30   # 36 - 6


def test_check_replenishment_request_when_inventory_is_zero():
    fw = make_forward_warehouse()
    fw.inventory[0] = 0

    requests = fw.check_replenishment_request()

    assert requests[0] == 36   # 36 - 0


# --- 手算完整一例 ---

def test_hand_trace_fw1_p1_one_day():
    """
    FW1, product P1 hand trace:

    Initial IF = 36
    Step 2: receive 5  -> IF = 41
    Step 3: demand 20  -> Y = 20, LS = 0, IF = 21
    Step 4: s = 13     -> 21 > 13, RF = 0
    """
    fw = make_forward_warehouse()

    assert fw.inventory[0] == 36

    fw.receive_shipments([5, 0, 0, 0, 0])
    assert fw.inventory[0] == 41

    fw.execute_daily_sales([20, 0, 0, 0, 0])
    assert fw.today_sales[0] == 20
    assert fw.today_lost_sales[0] == 0
    assert fw.inventory[0] == 21

    requests = fw.check_replenishment_request()
    assert requests[0] == 0