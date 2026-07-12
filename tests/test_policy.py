from __future__ import annotations
from config.input_product import (
    get_default_products,
    get_default_forward_warehouse_demand_intensity_multipliers,
)
from config.policy import (
    compute_forward_warehouse_policy,
    compute_central_warehouse_policy,
    compute_daily_demand_mean,
)

EXPECTED_FORWARD_WAREHOUSE_POLICY = {
    1: [(13, 36), (10, 25), (7, 17), (5, 12), (4, 8)],
    2: [(15, 40), (11, 28), (7, 19), (5, 13), (4, 9)],
    3: [(16, 44), (12, 31), (8, 21), (6, 14), (4, 10)],
    4: [(17, 46), (12, 33), (8, 22), (6, 14), (4, 11)],
    5: [(18, 48), (13, 34), (9, 23), (6, 15), (4, 11)],
    6: [(18, 50), (13, 35), (9, 24), (6, 16), (5, 11)],
    7: [(20, 54), (14, 38), (10, 25), (7, 17), (5, 12)],
    8: [(20, 57), (15, 39), (10, 26), (7, 17), (5, 12)],
    9: [(22, 61), (16, 42), (11, 28), (7, 18), (5, 13)],
    10: [(24, 67), (17, 46), (12, 31), (8, 20), (6, 14)],
}
EXPECTED_FORWARD_WAREHOUSE_ORDER_UP_TO_SUMS = {
    1: 98,
    2: 109,
    3: 120,
    4: 126,
    5: 131,
    6: 136,
    7: 146,
    8: 151,
    9: 162,
    10: 178,
}

EXPECTED_CENTRAL_WAREHOUSE_POLICY = [
    (429, 974),   
    (291, 657), 
    (187, 418),  
    (116, 256),  
    (80, 175),   
]

def test_daily_demand_mean_fw1_p1():
    products = get_default_products()
    multipliers = get_default_forward_warehouse_demand_intensity_multipliers()
    result = compute_daily_demand_mean(multipliers[0], products[0])
    assert result == 9.0
def test_forward_warehouse_policy_matches_table4():
    products = get_default_products()
    multipliers = get_default_forward_warehouse_demand_intensity_multipliers()
    for fw_id in range(1, 11):
        reorder_points, order_up_to_levels = compute_forward_warehouse_policy(
            multipliers[fw_id - 1], products
        )
        actual = list(zip(reorder_points, order_up_to_levels))
        expected = EXPECTED_FORWARD_WAREHOUSE_POLICY[fw_id]
        assert actual == expected
def test_forward_warehouse_order_up_to_sums_match_table4():
    products = get_default_products()
    multipliers = get_default_forward_warehouse_demand_intensity_multipliers()
    for fw_id in range(1, 11):
        _, order_up_to_levels = compute_forward_warehouse_policy(
            multipliers[fw_id - 1], products
        )
        assert sum(order_up_to_levels) == EXPECTED_FORWARD_WAREHOUSE_ORDER_UP_TO_SUMS[fw_id]
def test_central_warehouse_policy_matches_table5():
    products = get_default_products()
    reorder_points, order_up_to_levels = compute_central_warehouse_policy(products)
    actual = list(zip(reorder_points, order_up_to_levels))
    assert actual == EXPECTED_CENTRAL_WAREHOUSE_POLICY
def test_central_warehouse_capacity_feasibility():
    products = get_default_products()
    _, order_up_to_levels = compute_central_warehouse_policy(products)
    # PDF: sum(SC_k) = 2480 < CC = 3600
    assert sum(order_up_to_levels) == 2480
    assert sum(order_up_to_levels) < 3600
def test_forward_warehouse_capacity_feasibility():
    products = get_default_products()
    multipliers = get_default_forward_warehouse_demand_intensity_multipliers()
    for fw_id, multiplier in enumerate(multipliers, start=1):
        _, order_up_to_levels = compute_forward_warehouse_policy(
            multiplier, products
        )
        assert sum(order_up_to_levels) <= 180