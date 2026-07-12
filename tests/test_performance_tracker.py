from __future__ import annotations

import pytest

from config.input_product import get_default_products
from config.simulation_config import SimulationConfig
from metrics.performance_tracker import PerformanceTracker
from models.forward_warehouse import ForwardWarehouse


def make_forward_warehouse() -> ForwardWarehouse:
    config = SimulationConfig()
    return ForwardWarehouse(
        forward_warehouse_id=1,
        demand_intensity_multiplier=0.75,
        products=get_default_products(),
        forward_warehouse_capacity=config.forward_warehouse_capacity,
    )


def make_evaluation_snapshot(
    forward_inventory_begin: list[int],
    forward_inventory_end: list[int],
    central_inventory_begin: list[int],
    central_inventory_end: list[int],
) -> dict:
    return {
        "day": 31,
        "is_warmup": False,
        "central_inventory_begin": central_inventory_begin,
        "central_inventory_end": central_inventory_end,
        "forward_inventory_begin": {1: forward_inventory_begin},
        "forward_inventory_end": {1: forward_inventory_end},
        "shortage_flags": [False, False, False, False, False],
    }


def test_warmup_snapshot_is_skipped():
    config = SimulationConfig()
    products = get_default_products()
    tracker = PerformanceTracker(config, products)
    forward_warehouse = make_forward_warehouse()

    snapshot = {
        "day": 1,
        "is_warmup": True,
        "central_inventory_begin": [100, 100, 100, 100, 100],
        "central_inventory_end": [90, 90, 90, 90, 90],
        "forward_inventory_begin": {1: [10, 10, 10, 10, 10]},
        "forward_inventory_end": {1: [5, 5, 5, 5, 5]},
        "shortage_flags": [False] * 5,
    }

    tracker.log_daily_snapshot(snapshot, [forward_warehouse])

    with pytest.raises(ValueError, match="No evaluation days recorded"):
        tracker.summarize()


def test_daily_profit_and_fill_rate_on_one_evaluation_day():
    config = SimulationConfig(num_forward_warehouses=1)
    products = get_default_products()
    tracker = PerformanceTracker(config, products)
    forward_warehouse = make_forward_warehouse()

    forward_warehouse.today_demand = [10, 0, 0, 0, 0]
    forward_warehouse.today_sales = [8, 0, 0, 0, 0]
    forward_warehouse.today_lost_sales = [2, 0, 0, 0, 0]

    snapshot = make_evaluation_snapshot(
        forward_inventory_begin=[20, 0, 0, 0, 0],
        forward_inventory_end=[12, 0, 0, 0, 0],
        central_inventory_begin=[100, 0, 0, 0, 0],
        central_inventory_end=[90, 0, 0, 0, 0],
    )

    tracker.log_daily_snapshot(snapshot, [forward_warehouse])
    result = tracker.summarize()

    product = products[0]
    expected_revenue = product.selling_price * 8
    expected_penalty = product.lost_sales_penalty * 2
    average_forward_inventory = (20 + 12) / 2
    average_central_inventory = (100 + 90) / 2
    expected_holding_cost = (
        product.central_holding_cost * average_central_inventory
        + product.forward_holding_cost * average_forward_inventory
    )
    expected_profit = expected_revenue - expected_holding_cost - expected_penalty

    assert result["evaluation_days"] == 1
    assert result["total_profit"] == pytest.approx(expected_profit)
    assert result["fill_rate"] == pytest.approx(8 / 10)
    assert result["lost_sales_rate"] == pytest.approx(2 / 10)


def test_report_final_metrics_computes_mean_and_ci():
    replication_results = [
        {"total_profit": 100.0, "fill_rate": 0.9, "central_shortage_frequency_by_product": [0.1, 0.0, 0.0, 0.0, 0.0]},
        {"total_profit": 200.0, "fill_rate": 0.8, "central_shortage_frequency_by_product": [0.2, 0.0, 0.0, 0.0, 0.0]},
        {"total_profit": 300.0, "fill_rate": 0.7, "central_shortage_frequency_by_product": [0.3, 0.0, 0.0, 0.0, 0.0]},
    ]

    summary = PerformanceTracker.report_final_metrics(replication_results)

    assert summary["total_profit"]["mean"] == pytest.approx(200.0)
    assert summary["fill_rate"]["mean"] == pytest.approx(0.8)
    assert summary["total_profit"]["ci_half_width"] > 0
    assert len(summary["central_shortage_frequency_by_product"]) == 5
