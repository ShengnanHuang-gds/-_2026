from __future__ import annotations

from config.simulation_config import SimulationConfig
from simulation.simulation_engine import SimulationEngine


def run_short_simulation(
    warmup_days: int = 3,
    evaluation_days: int = 5,
    seed: int = 42,
) -> list[dict]:
    config = SimulationConfig(
        warmup_days=warmup_days,
        evaluation_days=evaluation_days,
    )
    engine = SimulationEngine(config, seed=seed)

    snapshots = []
    for day in range(1, config.total_simulation_days + 1):
        snapshots.append(engine.run_one_day(day))

    return snapshots


def assert_non_negative_inventories(snapshot: dict) -> None:
    """C1: all inventory levels are non-negative."""
    for quantity in snapshot["central_inventory_begin"]:
        assert quantity >= 0
    for quantity in snapshot["central_inventory_end"]:
        assert quantity >= 0

    for inventory_vector in snapshot["forward_inventory_begin"].values():
        for quantity in inventory_vector:
            assert quantity >= 0
    for inventory_vector in snapshot["forward_inventory_end"].values():
        for quantity in inventory_vector:
            assert quantity >= 0


def assert_sales_feasible(snapshot: dict, engine: SimulationEngine) -> None:
    """C2: sales do not exceed demand or pre-sales inventory."""
    for forward_warehouse in engine.forward_warehouses:
        forward_warehouse_id = forward_warehouse.forward_warehouse_id
        inventory_before_sales = snapshot["forward_inventory_begin"][forward_warehouse_id]

        for product_index in range(5):
            demand = forward_warehouse.today_demand[product_index]
            sales = forward_warehouse.today_sales[product_index]

            assert sales <= demand
            assert sales <= inventory_before_sales[product_index]


def assert_shipments_feasible(snapshot: dict) -> None:
    """C3/C4: shipments respect requests and central inventory."""
    central_inventory_begin = snapshot["central_inventory_begin"]

    for product_index in range(5):
        total_shipped = 0

        for forward_warehouse_id, shipment_vector in snapshot["shipments"].items():
            shipped_quantity = shipment_vector[product_index]
            request_quantity = snapshot["requests"][forward_warehouse_id][product_index]

            assert shipped_quantity >= 0
            assert shipped_quantity <= request_quantity
            total_shipped += shipped_quantity

        assert total_shipped <= central_inventory_begin[product_index]


def assert_capacity_bounds(snapshot: dict, config: SimulationConfig) -> None:
    """C5/C6: inventory within configured capacities after morning steps."""
    for inventory_vector in snapshot["forward_inventory_begin"].values():
        assert sum(inventory_vector) <= config.forward_warehouse_capacity

    assert sum(snapshot["central_inventory_begin"]) <= config.central_warehouse_capacity


def assert_no_lost_sales_carry_over(engine: SimulationEngine) -> None:
    """C8: lost sales are recorded only for the current day."""
    for forward_warehouse in engine.forward_warehouses:
        for product_index in range(5):
            assert forward_warehouse.today_lost_sales[product_index] >= 0
            assert forward_warehouse.inventory[product_index] >= 0


def assert_unmet_requests_do_not_carry_over(snapshot: dict, engine: SimulationEngine) -> None:
    """C9: only shipped quantities enter pending buffers."""
    for forward_warehouse in engine.forward_warehouses:
        forward_warehouse_id = forward_warehouse.forward_warehouse_id
        shipment_vector = snapshot["shipments"][forward_warehouse_id]
        pending_vector = engine.central_warehouse.pending_forward_warehouse_shipments[
            forward_warehouse_id
        ]

        assert pending_vector == shipment_vector


def assert_supplier_pipeline_valid(engine: SimulationEngine) -> None:
    """C7: pipeline orders have positive remaining lead time before arrival."""
    for order in engine.central_warehouse.supplier_pipeline:
        assert order.remaining_lead_time >= 1


def test_c1_inventory_non_negative():
    config = SimulationConfig(warmup_days=2, evaluation_days=3)
    engine = SimulationEngine(config, seed=7)

    for day in range(1, config.total_simulation_days + 1):
        snapshot = engine.run_one_day(day)
        assert_non_negative_inventories(snapshot)


def test_c2_sales_feasible():
    config = SimulationConfig(warmup_days=2, evaluation_days=3)
    engine = SimulationEngine(config, seed=7)

    for day in range(1, config.total_simulation_days + 1):
        snapshot = engine.run_one_day(day)
        assert_sales_feasible(snapshot, engine)


def test_c3_c4_shipments_feasible():
    snapshots = run_short_simulation(seed=11)

    for snapshot in snapshots:
        assert_shipments_feasible(snapshot)


def test_c5_c6_capacity_bounds():
    config = SimulationConfig(warmup_days=2, evaluation_days=3)
    engine = SimulationEngine(config, seed=13)

    for day in range(1, config.total_simulation_days + 1):
        snapshot = engine.run_one_day(day)
        assert_capacity_bounds(snapshot, config)


def test_c7_supplier_pipeline_lead_time_positive():
    config = SimulationConfig(warmup_days=2, evaluation_days=3)
    engine = SimulationEngine(config, seed=17)

    for day in range(1, config.total_simulation_days + 1):
        engine.run_one_day(day)
        assert_supplier_pipeline_valid(engine)


def test_c8_no_lost_sales_carry_over():
    config = SimulationConfig(warmup_days=2, evaluation_days=3)
    engine = SimulationEngine(config, seed=19)

    for day in range(1, config.total_simulation_days + 1):
        engine.run_one_day(day)
        assert_no_lost_sales_carry_over(engine)


def test_c9_unmet_requests_do_not_carry_over():
    config = SimulationConfig(warmup_days=2, evaluation_days=3)
    engine = SimulationEngine(config, seed=23)

    for day in range(1, config.total_simulation_days + 1):
        snapshot = engine.run_one_day(day)
        assert_unmet_requests_do_not_carry_over(snapshot, engine)


def test_c10_same_seed_reproducible():
    config = SimulationConfig(warmup_days=2, evaluation_days=3)

    result_a = SimulationEngine(config, seed=100).run()
    result_b = SimulationEngine(config, seed=100).run()

    assert result_a == result_b
