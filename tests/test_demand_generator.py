from __future__ import annotations

from config.input_product import get_default_products
from config.simulation_config import SimulationConfig
from models.demand_generator import DemandGenerator
from models.forward_warehouse import ForwardWarehouse


def make_forward_warehouses() -> list[ForwardWarehouse]:
    config = SimulationConfig()
    products = get_default_products()
    multipliers = [0.75, 0.85, 0.95, 1.00, 1.05, 1.10, 1.20, 1.25, 1.35, 1.50]

    return [
        ForwardWarehouse(
            forward_warehouse_id=forward_warehouse_id,
            demand_intensity_multiplier=multipliers[forward_warehouse_id - 1],
            products=products,
            forward_warehouse_capacity=config.forward_warehouse_capacity,
        )
        for forward_warehouse_id in range(1, 11)
    ]


def test_generate_all_returns_ten_forward_warehouses_with_five_products():
    forward_warehouses = make_forward_warehouses()
    products = get_default_products()
    generator = DemandGenerator(seed=42)

    demands = generator.generate_all(forward_warehouses, products)

    assert len(demands) == 10
    for forward_warehouse_id in range(1, 11):
        assert forward_warehouse_id in demands
        assert len(demands[forward_warehouse_id]) == 5


def test_generated_demands_are_non_negative_integers():
    forward_warehouses = make_forward_warehouses()
    products = get_default_products()
    generator = DemandGenerator(seed=7)

    demands = generator.generate_all(forward_warehouses, products)

    for demand_vector in demands.values():
        for demand in demand_vector:
            assert isinstance(demand, int)
            assert demand >= 0


def test_same_seed_produces_same_demands():
    forward_warehouses = make_forward_warehouses()
    products = get_default_products()

    demands_a = DemandGenerator(seed=99).generate_all(forward_warehouses, products)
    demands_b = DemandGenerator(seed=99).generate_all(forward_warehouses, products)

    assert demands_a == demands_b


def test_different_seed_produces_different_demands():
    forward_warehouses = make_forward_warehouses()
    products = get_default_products()

    demands_a = DemandGenerator(seed=1).generate_all(forward_warehouses, products)
    demands_b = DemandGenerator(seed=2).generate_all(forward_warehouses, products)

    assert demands_a != demands_b
