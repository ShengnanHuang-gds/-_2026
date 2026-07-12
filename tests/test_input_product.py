from __future__ import annotations

import pytest

from config.input_product import get_default_products
from config.simulation_config import SimulationConfig
from simulation.simulation_engine import SimulationEngine


def test_default_holding_ratio_matches_table2():
    product = get_default_products(forward_holding_cost_ratio=3.0)[0]

    assert product.central_holding_cost == 0.10
    assert product.forward_holding_cost == pytest.approx(0.30)


def test_holding_ratio_scales_forward_cost():
    product = get_default_products(forward_holding_cost_ratio=5.0)[0]

    assert product.forward_holding_cost == pytest.approx(product.central_holding_cost * 5.0)


def test_engine_uses_config_demand_intensity_scale():
    config = SimulationConfig(demand_intensity_scale=1.2)
    engine = SimulationEngine(config, seed=1)

    assert engine.demand_intensity_multipliers[0] == pytest.approx(0.75 * 1.2)


def test_engine_uses_config_forward_holding_cost_ratio():
    config = SimulationConfig(forward_holding_cost_ratio=5.0)
    engine = SimulationEngine(config, seed=1)

    assert engine.products[0].forward_holding_cost == pytest.approx(0.50)
