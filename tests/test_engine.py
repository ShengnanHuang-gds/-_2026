from __future__ import annotations

from config.simulation_config import SimulationConfig
from simulation.simulation_engine import SimulationEngine


def test_engine_builds_network():
    engine = SimulationEngine(SimulationConfig(), seed=42)

    assert len(engine.forward_warehouses) == 10
    assert engine.central_warehouse is not None
    assert engine.demand_generator is not None
    assert engine.performance_tracker is not None


def test_run_one_day_returns_complete_snapshot():
    engine = SimulationEngine(SimulationConfig(), seed=42)

    snapshot = engine.run_one_day(day=1)

    assert snapshot["day"] == 1
    assert snapshot["is_warmup"] is True
    assert "central_inventory_begin" in snapshot
    assert "central_inventory_end" in snapshot
    assert "forward_inventory_begin" in snapshot
    assert "forward_inventory_end" in snapshot
    assert "demands" in snapshot
    assert "requests" in snapshot
    assert "shipments" in snapshot
    assert "shortage_flags" in snapshot
    assert len(snapshot["demands"]) == 10


def test_warmup_flag_switches_after_warmup_days():
    config = SimulationConfig(warmup_days=2, evaluation_days=1)
    engine = SimulationEngine(config, seed=1)

    warmup_snapshot = engine.run_one_day(day=2)
    eval_snapshot = engine.run_one_day(day=3)

    assert warmup_snapshot["is_warmup"] is True
    assert eval_snapshot["is_warmup"] is False


def test_run_returns_metrics_dict():
    config = SimulationConfig(warmup_days=1, evaluation_days=2)
    engine = SimulationEngine(config, seed=42)

    result = engine.run()

    assert result["evaluation_days"] == 2
    assert "fill_rate" in result
    assert "total_profit" in result
    assert "central_shortage_frequency_by_product" in result


def test_run_is_reproducible_with_same_seed():
    config = SimulationConfig(warmup_days=1, evaluation_days=2)

    result_a = SimulationEngine(config, seed=99).run()
    result_b = SimulationEngine(config, seed=99).run()

    assert result_a == result_b
