from __future__ import annotations

import pytest

from config.disruption_chain import (
    DisruptionState,
    SupplierAvailability,
    TransportState,
)
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
    assert snapshot["disruption_state"] == "Z_00"
    assert snapshot["supplier_availability"] == 0
    assert snapshot["transport_state"] == 0
    assert "supplier_backlog" in snapshot
    assert "transport_waiting" in snapshot


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
    assert "central_zero_inventory_frequency_by_product" in result
    assert "forward_stockout_frequency_by_product" in result


def test_run_is_reproducible_with_same_seed():
    config = SimulationConfig(warmup_days=1, evaluation_days=2)

    result_a = SimulationEngine(config, seed=99).run()
    result_b = SimulationEngine(config, seed=99).run()

    assert result_a == result_b


def test_disruption_disabled_stays_at_z00():
    config = SimulationConfig(
        warmup_days=0,
        evaluation_days=20,
        enable_disruption=False,
    )
    engine = SimulationEngine(config, seed=7)

    codes = [engine.run_one_day(day=d)["disruption_state"] for d in range(1, 21)]

    assert all(code == "Z_00" for code in codes)


def test_disruption_enabled_can_leave_z00():
    config = SimulationConfig(
        warmup_days=0,
        evaluation_days=500,
        enable_disruption=True,
    )
    engine = SimulationEngine(config, seed=123)

    codes = [engine.run_one_day(day=d)["disruption_state"] for d in range(1, 501)]

    assert any(code != "Z_00" for code in codes)


def test_fixed_disruption_state_stays_locked():
    fixed = DisruptionState(
        supplier_availability=SupplierAvailability.A3,
        transport_state=TransportState.U3,
    )
    config = SimulationConfig(
        warmup_days=0,
        evaluation_days=5,
        enable_disruption=True,
    )
    engine = SimulationEngine(config, seed=1, fixed_disruption_state=fixed)

    codes = [engine.run_one_day(day=d)["disruption_state"] for d in range(1, 6)]

    assert all(code == "Z_33" for code in codes)
    assert engine.central_warehouse.disruption_state.code() == "Z_33"


def test_supply_only_keeps_u0_and_can_leave_a0():
    config = SimulationConfig(
        warmup_days=0,
        evaluation_days=500,
        disruption_mode="supply_only",
    )
    engine = SimulationEngine(config, seed=123)

    availabilities = []
    transport_states = []
    for day in range(1, 501):
        snapshot = engine.run_one_day(day=day)
        availabilities.append(snapshot["supplier_availability"])
        transport_states.append(snapshot["transport_state"])

    assert all(state == 0 for state in transport_states)
    assert any(availability > 0 for availability in availabilities)


def test_transport_only_keeps_a0_and_can_leave_u0():
    config = SimulationConfig(
        warmup_days=0,
        evaluation_days=500,
        disruption_mode="transport_only",
    )
    engine = SimulationEngine(config, seed=123)

    availabilities = []
    transport_states = []
    for day in range(1, 501):
        snapshot = engine.run_one_day(day=day)
        availabilities.append(snapshot["supplier_availability"])
        transport_states.append(snapshot["transport_state"])

    assert all(availability == 0 for availability in availabilities)
    assert any(state > 0 for state in transport_states)


def test_disruption_mode_none_matches_enable_false():
    config = SimulationConfig(disruption_mode="none")
    assert config.disruption_mode == "none"
    assert config.enable_disruption is False


def test_enable_disruption_true_maps_to_both():
    config = SimulationConfig(enable_disruption=True)
    assert config.disruption_mode == "both"
    assert config.enable_disruption is True


def test_default_matrix_profile_is_pdf():
    config = SimulationConfig()
    assert config.disruption_matrix_profile == "pdf"
    assert config.m_a[0][0] == pytest.approx(0.9950)


def test_harsh_matrix_profile_is_loaded():
    config = SimulationConfig(disruption_matrix_profile="harsh_mild")
    assert config.disruption_matrix_profile == "harsh_mild"
    assert config.m_a[0][0] == pytest.approx(0.9700)
    assert config.m_u[0][0] == pytest.approx(0.9600)


def test_custom_matrices_override_profile():
    custom_a = [
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
    ]
    custom_u = [
        [0.6, 0.4, 0.0, 0.0],
        [0.4, 0.6, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
    ]
    config = SimulationConfig(custom_m_a=custom_a, custom_m_u=custom_u)
    assert config.disruption_matrix_profile == "custom"
    assert config.m_a[0] == [0.5, 0.5, 0.0, 0.0]


def test_unknown_matrix_profile_raises_on_config():
    with pytest.raises(ValueError, match="unknown disruption_matrix_profile"):
        SimulationConfig(disruption_matrix_profile="nope")


def test_run_one_day_none_and_baseline_action_match():
    config = SimulationConfig(
        warmup_days=0,
        evaluation_days=5,
        disruption_mode="none",
    )
    engine_a = SimulationEngine(config, seed=11)
    engine_b = SimulationEngine(config, seed=11)

    snaps_a = [engine_a.run_one_day(day=d) for d in range(1, 6)]
    snaps_b = [
        engine_b.run_one_day(day=d, action=0) for d in range(1, 6)
    ]

    for snap_a, snap_b in zip(snaps_a, snaps_b):
        assert snap_a["requests"] == snap_b["requests"]
        assert snap_a["shipments"] == snap_b["shipments"]
        assert snap_a["central_inventory_end"] == snap_b["central_inventory_end"]
        assert snap_a["supplier_backlog"] == snap_b["supplier_backlog"]
        assert snap_b["action_id"] == 0
        assert snap_b["action"] == (0, 0, 0)


def test_run_with_baseline_action_matches_no_action():
    config = SimulationConfig(
        warmup_days=1,
        evaluation_days=3,
        disruption_mode="both",
    )
    result_none = SimulationEngine(config, seed=21).run()
    result_baseline = SimulationEngine(config, seed=21).run(action=0)
    assert result_none == result_baseline


def test_raised_buffer_action_changes_requests_when_inventory_low():
    from mdp.actions import encode_action

    config = SimulationConfig(
        warmup_days=0,
        evaluation_days=1,
        disruption_mode="none",
    )
    engine = SimulationEngine(config, seed=3)
    # Drain FW1 inventory so reorder can trigger under raised s
    engine.forward_warehouses[0].inventory = [5, 5, 5, 5, 5]

    snap_base = engine.run_one_day(day=1, action=0)
    engine2 = SimulationEngine(config, seed=3)
    engine2.forward_warehouses[0].inventory = [5, 5, 5, 5, 5]
    action_raise = encode_action(ell=2, expedite=0, allocation=1)
    snap_raise = engine2.run_one_day(day=1, action=action_raise)

    # Raised FW reorder point should request at least as much for FW1
    assert sum(snap_raise["requests"][1]) >= sum(snap_base["requests"][1])
