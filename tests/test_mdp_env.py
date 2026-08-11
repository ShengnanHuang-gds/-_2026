from __future__ import annotations

import pytest

from config.simulation_config import SimulationConfig
from mdp.env import MDPState, RetailMDPEnv
from simulation.simulation_engine import SimulationEngine


def _small_config(**kwargs) -> SimulationConfig:
    defaults = dict(warmup_days=2, evaluation_days=3, num_forward_warehouses=2)
    defaults.update(kwargs)
    return SimulationConfig(**defaults)


def test_reset_same_seed_same_state():
    config = _small_config()
    s1 = RetailMDPEnv(config, seed=99).reset()
    s2 = RetailMDPEnv(config, seed=99).reset()
    assert s1 == s2


def test_reset_different_seed_different_state():
    config = _small_config()
    s1 = RetailMDPEnv(config, seed=1).reset()
    s2 = RetailMDPEnv(config, seed=2).reset()
    assert s1 != s2


def test_reset_returns_post_demand_state_fields():
    config = _small_config()
    state = RetailMDPEnv(config, seed=42).reset()

    assert isinstance(state, MDPState)
    assert state.day == 1
    assert len(state.central_inventory) == 5
    assert len(state.forward_inventory) == 2
    assert len(state.forward_inventory[0]) == 5
    assert state.disruption_state.startswith("Z_")


def test_post_demand_inventory_is_after_sales_not_morning():
    """Post-demand CW inventory should reflect sales, not morning begin."""
    config = _small_config(warmup_days=0, evaluation_days=1)
    env = RetailMDPEnv(config, seed=7)
    state = env.reset()

    # Manually run morning only to compare
    engine = SimulationEngine(config, seed=7)
    engine.run_morning(1)
    obs = engine.post_demand_obs

    assert state.central_inventory == tuple(obs["central_inventory"])
    # FW inventory after sales should be <= morning begin in general
    # (at minimum the fields exist and match engine obs)
    assert state.forward_inventory[0] == tuple(obs["forward_inventory"][1])


def test_step_advances_eval_day_and_done():
    config = _small_config(warmup_days=0, evaluation_days=2)
    env = RetailMDPEnv(config, seed=1)
    env.reset()
    assert env._eval_day == 1
    assert env.done is False

    _, _, done, _ = env.step(0)
    assert env._eval_day == 2
    assert done is False

    _, _, done, _ = env.step(0)
    assert env._eval_day == 2  # stays at last eval day on terminal step
    assert done is True
    assert env.done is True


def test_step_after_done_raises():
    config = _small_config(warmup_days=0, evaluation_days=1)
    env = RetailMDPEnv(config, seed=1)
    env.reset()
    env.step(0)
    with pytest.raises(RuntimeError, match="episode is over"):
        env.step(0)


def test_invalid_action_raises():
    config = _small_config(warmup_days=0, evaluation_days=1)
    env = RetailMDPEnv(config, seed=1)
    env.reset()
    with pytest.raises(ValueError, match="action_id"):
        env.step(99)


def test_cumulative_reward_matches_engine_run():
    """Sum of env.step rewards ≈ engine.run total_profit for fixed action."""
    config = _small_config(warmup_days=1, evaluation_days=3)
    action_id = 0
    seed = 55

    env = RetailMDPEnv(config, seed=seed)
    env.reset()
    total = 0.0
    while not env.done:
        _, reward, _, _ = env.step(action_id)
        total += reward

    engine_result = SimulationEngine(config, seed=seed).run(action=action_id)
    assert total == pytest.approx(engine_result["total_profit"])


def test_expedite_action_reduces_reward_vs_no_expedite():
    config = _small_config(
        warmup_days=0,
        evaluation_days=2,
        disruption_mode="both",
    )
    from mdp.actions import encode_action

    action_no_exp = encode_action(ell=0, expedite=0, allocation=1)
    action_exp = encode_action(ell=0, expedite=1, allocation=1)

    def total_reward(action_id: int, seed: int) -> float:
        env = RetailMDPEnv(config, seed=seed)
        env.reset()
        total = 0.0
        while not env.done:
            _, r, _, _ = env.step(action_id)
            total += r
        return total

    r_no = total_reward(action_no_exp, seed=10)
    r_exp = total_reward(action_exp, seed=10)
    # Expedite costs money; profit should be lower (or equal if no orders)
    assert r_exp <= r_no


def test_run_morning_before_run_evening_required():
    config = _small_config(warmup_days=0, evaluation_days=1)
    engine = SimulationEngine(config, seed=1)
    with pytest.raises(RuntimeError, match="run_morning"):
        engine.run_evening(action=0)


def test_run_one_day_still_works_after_split():
    config = _small_config(warmup_days=1, evaluation_days=2)
    snap = SimulationEngine(config, seed=42).run_one_day(day=2, action=0)
    assert snap["day"] == 2
    assert "central_inventory_end" in snap
    assert "expedite_cost" in snap
