from __future__ import annotations

import numpy as np
import pytest

from config.simulation_config import SimulationConfig
from mdp.action_mask import (
    action_mask,
    is_action_legal,
    legal_action_ids,
    remap_illegal_action,
)
from mdp.actions import NUM_ACTIONS, decode_action, encode_action
from mdp.env import MDPState, RetailMDPEnv
from mdp.state_encoder import (
    RiskFeatureEncoder,
    StateEncoder,
    build_state_encoder_from_env,
)


def _toy_state(a: int = 0, u: int = 0, day: int = 10) -> MDPState:
    return MDPState(
        day=day,
        central_inventory=(100, 80, 60, 40, 20),
        forward_inventory=tuple(
            (5, 4, 3, 2, 1) for _ in range(10)
        ),
        supplier_pipeline=(
            ((10, 0, 0, 0, 0), 1),
            ((0, 20, 0, 0, 0), 2),
            ((0, 0, 30, 0, 0), 5),
        ),
        supplier_backlog=(1, 2, 3, 4, 5),
        transport_waiting=(5, 4, 3, 2, 1),
        supplier_availability=a,
        transport_state=u,
        disruption_state=f"Z_{a}{u}",
    )


def _toy_encoder() -> StateEncoder:
    cw_s = [100, 100, 100, 100, 100]
    fw_s = [[20, 20, 20, 20, 20] for _ in range(10)]
    return StateEncoder(cw_s, fw_s, evaluation_days=365)


def test_state_encoder_dim_and_dtype():
    encoder = _toy_encoder()
    assert encoder.dim == 94
    vec = encoder.encode(_toy_state())
    assert vec.shape == (94,)
    assert vec.dtype == np.float32


def test_state_encoder_is_deterministic():
    encoder = _toy_encoder()
    state = _toy_state(1, 2, day=50)
    a = encoder.encode(state)
    b = encoder.encode(state)
    assert np.array_equal(a, b)


def test_state_encoder_one_hots_and_day():
    encoder = _toy_encoder()
    vec = encoder.encode(_toy_state(a=2, u=3, day=365))
    # last 9 features: A(4) + U(4) + day(1)
    assert list(vec[-9:-5]) == [0, 0, 1, 0]  # A2
    assert list(vec[-5:-1]) == [0, 0, 0, 1]  # U3
    assert vec[-1] == pytest.approx(1.0)


def test_state_encoder_from_env():
    config = SimulationConfig(warmup_days=1, evaluation_days=2)
    env = RetailMDPEnv(config, seed=1)
    state = env.reset()
    encoder = build_state_encoder_from_env(env)
    vec = encoder.encode(state)
    assert vec.shape == (encoder.dim,)
    assert np.all(np.isfinite(vec))


def _toy_risk_encoder() -> RiskFeatureEncoder:
    fw_mu = [[1.0, 2.0, 3.0, 4.0, 5.0] for _ in range(10)]
    return RiskFeatureEncoder(fw_mu, evaluation_days=365)


def test_risk_encoder_dim_and_flags():
    encoder = _toy_risk_encoder()
    assert encoder.dim == 33
    vec = encoder.encode(_toy_state(a=2, u=3, day=365))
    assert vec.shape == (33,)
    assert vec.dtype == np.float32
    assert list(vec[0:4]) == [0, 0, 1, 0]  # A2
    assert list(vec[4:8]) == [0, 0, 0, 1]  # U3
    assert vec[8] == pytest.approx(1.0)  # severe
    assert vec[9] == pytest.approx(0.0)  # mild
    assert vec[10] == pytest.approx(1.0)  # u3
    assert vec[-1] == pytest.approx(1.0)
    assert np.all(np.isfinite(vec))


def test_risk_encoder_from_env():
    config = SimulationConfig(warmup_days=1, evaluation_days=2)
    env = RetailMDPEnv(config, seed=1)
    state = env.reset()
    encoder = build_state_encoder_from_env(env, encoder_type="risk")
    assert isinstance(encoder, RiskFeatureEncoder)
    assert encoder.dim == 33
    vec = encoder.encode(state)
    assert vec.shape == (33,)
    assert np.all(np.isfinite(vec))


def test_action_mask_all_legal_when_not_u3():
    for u in (0, 1, 2):
        mask = action_mask(u)
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.dtype == bool
        assert mask.all()


def test_action_mask_blocks_expedite_under_u3():
    mask = action_mask(3)
    assert mask.sum() == 9  # half of 18 have e=0
    for action_id in range(NUM_ACTIONS):
        action = decode_action(action_id)
        if action.expedite == 1:
            assert not mask[action_id]
        else:
            assert mask[action_id]


def test_forbid_expedite_masks_all_e1_even_when_not_u3():
    mask = action_mask(0, forbid_expedite=True)
    assert mask.sum() == 9
    for action_id in range(NUM_ACTIONS):
        action = decode_action(action_id)
        assert bool(mask[action_id]) == (action.expedite == 0)


def test_rule_buffer_mask_forces_ell2_on_severe():
    from mdp.action_mask import requires_high_buffer

    state = _toy_state(a=2, u=0)
    assert requires_high_buffer(state, None) is True
    mask = action_mask(
        state, forbid_expedite=True, rule_buffer_mask=True
    )
    assert mask.sum() == 3  # ell=2, e=0, g=0/1/2
    for aid in range(NUM_ACTIONS):
        if mask[aid]:
            assert decode_action(aid).ell == 2
            assert decode_action(aid).expedite == 0


def test_rule_buffer_mask_remaps_low_ell_to_ell2():
    state = _toy_state(a=3, u=0)
    low = encode_action(0, 0, 1)
    remapped = remap_illegal_action(
        low, state, forbid_expedite=True, rule_buffer_mask=True
    )
    assert decode_action(remapped).ell == 2
    assert decode_action(remapped).expedite == 0
    assert decode_action(remapped).allocation == 1


def test_rule_buffer_mask_allows_all_ell_when_healthy():
    state = _toy_state(a=0, u=0)
    # High inventory ⇒ DOS healthy; without fw means, only severity matters.
    mask = action_mask(
        state, forbid_expedite=True, rule_buffer_mask=True
    )
    assert mask.sum() == 9
    ells = {decode_action(i).ell for i in range(NUM_ACTIONS) if mask[i]}
    assert ells == {0, 1, 2}


def test_env_forbid_expedite_remaps_and_exposes_mask():
    config = SimulationConfig(
        warmup_days=0,
        evaluation_days=1,
        disruption_mode="none",
    )
    env = RetailMDPEnv(
        config,
        seed=7,
        reward_scale=10000.0,
        enforce_action_mask=True,
        forbid_expedite=True,
    )
    env.reset()
    mask = env.get_action_mask()
    assert mask.sum() == 9
    illegal = encode_action(1, 1, 0)
    assert not mask[illegal]
    _state, _reward, done, info = env.step(illegal)
    assert done is True
    assert info["action_masked"] is True
    assert info["action_id"] == encode_action(1, 0, 0)


def test_remap_illegal_action_drops_expedite():
    illegal = encode_action(ell=2, expedite=1, allocation=1)  # 16
    remapped = remap_illegal_action(illegal, 3)
    assert remapped == encode_action(ell=2, expedite=0, allocation=1)  # 13
    assert is_action_legal(remapped, 3)
    assert set(legal_action_ids(3)) == {
        encode_action(ell=e, expedite=0, allocation=g)
        for e in range(3)
        for g in range(3)
    }


def test_env_remaps_u3_expedite_and_scales_reward():
    from config.disruption_chain import (
        DisruptionState,
        SupplierAvailability,
        TransportState,
    )

    config = SimulationConfig(
        warmup_days=0,
        evaluation_days=1,
        disruption_mode="none",
    )
    env = RetailMDPEnv(config, seed=7, reward_scale=10000.0, enforce_action_mask=True)
    env.reset()
    # Force U3 on the pending decision state
    env.engine.disruption_state = DisruptionState(
        supplier_availability=SupplierAvailability.A0,
        transport_state=TransportState.U3,
    )
    # Also overwrite obs transport fields used by get_state()
    env.engine.post_demand_obs["transport_state"] = 3
    env.engine.post_demand_obs["disruption_state"] = "Z_03"
    env.engine.post_demand_obs["supplier_availability"] = 0

    illegal = encode_action(1, 1, 0)
    _state, reward, done, info = env.step(illegal)
    assert done is True
    assert info["action_masked"] is True
    assert info["action_id_requested"] == illegal
    assert info["action_id"] == encode_action(1, 0, 0)
    assert info["raw_reward"] == pytest.approx(reward * 10000.0)
    assert not info["action_mask"][illegal]
