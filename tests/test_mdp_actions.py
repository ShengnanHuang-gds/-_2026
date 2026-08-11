from __future__ import annotations

import pytest

from mdp.actions import (
    ALLOCATION_RULE_NAMES,
    BUFFER_MULTIPLIERS,
    NUM_ACTIONS,
    MDPAction,
    all_actions,
    baseline_action,
    decode_action,
    encode_action,
)


def test_num_actions_is_18():
    assert NUM_ACTIONS == 18
    assert len(all_actions()) == 18


def test_fine_action_space_size_and_roundtrip():
    from mdp.actions import (
        ACTION_SPACE_FINE,
        NUM_ACTIONS_FINE,
        FOCUS_TIGHTEST_SKU,
        all_actions as all_acts,
        encode_action,
        decode_action,
        num_actions_for,
    )

    assert num_actions_for("fine") == NUM_ACTIONS_FINE == 108
    assert len(all_acts(space="fine")) == 108
    for aid in range(NUM_ACTIONS_FINE):
        a = decode_action(aid, space="fine")
        assert a.action_id_for("fine") == aid
    # focus=none embeds coarse (2,0,0) as a distinct fine id
    fine_id = encode_action(2, 0, 0, focus=0, space=ACTION_SPACE_FINE)
    a = decode_action(fine_id, space=ACTION_SPACE_FINE)
    assert a.as_full_tuple() == (2, 0, 0, 0)
    boosted = encode_action(
        2, 0, 0, focus=FOCUS_TIGHTEST_SKU, space=ACTION_SPACE_FINE
    )
    assert boosted != fine_id


def test_resolve_tightest_sku_bumps_only_one_product():
    from mdp.actions import (
        FOCUS_TIGHTEST_SKU,
        MDPAction,
        resolve_product_buffer_levels,
    )

    action = MDPAction(ell=1, expedite=0, allocation=1, focus=FOCUS_TIGHTEST_SKU)
    cw = [100, 10, 100, 100, 100]
    fw = [[10, 1, 10, 10, 10] for _ in range(2)]
    mu = [[1.0, 1.0, 1.0, 1.0, 1.0] for _ in range(2)]
    cw_ell, fw_ell = resolve_product_buffer_levels(
        action,
        central_inventory=cw,
        forward_inventory=fw,
        fw_daily_demand_means=mu,
    )
    assert cw_ell[1] == 2
    assert all(cw_ell[k] == 1 for k in (0, 2, 3, 4))
    assert all(row[1] == 2 for row in fw_ell)


def test_action_id_zero_is_baseline():
    action = decode_action(0)
    assert action.as_tuple() == (0, 0, 0)
    assert action.is_baseline is True
    assert baseline_action() == action
    assert action.action_id == 0


def test_decode_encode_roundtrip_for_all_ids():
    for action_id in range(NUM_ACTIONS):
        action = decode_action(action_id)
        assert action.action_id == action_id
        assert encode_action(*action.as_tuple()) == action_id


def test_all_actions_cover_unique_triples():
    actions = all_actions()
    triples = [action.as_tuple() for action in actions]
    assert len(set(triples)) == 18
    assert triples[0] == (0, 0, 0)
    assert (2, 1, 2) in triples


def test_buffer_multipliers_match_pdf_table():
    assert BUFFER_MULTIPLIERS[0] == (1.00, 1.00, 1.00)
    assert BUFFER_MULTIPLIERS[1] == (1.10, 1.10, 1.05)
    assert BUFFER_MULTIPLIERS[2] == (1.25, 1.25, 1.10)

    mild = decode_action(encode_action(1, 0, 0))
    assert mild.eta_central_reorder == pytest.approx(1.10)
    assert mild.eta_central_order_up_to == pytest.approx(1.10)
    assert mild.eta_forward_reorder == pytest.approx(1.05)


def test_allocation_rule_names():
    assert ALLOCATION_RULE_NAMES[0] == "proportional"
    assert ALLOCATION_RULE_NAMES[1] == "lowest_dos"
    assert ALLOCATION_RULE_NAMES[2] == "value_penalty"
    assert decode_action(2).allocation_rule_name == "value_penalty"


def test_illegal_action_id_raises():
    with pytest.raises(ValueError, match="action_id must be in 0..17"):
        decode_action(-1)
    with pytest.raises(ValueError, match="action_id must be in 0..17"):
        decode_action(18)
    with pytest.raises(TypeError):
        decode_action(1.5)  # type: ignore[arg-type]


def test_illegal_components_raise():
    with pytest.raises(ValueError):
        MDPAction(ell=3, expedite=0, allocation=0)
    with pytest.raises(ValueError):
        MDPAction(ell=0, expedite=2, allocation=0)
    with pytest.raises(ValueError):
        MDPAction(ell=0, expedite=0, allocation=3)
