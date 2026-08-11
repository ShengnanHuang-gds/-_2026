from __future__ import annotations

from mdp.actions import decode_action
from mdp.env import MDPState
from mdp.policies import (
    BEST_FIXED_BY_PROFILE,
    FixedActionPolicy,
    RuleBasedPolicy,
    make_baseline_policies,
)


def _state(a: int, u: int, cw=(500, 400, 300, 200, 100)) -> MDPState:
    return MDPState(
        day=1,
        central_inventory=tuple(cw),
        forward_inventory=((10, 10, 10, 10, 10),),
        supplier_pipeline=(),
        supplier_backlog=(0, 0, 0, 0, 0),
        transport_waiting=(0, 0, 0, 0, 0),
        supplier_availability=a,
        transport_state=u,
        disruption_state=f"Z_{a}{u}",
    )


def test_fixed_action_policy_constant():
    policy = FixedActionPolicy(13)
    assert policy.select(_state(0, 0)) == 13
    assert policy.select(_state(3, 3)) == 13


def test_rule_normal_uses_ell0():
    policy = RuleBasedPolicy()
    action = decode_action(policy.select(_state(0, 0)))
    assert action.as_tuple() == (0, 0, 1)


def test_rule_mild_uses_ell1():
    policy = RuleBasedPolicy()
    for a, u in [(1, 0), (0, 1), (1, 1)]:
        action = decode_action(policy.select(_state(a, u)))
        assert action.ell == 1
        assert action.expedite == 0
        assert action.allocation == 1


def test_rule_severe_uses_ell2_no_expedite():
    policy = RuleBasedPolicy()
    for a, u in [(2, 0), (0, 2), (3, 0), (0, 3), (3, 3), (2, 2)]:
        action = decode_action(policy.select(_state(a, u)))
        assert action.ell == 2
        assert action.expedite == 0
        assert action.allocation == 1


def test_rule_u3_never_expedites_even_if_enabled():
    policy = RuleBasedPolicy(
        allow_conditional_expedite=True,
        cw_expedite_threshold=10_000,
    )
    action = decode_action(policy.select(_state(0, 3, cw=(0, 0, 0, 0, 0))))
    assert action.expedite == 0
    assert action.ell == 2


def test_rule_conditional_expedite_on_u1_low_inventory():
    policy = RuleBasedPolicy(
        allow_conditional_expedite=True,
        cw_expedite_threshold=400,
    )
    low = _state(0, 1, cw=(50, 50, 50, 50, 50))  # sum=250 < 400
    high = _state(0, 1, cw=(200, 200, 200, 200, 200))  # sum=1000
    assert decode_action(policy.select(low)).expedite == 1
    assert decode_action(policy.select(high)).expedite == 0


def test_make_baseline_policies_has_three():
    policies = make_baseline_policies("harsh_mild")
    assert len(policies) == 3
    assert policies[0].action_id == 0
    assert policies[1].action_id == BEST_FIXED_BY_PROFILE["harsh_mild"]
    assert policies[2].name == "rule_based"
