from __future__ import annotations

from mdp.actions import MDPAction
from mdp.thresholds import temporary_central_thresholds, temporary_forward_thresholds
from tests.test_central_warehouse import make_central_warehouse
from tests.test_forward_warehouse import make_forward_warehouse


def test_ell0_thresholds_match_base_policy():
    fw = make_forward_warehouse()
    cw = make_central_warehouse()
    action = MDPAction(ell=0, expedite=0, allocation=0)

    s_f, S_f = temporary_forward_thresholds(fw, action)
    s_c, S_c = temporary_central_thresholds(cw, action)

    assert s_f == fw.reorder_points
    assert S_f == fw.order_up_to_levels
    assert s_c == cw.reorder_points
    assert S_c == cw.order_up_to_levels


def test_ell2_raises_reorder_points():
    fw = make_forward_warehouse()
    cw = make_central_warehouse()
    action = MDPAction(ell=2, expedite=0, allocation=0)

    s_f, S_f = temporary_forward_thresholds(fw, action)
    s_c, S_c = temporary_central_thresholds(cw, action)

    assert S_f == fw.order_up_to_levels
    assert all(
        raised >= base for raised, base in zip(s_f, fw.reorder_points)
    )
    assert all(
        raised >= base for raised, base in zip(s_c, cw.reorder_points)
    )
    assert all(
        raised >= base for raised, base in zip(S_c, cw.order_up_to_levels)
    )
