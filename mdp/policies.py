"""
Baseline policies for Phase 3 evaluation.

- FixedActionPolicy: always use one action_id
- RuleBasedPolicy: map disruption severity (and optional inventory) to (ell,e,g)

Rules are calibrated from harsh_mild / harsh_strong fixed-action grids:
  - raising ell under disruption improves profit
  - always-on expedite (e=1) hurts profit despite better fill
  - g=lowest_dos is a robust default; U3 must keep e=0
"""
from __future__ import annotations

from typing import Optional, Protocol, Union

from mdp.actions import decode_action, encode_action
from mdp.env import MDPState


class Policy(Protocol):
    def select(self, state: MDPState) -> int:
        ...


class FixedActionPolicy:
    """Always return the same action_id (in the configured action space)."""

    def __init__(
        self,
        action_id: int,
        name: str = "",
        *,
        action_space: str = "coarse",
        ell: Optional[int] = None,
        expedite: Optional[int] = None,
        allocation: Optional[int] = None,
        focus: int = 0,
    ) -> None:
        self.action_space = str(action_space).lower()
        self.focus = int(focus)
        if ell is not None and expedite is not None and allocation is not None:
            self.ell = int(ell)
            self.expedite = int(expedite)
            self.allocation = int(allocation)
            self.action_id = encode_action(
                self.ell,
                self.expedite,
                self.allocation,
                focus=self.focus,
                space=self.action_space,
            )
        else:
            decoded = decode_action(int(action_id), space=self.action_space)
            self.ell = decoded.ell
            self.expedite = decoded.expedite
            self.allocation = decoded.allocation
            self.focus = decoded.focus
            self.action_id = int(action_id)
        self.name = name or f"fixed_{self.action_id}"

    def select(self, state: MDPState) -> int:
        return self.action_id


class RuleBasedPolicy:
    """
    Risk-response rule based on A_t / U_t (and optional CW inventory).

    Default mapping (v1, no always-on expedite):

    | Condition                         | Action (ell, e, g) |
    |-----------------------------------|--------------------|
    | A0 and U0 (normal)                | (0, 0, 1)          |
    | mild: A1 or U1 (and not severe)   | (1, 0, 1)          |
    | severe: A>=2 or U>=2              | (2, 0, 1)          |
    | U3 (blocked)                      | (2, 0, 1)  e=0     |

    Under fine action space, focus is always none (recovers coarse rule).
    """

    def __init__(
        self,
        name: str = "rule_based",
        allow_conditional_expedite: bool = False,
        cw_expedite_threshold: int = 400,
        action_space: str = "coarse",
    ) -> None:
        self.name = name
        self.allow_conditional_expedite = allow_conditional_expedite
        self.cw_expedite_threshold = cw_expedite_threshold
        self.action_space = str(action_space).lower()

    def select(self, state: MDPState) -> int:
        a = int(state.supplier_availability)
        u = int(state.transport_state)

        if a == 0 and u == 0:
            ell = 0
        elif a >= 2 or u >= 2:
            ell = 2
        else:
            ell = 1

        expedite = 0
        if (
            self.allow_conditional_expedite
            and u in (1, 2)
            and a < 3
            and sum(state.central_inventory) < self.cw_expedite_threshold
        ):
            expedite = 1

        allocation = 1  # lowest_dos
        return encode_action(
            ell=ell,
            expedite=expedite,
            allocation=allocation,
            focus=0,
            space=self.action_space,
        )


# Profile-specific best fixed actions from grids (Both mode).
BEST_FIXED_BY_PROFILE = {
    "pdf": 1,  # (0,0,1) lowest_dos
    "harsh_mild": 13,  # (2,0,1)
    "harsh_strong": 12,  # (2,0,0) slightly ahead of 13
}


def make_baseline_policies(
    matrix_profile: str,
    *,
    action_space: str = "coarse",
) -> list:
    """Return [fixed_0, best_fixed, rule_based] for one matrix profile."""
    best_coarse = BEST_FIXED_BY_PROFILE.get(matrix_profile, 13)
    best = decode_action(best_coarse, space="coarse")
    space = str(action_space).lower()
    return [
        FixedActionPolicy(
            0,
            name="fixed_0_(s,S)",
            action_space=space,
            ell=0,
            expedite=0,
            allocation=0,
            focus=0,
        ),
        FixedActionPolicy(
            0,
            name=f"best_fixed_{best_coarse}",
            action_space=space,
            ell=best.ell,
            expedite=best.expedite,
            allocation=best.allocation,
            focus=0,
        ),
        RuleBasedPolicy(name="rule_based", action_space=space),
    ]


PolicyLike = Union[FixedActionPolicy, RuleBasedPolicy]
