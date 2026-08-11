"""
MDP environment wrapper for the retail simulation (Phase 3).

Episode covers evaluation_days only.
Warmup is silently burned during reset().

Observation timing (strict MDP):
  State is captured after step 3 (demand realised) and before step 4
  (replenishment decision).  The agent observes s_t, chooses a_t, receives
  r_t, then observes s_{t+1}.

Usage::

    from config.simulation_config import SimulationConfig
    from mdp.env import RetailMDPEnv

    config = SimulationConfig(
        warmup_days=30, evaluation_days=365, disruption_mode="both"
    )
    env = RetailMDPEnv(config, seed=42)
    state = env.reset()          # post-demand obs of eval day 1

    total_reward = 0.0
    while not env.done:
        action_id = 0
        state, reward, done, info = env.step(action_id)
        total_reward += reward
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from config.disruption_chain import DisruptionState
from config.simulation_config import SimulationConfig
from mdp.action_mask import action_mask, remap_illegal_action
from mdp.actions import (
    ACTION_SPACE_COARSE,
    decode_action,
    num_actions_for,
)
from simulation.simulation_engine import SimulationEngine


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MDPState:
    """
    Post-demand, pre-replenishment-decision observation.

    Captured after step 3 (demand realised) and before step 4
    (FW replenishment requests + CW allocation).
    """

    day: int                                          # eval day index (1 … T)
    central_inventory: Tuple[int, ...]                # CW on-hand after sales
    forward_inventory: Tuple[Tuple[int, ...], ...]    # FW on-hand after sales
    supplier_pipeline: Tuple[Tuple[Tuple[int, ...], int], ...]
    supplier_backlog: Tuple[int, ...]
    transport_waiting: Tuple[int, ...]
    supplier_availability: int
    transport_state: int
    disruption_state: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "day": self.day,
            "central_inventory": list(self.central_inventory),
            "forward_inventory": [list(inv) for inv in self.forward_inventory],
            "supplier_pipeline": [
                {"quantities": list(q), "remaining_lead_time": lt}
                for q, lt in self.supplier_pipeline
            ],
            "supplier_backlog": list(self.supplier_backlog),
            "transport_waiting": list(self.transport_waiting),
            "supplier_availability": self.supplier_availability,
            "transport_state": self.transport_state,
            "disruption_state": self.disruption_state,
        }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class RetailMDPEnv:
    """
    Gym-style MDP wrapper around SimulationEngine.

    reset()  – burn warmup silently, run morning of eval day 1, return s_1
    step(a)  – run evening with action a, reward r_t; if not done, run
               morning of next eval day and return s_{t+1}
    done     – True after evaluation_days steps
    reward   – daily_profit (revenue - holding - penalty - expedite_cost)
    """

    def __init__(
        self,
        config: SimulationConfig,
        seed: Optional[int] = None,
        reward_scale: float = 1.0,
        enforce_action_mask: bool = True,
        forbid_expedite: bool = False,
        action_space: str = ACTION_SPACE_COARSE,
        fixed_disruption_state: Optional[DisruptionState] = None,
        rule_buffer_mask: bool = False,
        dos_threshold: float = 1.0,
    ) -> None:
        self.config = config
        self.seed: int = seed if seed is not None else config.random_seed_base
        # Training tip: use reward_scale=10000.0 so daily rewards are O(1).
        # Evaluation/reporting should still use PerformanceTracker raw profit.
        if reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        self.reward_scale = float(reward_scale)
        self.enforce_action_mask = bool(enforce_action_mask)
        # When True, mask all e=1 actions (RL ablation / curriculum).
        self.forbid_expedite = bool(forbid_expedite)
        self.action_space = str(action_space).lower()
        self.num_actions = num_actions_for(self.action_space)
        self.fixed_disruption_state = fixed_disruption_state
        # Business shield: severe / low-DOS → only ell=2 is legal.
        self.rule_buffer_mask = bool(rule_buffer_mask)
        self.dos_threshold = float(dos_threshold)
        self._fw_daily_demand_means: Optional[list] = None

        self.engine: Optional[SimulationEngine] = None
        self._eval_day: int = 0
        self.done: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> MDPState:
        """
        Fresh engine → silent warmup → morning of eval day 1 → return s_1.

        Same seed always yields the same initial state.
        """
        if seed is not None:
            self.seed = seed

        self.engine = SimulationEngine(
            self.config,
            seed=self.seed,
            action_space=self.action_space,
            fixed_disruption_state=self.fixed_disruption_state,
        )
        self._fw_daily_demand_means = [
            list(fw.get_daily_demand_means())
            for fw in self.engine.forward_warehouses
        ]
        self._eval_day = 0
        self.done = False

        # Silent warmup: full days with baseline (action=None) policy
        for d in range(1, self.config.warmup_days + 1):
            self.engine.run_one_day(d, action=None)

        # Morning of first evaluation day → post-demand observation s_1
        first_eval_sim_day = self.config.warmup_days + 1
        self.engine.run_morning(first_eval_sim_day)
        self._eval_day = 1

        return self.get_state()

    def get_state(self) -> MDPState:
        """Return current post-demand observation."""
        if self.engine is None:
            raise RuntimeError("call reset() before get_state()")
        return self._build_state_from_obs(self.engine.post_demand_obs)

    def _mask_kwargs(self) -> Dict[str, Any]:
        return {
            "forbid_expedite": self.forbid_expedite,
            "space": self.action_space,
            "rule_buffer_mask": self.rule_buffer_mask,
            "fw_daily_demand_means": self._fw_daily_demand_means,
            "dos_threshold": self.dos_threshold,
        }

    def get_action_mask(self) -> Any:
        """Boolean mask over actions for the current post-demand state."""
        return action_mask(self.get_state(), **self._mask_kwargs())

    def step(
        self, action_id: int
    ) -> Tuple[MDPState, float, bool, Dict[str, Any]]:
        """
        Apply action_id to the pending evening of the current eval day.

        Returns (next_state, reward, done, info).

        reward is raw_daily_profit / reward_scale.
        info includes raw_reward, action_mask, and mask remapping metadata.
        """
        if self.done:
            raise RuntimeError("episode is over; call reset() first")
        if self.engine is None:
            raise RuntimeError("call reset() before step()")
        if not isinstance(action_id, int) or isinstance(action_id, bool):
            raise TypeError(f"action_id must be int, got {type(action_id).__name__}")
        if not (0 <= action_id < self.num_actions):
            raise ValueError(
                f"action_id must be in 0..{self.num_actions - 1}, got {action_id}"
            )

        current_state = self.get_state()
        mask_kwargs = self._mask_kwargs()
        mask = action_mask(current_state, **mask_kwargs)
        requested_action_id = action_id
        remapped = False
        if self.enforce_action_mask and not bool(mask[action_id]):
            action_id = remap_illegal_action(
                action_id, current_state, **mask_kwargs
            )
            remapped = action_id != requested_action_id
            mask = action_mask(current_state, **mask_kwargs)

        # Evening of current eval day (steps 4–7) with agent action
        snapshot = self.engine.run_evening(action_id)
        raw_reward = self.compute_reward(snapshot)
        reward = raw_reward / self.reward_scale

        done = self._eval_day >= self.config.evaluation_days
        self.done = done

        if not done:
            # Advance to next eval day: run morning (steps 1–3)
            self._eval_day += 1
            next_sim_day = self.config.warmup_days + self._eval_day
            self.engine.run_morning(next_sim_day)
            next_state = self.get_state()
        else:
            # Terminal: build end-of-day state from evening snapshot
            next_state = self._build_terminal_state(snapshot)

        info: Dict[str, Any] = {
            "eval_day": self._eval_day,
            "sim_day": snapshot["day"],
            "action_id": action_id,
            "action_id_requested": requested_action_id,
            "action": decode_action(
                action_id, space=self.action_space
            ).as_full_tuple(),
            "action_masked": remapped,
            "action_mask": mask,
            "expedite_cost": snapshot.get("expedite_cost", 0.0),
            "disruption_state": snapshot.get("disruption_state", ""),
            "raw_reward": raw_reward,
            "reward_scale": self.reward_scale,
            "action_space": self.action_space,
        }

        return next_state, reward, done, info

    def compute_reward(self, snapshot: dict) -> float:
        """
        Daily reward = revenue - holding_cost - lost_sales_penalty - expedite_cost.

        Matches PerformanceTracker formula.  Warmup days return 0.
        """
        if snapshot.get("is_warmup", False):
            return 0.0

        products = self.engine.products
        fws = self.engine.forward_warehouses

        cw_begin = snapshot["central_inventory_begin"]
        cw_end = snapshot["central_inventory_end"]
        fw_begin = snapshot["forward_inventory_begin"]
        fw_end = snapshot["forward_inventory_end"]

        revenue = 0.0
        penalty = 0.0
        holding = 0.0

        for fw in fws:
            fw_id = fw.forward_warehouse_id
            for k, product in enumerate(products):
                revenue += product.selling_price * fw.today_sales[k]
                penalty += product.lost_sales_penalty * fw.today_lost_sales[k]

        for k, product in enumerate(products):
            avg_cw = (cw_begin[k] + cw_end[k]) / 2.0
            holding += product.central_holding_cost * avg_cw
            for fw in fws:
                fw_id = fw.forward_warehouse_id
                avg_fw = (fw_begin[fw_id][k] + fw_end[fw_id][k]) / 2.0
                holding += product.forward_holding_cost * avg_fw

        expedite_cost = float(snapshot.get("expedite_cost", 0.0) or 0.0)
        return revenue - holding - penalty - expedite_cost

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_state_from_obs(self, obs: Optional[dict]) -> MDPState:
        if obs is None:
            raise RuntimeError(
                "post_demand_obs is not set; run_morning() has not been called"
            )

        fws = self.engine.forward_warehouses
        fw_ids = [fw.forward_warehouse_id for fw in fws]

        return MDPState(
            day=self._eval_day,
            central_inventory=tuple(obs["central_inventory"]),
            forward_inventory=tuple(
                tuple(obs["forward_inventory"][fw_id]) for fw_id in fw_ids
            ),
            supplier_pipeline=tuple(
                (tuple(entry["quantities"]), entry["remaining_lead_time"])
                for entry in obs["supplier_pipeline"]
            ),
            supplier_backlog=tuple(obs["supplier_backlog"]),
            transport_waiting=tuple(obs["transport_waiting"]),
            supplier_availability=obs["supplier_availability"],
            transport_state=obs["transport_state"],
            disruption_state=obs["disruption_state"],
        )

    def _build_terminal_state(self, snapshot: dict) -> MDPState:
        """End-of-day state after the final evening step."""
        cw = self.engine.central_warehouse
        fws = self.engine.forward_warehouses
        fw_ids = [fw.forward_warehouse_id for fw in fws]
        z = self.engine.disruption_state

        return MDPState(
            day=self._eval_day,
            central_inventory=tuple(snapshot["central_inventory_end"]),
            forward_inventory=tuple(
                tuple(snapshot["forward_inventory_end"][fw_id])
                for fw_id in fw_ids
            ),
            supplier_pipeline=tuple(
                (tuple(order.quantities), order.remaining_lead_time)
                for order in cw.supplier_pipeline
            ),
            supplier_backlog=tuple(cw.upstream.supplier_backlog),
            transport_waiting=tuple(cw.upstream.transport_waiting),
            supplier_availability=z.supplier_availability.value,
            transport_state=z.transport_state.value,
            disruption_state=z.code(),
        )
