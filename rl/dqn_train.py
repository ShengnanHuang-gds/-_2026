"""
Shared DQN / Double DQN training and evaluation utilities.

Used by smoke and full-horizon trainers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from config.disruption_chain import (
    DisruptionState,
    SupplierAvailability,
    TransportState,
)
from config.simulation_config import SimulationConfig
from mdp.actions import VALID_ACTION_SPACES, decode_action, num_actions_for
from mdp.env import MDPState, RetailMDPEnv
from mdp.policies import RuleBasedPolicy
from mdp.state_encoder import (
    VALID_ENCODER_TYPES,
    EncoderLike,
    build_state_encoder_from_env,
)
from rl.dqn import (
    QNetwork,
    build_q_network,
    states_to_tensor,
    uses_double_q,
)
from rl.replay_buffer import ReplayBuffer

VALID_ALGORITHMS = frozenset({"dqn", "ddqn", "dueling", "d3qn"})


@dataclass
class DQNTrainConfig:
    disruption_matrix_profile: str = "harsh_mild"
    disruption_mode: str = "both"
    # Optional locked disruption Z=(A,U); overrides Markov sampling when set.
    fixed_supplier_availability: Optional[int] = None
    fixed_transport_state: Optional[int] = None
    warmup_days: int = 30
    # Horizon used while collecting experience / updating.
    evaluation_days: int = 365
    num_forward_warehouses: int = 10
    demand_intensity_scale: float = 1.2
    reward_scale: float = 10000.0
    enforce_action_mask: bool = True
    # Hard-ban all e=1 actions during train/eval (9-action subspace).
    forbid_expedite: bool = True
    # Business shield: severe / low-DOS days only allow ell=2.
    rule_buffer_mask: bool = False
    dos_threshold: float = 1.0

    # State featurizer: "raw" (94-d) or "risk" (~33-d DOS/shortage features).
    state_encoder: str = "raw"

    # Action space: "coarse" (ell,e,g)=18 or "fine" (ell,focus,e,g)=108.
    action_space: str = "coarse"

    # "dqn" | "ddqn" | "dueling" | "d3qn" (dueling + double).
    algorithm: str = "ddqn"

    episodes: int = 50
    buffer_capacity: int = 100_000
    batch_size: int = 64
    start_learning_after: int = 1_000
    train_every: int = 1
    target_update_every: int = 500
    gamma: float = 0.99
    lr: float = 3e-4

    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 30_000

    seed: int = 42
    device: str = "cpu"
    log_every_episode: int = 1

    # Periodic greedy full-horizon evaluation (eps=0).
    eval_every_episodes: int = 5
    eval_evaluation_days: int = 365
    eval_warmup_days: int = 30
    eval_num_seeds: int = 3
    eval_seed_offset: int = 10_000

    checkpoint_path: Optional[str] = None

    # Rule warm-start: fill replay with RuleBasedPolicy, optional BC, then RL.
    rule_warmstart_episodes: int = 0
    rule_bc_updates: int = 0
    rule_bc_batch_size: int = 64
    rule_bc_lr: float = 1e-3
    # TD updates on rule buffer before RL; set 0 for "prefill-only" ablation.
    rule_td_pre_updates: int = 500

    def __post_init__(self) -> None:
        algo = str(self.algorithm).lower()
        if algo not in VALID_ALGORITHMS:
            raise ValueError(
                f"unknown algorithm {self.algorithm!r}; "
                f"expected one of {sorted(VALID_ALGORITHMS)}"
            )
        self.algorithm = algo
        enc = str(self.state_encoder).lower()
        if enc not in VALID_ENCODER_TYPES:
            raise ValueError(
                f"unknown state_encoder {self.state_encoder!r}; "
                f"expected one of {sorted(VALID_ENCODER_TYPES)}"
            )
        self.state_encoder = enc
        space = str(self.action_space).lower()
        if space not in VALID_ACTION_SPACES:
            raise ValueError(
                f"unknown action_space {self.action_space!r}; "
                f"expected one of {sorted(VALID_ACTION_SPACES)}"
            )
        self.action_space = space
        if self.rule_warmstart_episodes < 0:
            raise ValueError("rule_warmstart_episodes must be >= 0")
        if self.rule_bc_updates < 0:
            raise ValueError("rule_bc_updates must be >= 0")
        if self.rule_td_pre_updates < 0:
            raise ValueError("rule_td_pre_updates must be >= 0")
        for name, val in (
            ("fixed_supplier_availability", self.fixed_supplier_availability),
            ("fixed_transport_state", self.fixed_transport_state),
        ):
            if val is not None and not (0 <= int(val) <= 3):
                raise ValueError(f"{name} must be in 0..3 or None")


def fixed_disruption_from_cfg(cfg: DQNTrainConfig) -> Optional[DisruptionState]:
    if (
        cfg.fixed_supplier_availability is None
        and cfg.fixed_transport_state is None
    ):
        return None
    a = (
        0
        if cfg.fixed_supplier_availability is None
        else int(cfg.fixed_supplier_availability)
    )
    u = (
        0
        if cfg.fixed_transport_state is None
        else int(cfg.fixed_transport_state)
    )
    return DisruptionState(
        supplier_availability=SupplierAvailability(a),
        transport_state=TransportState(u),
    )


@dataclass
class EvalResult:
    episode: int
    mean_raw_profit: float
    std_raw_profit: float
    profits: List[float]
    mean_scaled_return: float


def build_env(
    cfg: DQNTrainConfig,
    seed: int,
    *,
    evaluation_days: Optional[int] = None,
    warmup_days: Optional[int] = None,
) -> RetailMDPEnv:
    sim = SimulationConfig(
        warmup_days=cfg.warmup_days if warmup_days is None else warmup_days,
        evaluation_days=(
            cfg.evaluation_days if evaluation_days is None else evaluation_days
        ),
        num_replications=1,
        num_forward_warehouses=cfg.num_forward_warehouses,
        disruption_mode=cfg.disruption_mode,
        disruption_matrix_profile=cfg.disruption_matrix_profile,
        demand_intensity_scale=cfg.demand_intensity_scale,
        random_seed_base=seed,
    )
    return RetailMDPEnv(
        sim,
        seed=seed,
        reward_scale=cfg.reward_scale,
        enforce_action_mask=cfg.enforce_action_mask,
        forbid_expedite=cfg.forbid_expedite,
        action_space=cfg.action_space,
        fixed_disruption_state=fixed_disruption_from_cfg(cfg),
        rule_buffer_mask=cfg.rule_buffer_mask,
        dos_threshold=cfg.dos_threshold,
    )


def epsilon_by_step(cfg: DQNTrainConfig, global_step: int) -> float:
    if cfg.eps_decay_steps <= 0:
        return cfg.eps_end
    t = min(1.0, global_step / float(cfg.eps_decay_steps))
    return cfg.eps_start + t * (cfg.eps_end - cfg.eps_start)


def select_masked_epsilon_greedy(
    online: QNetwork,
    state_vec: np.ndarray,
    mask: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        raise RuntimeError("no legal actions under current mask")
    if rng.random() < epsilon:
        return int(rng.choice(legal))
    state_t = states_to_tensor(state_vec, device=device)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
    return online.select_greedy(state_t, action_mask=mask_t)


def compute_td_targets(
    online: QNetwork,
    target: QNetwork,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_masks: torch.Tensor,
    gamma: float,
    algorithm: str = "ddqn",
) -> torch.Tensor:
    """
    Bootstrap targets for value-based algorithms (mask-aware).

    DQN / Dueling:  y = r + γ (1-d) max_a' Q_target(s', a')
    DDQN / D3QN:    a* = argmax Q_online(s'); y = r + γ (1-d) Q_target(s', a*)
    """
    algo = algorithm.lower()
    if algo not in VALID_ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}")

    if uses_double_q(algo):
        online_next = online(next_states)
        online_next = online_next.masked_fill(~next_masks, float("-inf"))
        next_actions = online_next.argmax(dim=1)
        target_next = target(next_states)
        next_values = target_next.gather(1, next_actions.unsqueeze(1)).squeeze(1)
    else:
        target_next = target(next_states)
        target_next = target_next.masked_fill(~next_masks, float("-inf"))
        next_values, _ = target_next.max(dim=1)

    next_values = torch.where(
        torch.isfinite(next_values), next_values, torch.zeros_like(next_values)
    )
    return rewards + gamma * (1.0 - dones) * next_values


def dqn_update(
    online: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    algorithm: str = "ddqn",
) -> float:
    states, actions, rewards, next_states, dones, next_masks = buffer.sample_tensors(
        batch_size, device=device
    )
    q_all = online(states)
    q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        targets = compute_td_targets(
            online,
            target,
            next_states,
            rewards,
            dones,
            next_masks,
            gamma,
            algorithm=algorithm,
        )

    loss = F.mse_loss(q_sa, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def hard_update(target: QNetwork, online: QNetwork) -> None:
    target.load_state_dict(online.state_dict())


def fill_buffer_with_rule_policy(
    cfg: DQNTrainConfig,
    buffer: ReplayBuffer,
    *,
    episodes: int,
    seed_base: int,
) -> dict:
    """
    Roll out RuleBasedPolicy for `episodes` and store transitions.

    Teaches the buffer the mapping:
      normal → (0,0,1), mild → (1,0,1), severe → (2,0,1), e=0.
    """
    if episodes <= 0:
        return {"episodes": 0, "transitions": 0, "raw_profits": []}

    policy = RuleBasedPolicy(
        name="rule_warmstart", action_space=cfg.action_space
    )
    raw_profits: List[float] = []
    transitions = 0

    for ep in range(episodes):
        seed = seed_base + ep
        env = build_env(cfg, seed=seed)
        state = env.reset(seed=seed)
        encoder = build_state_encoder_from_env(
            env, encoder_type=cfg.state_encoder
        )
        state_vec = encoder.encode(state)
        ep_raw = 0.0

        while not env.done:
            action = policy.select(state)
            # Respect mask (e.g. forbid_expedite); rule already uses e=0.
            mask = env.get_action_mask()
            if not bool(mask[action]):
                # Should not happen for default rule under forbid_e1
                legal = np.flatnonzero(mask)
                action = int(legal[0])

            next_state, reward, done, info = env.step(action)
            next_vec = encoder.encode(next_state)
            next_mask = env.get_action_mask()
            buffer.add(state_vec, action, reward, next_vec, done, next_mask)
            transitions += 1
            ep_raw += float(info["raw_reward"])
            state = next_state
            state_vec = next_vec

        raw_profits.append(ep_raw)
        print(
            f"[warmstart rule ep {ep + 1:03d}/{episodes}] "
            f"raw_profit={ep_raw:.1f} buf={len(buffer)}",
            flush=True,
        )

    return {
        "episodes": episodes,
        "transitions": transitions,
        "raw_profits": raw_profits,
        "mean_raw_profit": float(np.mean(raw_profits)) if raw_profits else 0.0,
    }


def behavioral_clone_pretrain(
    online: QNetwork,
    buffer: ReplayBuffer,
    *,
    updates: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> List[float]:
    """Supervised CE on (s, a_rule) samples already in the buffer."""
    if updates <= 0:
        return []
    if len(buffer) < batch_size:
        raise ValueError(
            f"need at least batch_size={batch_size} rule transitions for BC, "
            f"got {len(buffer)}"
        )

    opt = torch.optim.Adam(online.parameters(), lr=lr)
    losses: List[float] = []
    online.train()
    for i in range(updates):
        states, actions, _, _, _, _ = buffer.sample_tensors(batch_size, device=device)
        logits = online(states)
        loss = F.cross_entropy(logits, actions)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
        opt.step()
        losses.append(float(loss.item()))
        if (i + 1) % max(1, updates // 5) == 0 or i == 0:
            print(
                f"[warmstart BC {i + 1:04d}/{updates}] loss={losses[-1]:.4f}",
                flush=True,
            )
    return losses


@torch.no_grad()
def evaluate_greedy(
    online: QNetwork,
    cfg: DQNTrainConfig,
    *,
    episode: int,
    device: torch.device,
) -> EvalResult:
    """
    Greedy (ε=0) rollouts on full-horizon eval env over several seeds.
    Uses PerformanceTracker total profit when available; else sums raw_reward.
    """
    profits: List[float] = []
    scaled_returns: List[float] = []

    for i in range(cfg.eval_num_seeds):
        seed = cfg.seed + cfg.eval_seed_offset + i
        env = build_env(
            cfg,
            seed=seed,
            evaluation_days=cfg.eval_evaluation_days,
            warmup_days=cfg.eval_warmup_days,
        )
        state = env.reset(seed=seed)
        encoder = build_state_encoder_from_env(
            env, encoder_type=cfg.state_encoder
        )
        state_vec = encoder.encode(state)
        ep_scaled = 0.0
        ep_raw = 0.0

        while not env.done:
            mask = env.get_action_mask()
            state_t = states_to_tensor(state_vec, device=device)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
            action = online.select_greedy(state_t, action_mask=mask_t)
            next_state, reward, _done, info = env.step(action)
            ep_scaled += reward
            ep_raw += float(info["raw_reward"])
            state = next_state
            state_vec = encoder.encode(state)

        # Prefer tracker summary (matches baseline logs).
        tracker_profit = float(env.engine.performance_tracker.summarize()["total_profit"])
        profits.append(tracker_profit)
        scaled_returns.append(ep_scaled)
        # Sanity: raw sum should match tracker closely
        _ = ep_raw

    arr = np.asarray(profits, dtype=np.float64)
    return EvalResult(
        episode=episode,
        mean_raw_profit=float(arr.mean()),
        std_raw_profit=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        profits=profits,
        mean_scaled_return=float(np.mean(scaled_returns)),
    )


def run_dqn_training(cfg: DQNTrainConfig) -> Dict[str, object]:
    """
    Full DQN training loop with optional periodic full-horizon greedy eval.
    """
    device = torch.device(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = build_env(cfg, seed=cfg.seed)
    state: MDPState = env.reset(seed=cfg.seed)
    encoder: EncoderLike = build_state_encoder_from_env(
        env, encoder_type=cfg.state_encoder
    )
    print(
        f"[train] state_encoder={cfg.state_encoder} dim={encoder.dim} "
        f"action_space={cfg.action_space} n_actions={num_actions_for(cfg.action_space)} "
        f"algo={cfg.algorithm} forbid_expedite={cfg.forbid_expedite}",
        flush=True,
    )

    n_actions = num_actions_for(cfg.action_space)
    online = build_q_network(
        state_dim=encoder.dim,
        num_actions=n_actions,
        device=device,
        algorithm=cfg.algorithm,
    )
    target = build_q_network(
        state_dim=encoder.dim,
        num_actions=n_actions,
        device=device,
        algorithm=cfg.algorithm,
    )
    hard_update(target, online)
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(
        capacity=cfg.buffer_capacity,
        state_dim=encoder.dim,
        num_actions=n_actions,
        seed=cfg.seed,
    )

    episode_returns: List[float] = []
    episode_raw_profits: List[float] = []
    episode_losses: List[float] = []
    train_e1_ratios: List[float] = []
    train_e1_legal_ratios: List[float] = []
    eval_history: List[EvalResult] = []
    global_step = 0
    updates = 0
    best_eval_profit = float("-inf")
    best_state_dict = None
    warmstart_info: Dict[str, object] = {
        "episodes": 0,
        "transitions": 0,
        "bc_losses": [],
    }

    t0 = time.time()

    # --- Rule warm-start (optional) ---
    if cfg.rule_warmstart_episodes > 0:
        print(
            f"[warmstart] collecting {cfg.rule_warmstart_episodes} rule episodes "
            f"(forbid_expedite={cfg.forbid_expedite}) ...",
            flush=True,
        )
        fill_stats = fill_buffer_with_rule_policy(
            cfg,
            buffer,
            episodes=cfg.rule_warmstart_episodes,
            seed_base=cfg.seed + 50_000,
        )
        warmstart_info["episodes"] = fill_stats["episodes"]
        warmstart_info["transitions"] = fill_stats["transitions"]
        warmstart_info["rule_mean_raw_profit"] = fill_stats["mean_raw_profit"]

        if cfg.rule_bc_updates > 0:
            print(
                f"[warmstart] behavioral cloning {cfg.rule_bc_updates} updates ...",
                flush=True,
            )
            bc_losses = behavioral_clone_pretrain(
                online,
                buffer,
                updates=cfg.rule_bc_updates,
                batch_size=min(cfg.rule_bc_batch_size, len(buffer)),
                lr=cfg.rule_bc_lr,
                device=device,
            )
            warmstart_info["bc_losses"] = bc_losses
            hard_update(target, online)

        # Optional: a few TD updates on rule data before RL exploration
        td_pre = min(int(cfg.rule_td_pre_updates), len(buffer))
        if td_pre > 0 and len(buffer) >= cfg.batch_size:
            for _ in range(td_pre):
                dqn_update(
                    online,
                    target,
                    optimizer,
                    buffer,
                    cfg.batch_size,
                    cfg.gamma,
                    device,
                    algorithm=cfg.algorithm,
                )
                updates += 1
                if updates % cfg.target_update_every == 0:
                    hard_update(target, online)
            warmstart_info["td_pre_updates"] = td_pre
        else:
            warmstart_info["td_pre_updates"] = 0

        online.eval()
        pre_eval = evaluate_greedy(online, cfg, episode=0, device=device)
        online.train()
        warmstart_info["pre_rl_eval_profit"] = pre_eval.mean_raw_profit
        print(
            f"[warmstart] pre-RL greedy eval profit="
            f"{pre_eval.mean_raw_profit:.1f} ± {pre_eval.std_raw_profit:.1f}",
            flush=True,
        )
        if pre_eval.mean_raw_profit > best_eval_profit:
            best_eval_profit = pre_eval.mean_raw_profit
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in online.state_dict().items()
            }

        # Refresh train env after warmstart rollouts
        state = env.reset(seed=cfg.seed)
        encoder = build_state_encoder_from_env(
            env, encoder_type=cfg.state_encoder
        )

    for ep in range(1, cfg.episodes + 1):
        if ep > 1:
            state = env.reset(seed=cfg.seed + ep - 1)
            encoder = build_state_encoder_from_env(
                env, encoder_type=cfg.state_encoder
            )

        state_vec = encoder.encode(state)
        ep_return = 0.0
        ep_raw = 0.0
        ep_loss_sum = 0.0
        ep_loss_n = 0
        steps = 0
        ep_e1 = 0
        ep_e1_legal = 0

        while not env.done:
            mask = env.get_action_mask()
            eps = epsilon_by_step(cfg, global_step)
            action = select_masked_epsilon_greedy(
                online, state_vec, mask, eps, rng, device
            )
            next_state, reward, done, info = env.step(action)
            next_vec = encoder.encode(next_state)
            next_mask = env.get_action_mask()
            buffer.add(state_vec, action, reward, next_vec, done, next_mask)

            if decode_action(action, space=cfg.action_space).expedite == 1:
                ep_e1 += 1
            for aid in range(len(mask)):
                if mask[aid] and decode_action(
                    aid, space=cfg.action_space
                ).expedite == 1:
                    ep_e1_legal += 1
                    break

            ep_return += reward
            ep_raw += float(info["raw_reward"])
            state = next_state
            state_vec = next_vec
            global_step += 1
            steps += 1

            if (
                len(buffer) >= max(cfg.start_learning_after, cfg.batch_size)
                and global_step % cfg.train_every == 0
            ):
                loss = dqn_update(
                    online,
                    target,
                    optimizer,
                    buffer,
                    cfg.batch_size,
                    cfg.gamma,
                    device,
                    algorithm=cfg.algorithm,
                )
                updates += 1
                ep_loss_sum += loss
                ep_loss_n += 1
                if updates % cfg.target_update_every == 0:
                    hard_update(target, online)

        mean_loss = ep_loss_sum / ep_loss_n if ep_loss_n else float("nan")
        episode_returns.append(ep_return)
        episode_raw_profits.append(ep_raw)
        episode_losses.append(mean_loss)
        e1_ratio = ep_e1 / steps if steps else 0.0
        e1_legal_ratio = ep_e1_legal / steps if steps else 0.0
        train_e1_ratios.append(e1_ratio)
        train_e1_legal_ratios.append(e1_legal_ratio)

        if ep % cfg.log_every_episode == 0:
            print(
                f"[train ep {ep:03d}/{cfg.episodes}] "
                f"algo={cfg.algorithm} "
                f"steps={steps} return={ep_return:.3f} "
                f"raw_profit={ep_raw:.1f} "
                f"loss={mean_loss:.5f} "
                f"eps={epsilon_by_step(cfg, global_step):.3f} "
                f"sample_e1={e1_ratio:.1%} "
                f"e1_legal_days={e1_legal_ratio:.1%} "
                f"buf={len(buffer)} updates={updates}",
                flush=True,
            )

        do_eval = (
            cfg.eval_every_episodes > 0
            and (ep % cfg.eval_every_episodes == 0 or ep == cfg.episodes)
        )
        if do_eval:
            online.eval()
            result = evaluate_greedy(online, cfg, episode=ep, device=device)
            online.train()
            eval_history.append(result)
            print(
                f"[eval  ep {ep:03d}] "
                f"greedy_profit={result.mean_raw_profit:.1f} "
                f"± {result.std_raw_profit:.1f} "
                f"(n={len(result.profits)}, "
                f"days={cfg.eval_evaluation_days})",
                flush=True,
            )
            if result.mean_raw_profit > best_eval_profit:
                best_eval_profit = result.mean_raw_profit
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in online.state_dict().items()
                }

    if best_state_dict is not None and cfg.checkpoint_path:
        path = cfg.checkpoint_path
        torch.save(
            {
                "model_state_dict": best_state_dict,
                "best_eval_profit": best_eval_profit,
                "config": cfg.__dict__,
            },
            path,
        )
        print(f"saved best checkpoint → {path} (profit={best_eval_profit:.1f})", flush=True)

    elapsed = time.time() - t0
    mean_train_e1 = float(np.mean(train_e1_ratios)) if train_e1_ratios else 0.0
    mean_e1_legal = (
        float(np.mean(train_e1_legal_ratios)) if train_e1_legal_ratios else 0.0
    )
    print(
        f"[train done] algo={cfg.algorithm} forbid_e1={cfg.forbid_expedite} "
        f"mean_sample_e1={mean_train_e1:.1%} "
        f"mean_e1_legal_days={mean_e1_legal:.1%}",
        flush=True,
    )
    return {
        "episodes": cfg.episodes,
        "global_steps": global_step,
        "updates": updates,
        "elapsed_sec": elapsed,
        "episode_returns": episode_returns,
        "episode_raw_profits": episode_raw_profits,
        "episode_losses": episode_losses,
        "train_e1_ratios": train_e1_ratios,
        "train_e1_legal_ratios": train_e1_legal_ratios,
        "mean_train_e1_ratio": mean_train_e1,
        "mean_train_e1_legal_ratio": mean_e1_legal,
        "eval_history": eval_history,
        "best_eval_profit": best_eval_profit if best_state_dict is not None else None,
        "final_eps": epsilon_by_step(cfg, global_step),
        "buffer_size": len(buffer),
        "online": online,
        "best_state_dict": best_state_dict,
        "warmstart": warmstart_info,
    }
