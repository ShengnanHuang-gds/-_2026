"""
PPO training for the retail MDP (masked discrete actions).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from mdp.actions import decode_action, num_actions_for
from mdp.env import MDPState
from mdp.policies import RuleBasedPolicy
from mdp.state_encoder import EncoderLike, build_state_encoder_from_env
from rl.dqn import states_to_tensor
from rl.dqn_train import (
    DQNTrainConfig,
    EvalResult,
    build_env,
    evaluate_greedy,
)
from rl.ppo import ActorCritic, build_actor_critic


@dataclass
class PPOTrainConfig(DQNTrainConfig):
    """PPO hyperparameters; reuses env/warmstart fields from DQNTrainConfig."""

    algorithm: str = "ppo"
    # PPO-specific
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 256
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    # Load weights before RL (e.g. forbid_e1 → allow_e1 curriculum).
    init_checkpoint: Optional[str] = None
    # Override defaults more suitable for on-policy
    lr: float = 3e-4
    eps_start: float = 0.0  # unused
    episodes: int = 50

    def __post_init__(self) -> None:
        # Bypass DQN algorithm check by temporarily using a valid label,
        # then force algorithm=ppo after parent validation of other fields.
        requested = str(self.algorithm).lower()
        self.algorithm = "ddqn"
        super().__post_init__()
        if requested != "ppo":
            raise ValueError("PPOTrainConfig.algorithm must be 'ppo'")
        self.algorithm = "ppo"


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float,
    lam: float,
) -> tuple:
    advantages: List[float] = []
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages.insert(0, gae)
        next_value = values[t]
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


@torch.no_grad()
def evaluate_ppo_greedy(
    actor: ActorCritic,
    cfg: PPOTrainConfig,
    *,
    episode: int,
    device: torch.device,
) -> EvalResult:
    # Reuse DQN greedy eval by duck-typing select_greedy.
    return evaluate_greedy(actor, cfg, episode=episode, device=device)  # type: ignore[arg-type]


def behavioral_clone_actor(
    actor: ActorCritic,
    states: np.ndarray,
    actions: np.ndarray,
    masks: np.ndarray,
    *,
    updates: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> List[float]:
    if len(states) == 0 or updates <= 0:
        return []
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    losses: List[float] = []
    n = len(states)
    actor.train()
    for i in range(updates):
        idx = np.random.randint(0, n, size=min(batch_size, n))
        s = torch.as_tensor(states[idx], dtype=torch.float32, device=device)
        a = torch.as_tensor(actions[idx], dtype=torch.int64, device=device)
        m = torch.as_tensor(masks[idx], dtype=torch.bool, device=device)
        logits, _ = actor(s)
        logits = actor.masked_logits(logits, m)
        loss = F.cross_entropy(logits, a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
        opt.step()
        losses.append(float(loss.item()))
        if (i + 1) % max(1, updates // 5) == 0 or i == 0:
            print(
                f"[warmstart PPO-BC {i + 1:04d}/{updates}] loss={loss.item():.4f}",
                flush=True,
            )
    return losses


def collect_rule_trajectories(
    cfg: PPOTrainConfig,
    *,
    episodes: int,
    seed_base: int,
) -> tuple:
    policy = RuleBasedPolicy(name="rule_ws", action_space=cfg.action_space)
    states_l: List[np.ndarray] = []
    actions_l: List[int] = []
    masks_l: List[np.ndarray] = []
    for ep in range(episodes):
        seed = seed_base + ep
        env = build_env(cfg, seed=seed)
        state = env.reset(seed=seed)
        encoder = build_state_encoder_from_env(
            env, encoder_type=cfg.state_encoder
        )
        while not env.done:
            mask = env.get_action_mask()
            action = policy.select(state)
            if not bool(mask[action]):
                action = int(np.flatnonzero(mask)[0])
            states_l.append(encoder.encode(state))
            actions_l.append(action)
            masks_l.append(mask.copy())
            state, _, _, _ = env.step(action)
    return (
        np.asarray(states_l, dtype=np.float32),
        np.asarray(actions_l, dtype=np.int64),
        np.asarray(masks_l, dtype=bool),
    )


def run_ppo_training(cfg: PPOTrainConfig) -> Dict[str, object]:
    device = torch.device(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = build_env(cfg, seed=cfg.seed)
    state: MDPState = env.reset(seed=cfg.seed)
    encoder: EncoderLike = build_state_encoder_from_env(
        env, encoder_type=cfg.state_encoder
    )
    n_actions = num_actions_for(cfg.action_space)
    print(
        f"[train PPO] encoder={cfg.state_encoder} dim={encoder.dim} "
        f"n_actions={n_actions} forbid_e1={cfg.forbid_expedite}",
        flush=True,
    )

    actor = build_actor_critic(
        state_dim=encoder.dim, num_actions=n_actions, device=device
    )
    if cfg.init_checkpoint:
        ckpt = torch.load(
            cfg.init_checkpoint, map_location=device, weights_only=False
        )
        actor.load_state_dict(ckpt["model_state_dict"])
        print(
            f"[train PPO] loaded init_checkpoint={cfg.init_checkpoint} "
            f"(prior_best={ckpt.get('best_eval_profit')})",
            flush=True,
        )
    optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr)

    warmstart_info: Dict[str, object] = {"episodes": 0, "bc_losses": []}
    best_eval_profit = float("-inf")
    best_state_dict = None
    eval_history: List[EvalResult] = []
    episode_raw_profits: List[float] = []
    train_e1_ratios: List[float] = []
    train_e1_legal_ratios: List[float] = []
    t0 = time.time()

    if cfg.rule_warmstart_episodes > 0:
        print(
            f"[warmstart] collecting {cfg.rule_warmstart_episodes} rule episodes ...",
            flush=True,
        )
        s_arr, a_arr, m_arr = collect_rule_trajectories(
            cfg,
            episodes=cfg.rule_warmstart_episodes,
            seed_base=cfg.seed + 50_000,
        )
        warmstart_info["episodes"] = cfg.rule_warmstart_episodes
        warmstart_info["transitions"] = int(len(s_arr))
        if cfg.rule_bc_updates > 0:
            bc_losses = behavioral_clone_actor(
                actor,
                s_arr,
                a_arr,
                m_arr,
                updates=cfg.rule_bc_updates,
                batch_size=cfg.rule_bc_batch_size,
                lr=cfg.rule_bc_lr,
                device=device,
            )
            warmstart_info["bc_losses"] = bc_losses
        actor.eval()
        pre = evaluate_ppo_greedy(actor, cfg, episode=0, device=device)
        actor.train()
        warmstart_info["pre_rl_eval_profit"] = pre.mean_raw_profit
        print(
            f"[warmstart] pre-RL greedy eval={pre.mean_raw_profit:.1f}",
            flush=True,
        )
        if pre.mean_raw_profit > best_eval_profit:
            best_eval_profit = pre.mean_raw_profit
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in actor.state_dict().items()
            }
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

        # --- collect one episode ---
        states: List[np.ndarray] = []
        actions: List[int] = []
        rewards: List[float] = []
        dones: List[bool] = []
        values: List[float] = []
        log_probs: List[float] = []
        masks: List[np.ndarray] = []
        ep_raw = 0.0
        ep_e1 = 0
        ep_e1_legal = 0
        ep_steps = 0

        actor.eval()
        while not env.done:
            mask = env.get_action_mask()
            vec = encoder.encode(state)
            state_t = states_to_tensor(vec, device=device)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
            with torch.no_grad():
                action, log_p, value = actor.act(state_t, mask_t, deterministic=False)
            next_state, reward, done, info = env.step(action)
            states.append(vec)
            actions.append(action)
            rewards.append(float(reward))
            dones.append(bool(done))
            values.append(float(value.item()))
            log_probs.append(float(log_p.item()))
            masks.append(mask.copy())
            ep_raw += float(info["raw_reward"])
            act = decode_action(action, space=cfg.action_space)
            if act.expedite == 1:
                ep_e1 += 1
            # How often was e=1 even legal under the current mask?
            for aid in range(len(mask)):
                if mask[aid] and decode_action(aid, space=cfg.action_space).expedite == 1:
                    ep_e1_legal += 1
                    break
            ep_steps += 1
            state = next_state

        tracker_profit = float(
            env.engine.performance_tracker.summarize()["total_profit"]
        )
        episode_raw_profits.append(tracker_profit)
        e1_ratio = ep_e1 / ep_steps if ep_steps else 0.0
        e1_legal_ratio = ep_e1_legal / ep_steps if ep_steps else 0.0
        train_e1_ratios.append(e1_ratio)
        train_e1_legal_ratios.append(e1_legal_ratio)

        advantages, returns = compute_gae(
            rewards, values, dones, cfg.gamma, cfg.gae_lambda
        )
        adv_arr = np.asarray(advantages, dtype=np.float32)
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        s_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
        a_t = torch.as_tensor(actions, dtype=torch.int64, device=device)
        old_log_t = torch.as_tensor(log_probs, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv_arr, dtype=torch.float32, device=device)
        m_t = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=device)

        # --- PPO updates ---
        actor.train()
        n = len(actions)
        idxs = np.arange(n)
        ep_loss = 0.0
        n_updates = 0
        for _ in range(cfg.ppo_epochs):
            rng.shuffle(idxs)
            for start in range(0, n, cfg.ppo_minibatch_size):
                mb = idxs[start : start + cfg.ppo_minibatch_size]
                logits, values_pred = actor(s_t[mb])
                logits = actor.masked_logits(logits, m_t[mb])
                dist = Categorical(logits=logits)
                new_log = dist.log_prob(a_t[mb])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log - old_log_t[mb])
                adv = adv_t[mb]
                unclipped = ratio * adv
                clipped = (
                    torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
                )
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values_pred, ret_t[mb])
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), max_norm=cfg.max_grad_norm
                )
                optimizer.step()
                ep_loss += float(loss.item())
                n_updates += 1

        if ep % cfg.log_every_episode == 0 or ep == 1:
            print(
                f"[train PPO ep {ep:03d}/{cfg.episodes}] "
                f"raw_profit={tracker_profit:.1f} "
                f"loss={ep_loss / max(1, n_updates):.4f} "
                f"sample_e1={e1_ratio:.1%} "
                f"e1_legal_days={e1_legal_ratio:.1%}",
                flush=True,
            )

        if cfg.eval_every_episodes > 0 and (
            ep % cfg.eval_every_episodes == 0 or ep == cfg.episodes
        ):
            actor.eval()
            result = evaluate_ppo_greedy(actor, cfg, episode=ep, device=device)
            actor.train()
            eval_history.append(result)
            print(
                f"[eval  PPO ep {ep:03d}] greedy_profit="
                f"{result.mean_raw_profit:.1f} ± {result.std_raw_profit:.1f}",
                flush=True,
            )
            if result.mean_raw_profit > best_eval_profit:
                best_eval_profit = result.mean_raw_profit
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in actor.state_dict().items()
                }

    if best_state_dict is not None and cfg.checkpoint_path:
        path = cfg.checkpoint_path
        torch.save(
            {
                "model_state_dict": best_state_dict,
                "best_eval_profit": best_eval_profit,
                "algorithm": "ppo",
                "state_encoder": cfg.state_encoder,
                "action_space": cfg.action_space,
            },
            path,
        )
        print(
            f"saved best checkpoint → {path} (profit={best_eval_profit:.1f})",
            flush=True,
        )

    elapsed = time.time() - t0
    mean_train_e1 = float(np.mean(train_e1_ratios)) if train_e1_ratios else 0.0
    mean_e1_legal = (
        float(np.mean(train_e1_legal_ratios)) if train_e1_legal_ratios else 0.0
    )
    print(
        f"[train PPO done] mean_sample_e1={mean_train_e1:.1%} "
        f"mean_e1_legal_days={mean_e1_legal:.1%} "
        f"forbid_e1={cfg.forbid_expedite}",
        flush=True,
    )
    return {
        "algorithm": "ppo",
        "actor": actor,
        "online": actor,  # alias for eval helpers expecting "online"
        "best_eval_profit": best_eval_profit if best_state_dict is not None else None,
        "eval_history": eval_history,
        "episode_raw_profits": episode_raw_profits,
        "warmstart": warmstart_info,
        "train_e1_ratios": train_e1_ratios,
        "train_e1_legal_ratios": train_e1_legal_ratios,
        "mean_train_e1_ratio": mean_train_e1,
        "mean_train_e1_legal_ratio": mean_e1_legal,
        "elapsed_sec": elapsed,
        "checkpoint_path": cfg.checkpoint_path,
    }
