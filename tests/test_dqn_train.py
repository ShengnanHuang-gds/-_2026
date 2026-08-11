from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rl.dqn import QNetwork
from rl.dqn_train import DQNTrainConfig, compute_td_targets, run_dqn_training


def test_ddqn_targets_differ_from_vanilla_when_online_disagrees():
    """
    Construct online/target so argmax_online != argmax_target;
    DDQN and DQN bootstrap values must differ.
    """
    online = QNetwork(state_dim=4, num_actions=3, hidden_sizes=(8,))
    target = QNetwork(state_dim=4, num_actions=3, hidden_sizes=(8,))
    with torch.no_grad():
        for p in online.parameters():
            p.zero_()
        for p in target.parameters():
            p.zero_()
        # Bias-only Q: online prefers action 0, target prefers action 2
        online.net[-1].bias.copy_(torch.tensor([3.0, 1.0, 0.0]))
        target.net[-1].bias.copy_(torch.tensor([0.0, 1.0, 5.0]))

    next_states = torch.zeros(2, 4)
    rewards = torch.zeros(2)
    dones = torch.zeros(2)
    masks = torch.ones(2, 3, dtype=torch.bool)

    y_dqn = compute_td_targets(
        online, target, next_states, rewards, dones, masks, gamma=1.0, algorithm="dqn"
    )
    y_ddqn = compute_td_targets(
        online, target, next_states, rewards, dones, masks, gamma=1.0, algorithm="ddqn"
    )
    # DQN takes max over target → 5; DDQN takes target[online_argmax=0] → 0
    assert torch.allclose(y_dqn, torch.tensor([5.0, 5.0]))
    assert torch.allclose(y_ddqn, torch.tensor([0.0, 0.0]))


def test_ddqn_respects_action_mask():
    online = QNetwork(state_dim=4, num_actions=3, hidden_sizes=(8,))
    target = QNetwork(state_dim=4, num_actions=3, hidden_sizes=(8,))
    with torch.no_grad():
        for p in online.parameters():
            p.zero_()
        for p in target.parameters():
            p.zero_()
        online.net[-1].bias.copy_(torch.tensor([9.0, 1.0, 2.0]))
        target.net[-1].bias.copy_(torch.tensor([0.0, 4.0, 1.0]))

    next_states = torch.zeros(1, 4)
    rewards = torch.zeros(1)
    dones = torch.zeros(1)
    # Mask out action 0 so online must pick action 2 (bias 2 > 1)
    masks = torch.tensor([[False, True, True]])

    y = compute_td_targets(
        online, target, next_states, rewards, dones, masks, gamma=1.0, algorithm="ddqn"
    )
    assert torch.allclose(y, torch.tensor([1.0]))


def test_ddqn_long_loop_with_periodic_eval(tmp_path: Path):
    ckpt = tmp_path / "ddqn_best.pt"
    cfg = DQNTrainConfig(
        algorithm="ddqn",
        warmup_days=0,
        evaluation_days=4,
        episodes=4,
        num_forward_warehouses=2,
        buffer_capacity=200,
        batch_size=8,
        start_learning_after=8,
        train_every=1,
        target_update_every=5,
        lr=3e-4,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay_steps=20,
        seed=0,
        device="cpu",
        eval_every_episodes=2,
        eval_evaluation_days=4,
        eval_warmup_days=0,
        eval_num_seeds=2,
        checkpoint_path=str(ckpt),
    )
    summary = run_dqn_training(cfg)
    assert summary["global_steps"] == 4 * 4
    assert summary["updates"] >= 1
    assert summary["final_eps"] < 0.5
    assert len(summary["eval_history"]) >= 2
    assert summary["best_eval_profit"] is not None
    assert ckpt.is_file()


def test_invalid_algorithm_raises():
    with pytest.raises(ValueError, match="unknown algorithm"):
        DQNTrainConfig(algorithm="ppo")


def test_dueling_and_d3qn_accepted():
    DQNTrainConfig(algorithm="dueling")
    DQNTrainConfig(algorithm="d3qn")


def test_ddqn_forbid_expedite_never_selects_e1():
    from mdp.actions import decode_action
    from mdp.state_encoder import build_state_encoder_from_env
    from rl.dqn import states_to_tensor
    from rl.dqn_train import build_env

    cfg = DQNTrainConfig(
        algorithm="ddqn",
        forbid_expedite=True,
        warmup_days=0,
        evaluation_days=5,
        episodes=2,
        num_forward_warehouses=2,
        buffer_capacity=200,
        batch_size=8,
        start_learning_after=8,
        eps_start=1.0,
        eps_end=1.0,
        eps_decay_steps=1,
        eval_every_episodes=0,
        seed=0,
    )
    summary = run_dqn_training(cfg)
    online = summary["online"]
    assert online is not None
    online.eval()

    env = build_env(cfg, seed=99, evaluation_days=5, warmup_days=0)
    state = env.reset(seed=99)
    enc = build_state_encoder_from_env(env)
    for _ in range(5):
        mask = env.get_action_mask()
        assert int(mask.sum()) == 9
        a = online.select_greedy(
            states_to_tensor(enc.encode(state)),
            action_mask=torch.as_tensor(mask, dtype=torch.bool),
        )
        assert decode_action(a).expedite == 0
        state, _, done, _ = env.step(a)
        if done:
            break


def test_smoke_compat_alias():
    from experiments.train_dqn_smoke import SmokeConfig, run_smoke_training

    cfg = SmokeConfig(
        algorithm="ddqn",
        forbid_expedite=True,
        warmup_days=0,
        evaluation_days=4,
        episodes=3,
        num_forward_warehouses=2,
        buffer_capacity=100,
        batch_size=4,
        start_learning_after=4,
        eps_end=0.01,
        eps_decay_steps=10,
        eval_every_episodes=0,
        seed=1,
    )
    summary = run_smoke_training(cfg)
    assert len(summary["episode_returns"]) == 3
