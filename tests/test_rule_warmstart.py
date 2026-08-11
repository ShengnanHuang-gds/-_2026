from __future__ import annotations

from pathlib import Path

import torch

from mdp.actions import decode_action
from mdp.policies import RuleBasedPolicy
from rl.dqn_train import (
    DQNTrainConfig,
    behavioral_clone_pretrain,
    fill_buffer_with_rule_policy,
    run_dqn_training,
)
from rl.replay_buffer import ReplayBuffer
from rl.dqn import build_q_network


def test_fill_buffer_with_rule_only_e0_actions():
    cfg = DQNTrainConfig(
        disruption_matrix_profile="harsh_strong",
        warmup_days=0,
        evaluation_days=5,
        num_forward_warehouses=2,
        forbid_expedite=True,
        demand_intensity_scale=1.0,
        seed=0,
    )
    # state dim for 2 FW: 5 + 10 + 20 + 5 + 5 + 4 + 4 + 1 = 54
    buf = ReplayBuffer(capacity=200, state_dim=54, seed=0)
    stats = fill_buffer_with_rule_policy(cfg, buf, episodes=2, seed_base=0)
    assert stats["transitions"] == 10
    assert len(buf) == 10
    for i in range(len(buf)):
        assert decode_action(int(buf.actions[i])).expedite == 0


def test_rule_warmstart_short_train(tmp_path: Path):
    ckpt = tmp_path / "ws.pt"
    cfg = DQNTrainConfig(
        algorithm="ddqn",
        disruption_matrix_profile="harsh_strong",
        warmup_days=0,
        evaluation_days=4,
        episodes=3,
        num_forward_warehouses=2,
        forbid_expedite=True,
        buffer_capacity=500,
        batch_size=8,
        start_learning_after=0,
        rule_warmstart_episodes=2,
        rule_bc_updates=20,
        rule_bc_batch_size=8,
        eps_start=0.2,
        eps_end=0.05,
        eps_decay_steps=20,
        eval_every_episodes=3,
        eval_evaluation_days=4,
        eval_warmup_days=0,
        eval_num_seeds=1,
        seed=0,
        checkpoint_path=str(ckpt),
    )
    summary = run_dqn_training(cfg)
    assert summary["warmstart"]["transitions"] == 8
    assert summary["warmstart"]["episodes"] == 2
    assert len(summary["warmstart"]["bc_losses"]) == 20
    assert summary["warmstart"]["pre_rl_eval_profit"] is not None
    assert ckpt.is_file()
