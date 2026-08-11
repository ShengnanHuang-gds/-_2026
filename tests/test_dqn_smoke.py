from __future__ import annotations

from experiments.train_dqn_smoke import SmokeConfig, run_smoke_training


def test_dqn_smoke_runs_and_learns():
    """Tiny end-to-end loop: fills buffer and performs at least one update."""
    cfg = SmokeConfig(
        warmup_days=0,
        evaluation_days=4,
        episodes=3,
        num_forward_warehouses=2,
        buffer_capacity=200,
        batch_size=8,
        start_learning_after=8,
        train_every=1,
        target_update_every=5,
        eps_decay_steps=20,
        seed=0,
        device="cpu",
        eval_every_episodes=0,
    )
    summary = run_smoke_training(cfg)
    assert summary["global_steps"] == 4 * 3
    assert summary["updates"] >= 1
    assert len(summary["episode_returns"]) == 3
    assert len(summary["episode_raw_profits"]) == 3
    assert any(loss == loss for loss in summary["episode_losses"])
