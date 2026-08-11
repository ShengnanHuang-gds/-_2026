from __future__ import annotations

import numpy as np
import pytest
import torch

from mdp.actions import NUM_ACTIONS
from rl.dqn import DEFAULT_STATE_DIM
from rl.replay_buffer import ReplayBuffer


def _transition(i: int = 0):
    state = np.full(DEFAULT_STATE_DIM, float(i), dtype=np.float32)
    next_state = np.full(DEFAULT_STATE_DIM, float(i + 1), dtype=np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=bool)
    if i % 2 == 0:
        mask[3] = False  # sample illegal expedite under U3-like mask
    return state, i % NUM_ACTIONS, float(i), next_state, i % 5 == 0, mask


def test_add_and_len():
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0
    buf.add(*_transition(0))
    assert len(buf) == 1
    assert buf.size == 1


def test_capacity_overwrite():
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.add(*_transition(i))
    assert len(buf) == 3
    batch = buf.sample(3)
    # Oldest kept entries should be i=2,3,4 (circular overwrite of 0,1)
    assert set(batch.actions.tolist()) == {2 % NUM_ACTIONS, 3 % NUM_ACTIONS, 4 % NUM_ACTIONS}


def test_sample_shapes_and_dtypes():
    buf = ReplayBuffer(capacity=20, seed=0)
    for i in range(10):
        buf.add(*_transition(i))
    batch = buf.sample(4)
    assert batch.states.shape == (4, DEFAULT_STATE_DIM)
    assert batch.actions.shape == (4,)
    assert batch.rewards.shape == (4,)
    assert batch.next_states.shape == (4, DEFAULT_STATE_DIM)
    assert batch.dones.shape == (4,)
    assert batch.next_masks.shape == (4, NUM_ACTIONS)
    assert batch.states.dtype == np.float32
    assert batch.actions.dtype == np.int64
    assert batch.rewards.dtype == np.float32
    assert batch.dones.dtype == np.float32
    assert batch.next_masks.dtype == np.bool_


def test_sample_rejects_oversized_batch():
    buf = ReplayBuffer(capacity=5)
    buf.add(*_transition(0))
    with pytest.raises(ValueError, match="batch_size"):
        buf.sample(2)


def test_sample_rejects_empty():
    buf = ReplayBuffer(capacity=5)
    with pytest.raises(ValueError, match="empty"):
        buf.sample(1)


def test_add_rejects_bad_action():
    buf = ReplayBuffer(capacity=5)
    state, _, reward, next_state, done, mask = _transition(0)
    with pytest.raises(ValueError, match="action must be"):
        buf.add(state, NUM_ACTIONS, reward, next_state, done, mask)


def test_sample_tensors():
    buf = ReplayBuffer(capacity=20, seed=1)
    for i in range(8):
        buf.add(*_transition(i))
    states, actions, rewards, next_states, dones, next_masks = buf.sample_tensors(3)
    assert isinstance(states, torch.Tensor)
    assert states.shape == (3, DEFAULT_STATE_DIM)
    assert actions.dtype == torch.int64
    assert next_masks.dtype == torch.bool
    assert next_masks.shape == (3, NUM_ACTIONS)


def test_clear():
    buf = ReplayBuffer(capacity=5)
    buf.add(*_transition(0))
    buf.clear()
    assert len(buf) == 0
    with pytest.raises(ValueError, match="empty"):
        buf.sample(1)
