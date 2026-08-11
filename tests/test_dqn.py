from __future__ import annotations

import numpy as np
import pytest
import torch

from mdp.actions import NUM_ACTIONS
from rl.dqn import DEFAULT_STATE_DIM, QNetwork, build_q_network, states_to_tensor


def test_q_network_output_shape_single_and_batch():
    net = QNetwork()
    single = torch.randn(DEFAULT_STATE_DIM)
    q_single = net(single)
    assert q_single.shape == (NUM_ACTIONS,)

    batch = torch.randn(8, DEFAULT_STATE_DIM)
    q_batch = net(batch)
    assert q_batch.shape == (8, NUM_ACTIONS)


def test_q_network_rejects_wrong_state_dim():
    net = QNetwork()
    with pytest.raises(ValueError, match="expected state shape"):
        net(torch.randn(10))


def test_select_greedy_respects_action_mask():
    net = QNetwork()
    # Make Q deterministic-ish by evaluating once and forcing mask
    state = torch.zeros(DEFAULT_STATE_DIM)
    with torch.no_grad():
        q = net(state).clone()
    # Prefer action 16 unless masked; mask out all e=1 actions (odd pattern:
    # actions with expedite=1 are 3,4,5,9,10,11,15,16,17)
    mask = torch.ones(NUM_ACTIONS, dtype=torch.bool)
    best = int(torch.argmax(q).item())
    mask[best] = False
    chosen = net.select_greedy(state, action_mask=mask)
    assert chosen != best
    assert bool(mask[chosen])


def test_select_greedy_raises_if_mask_empty():
    net = QNetwork()
    mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
    with pytest.raises(ValueError, match="no legal actions"):
        net.select_greedy(torch.zeros(DEFAULT_STATE_DIM), action_mask=mask)


def test_build_q_network_and_states_to_tensor():
    net = build_q_network()
    assert isinstance(net, QNetwork)
    arr = np.zeros((4, DEFAULT_STATE_DIM), dtype=np.float32)
    tensor = states_to_tensor(arr)
    assert tensor.shape == (4, DEFAULT_STATE_DIM)
    assert tensor.dtype == torch.float32
    q = net(tensor)
    assert q.shape == (4, NUM_ACTIONS)


def test_dueling_network_shapes_and_greedy():
    from rl.dqn import DuelingQNetwork

    net = build_q_network(algorithm="d3qn")
    assert isinstance(net, DuelingQNetwork)
    q = net(torch.randn(DEFAULT_STATE_DIM))
    assert q.shape == (NUM_ACTIONS,)
    mask = torch.ones(NUM_ACTIONS, dtype=torch.bool)
    mask[0] = False
    a = net.select_greedy(torch.zeros(DEFAULT_STATE_DIM), action_mask=mask)
    assert a != 0

