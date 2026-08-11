"""
Experience replay for DQN training.

Stores transitions:
    (s, a, r, s', done, mask')
where mask' is the legal-action mask at the next state (True = legal).
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import torch

from mdp.actions import NUM_ACTIONS
from rl.dqn import DEFAULT_STATE_DIM


class TransitionBatch(NamedTuple):
    states: np.ndarray          # (B, state_dim) float32
    actions: np.ndarray         # (B,) int64
    rewards: np.ndarray         # (B,) float32
    next_states: np.ndarray     # (B, state_dim) float32
    dones: np.ndarray           # (B,) float32  (1.0 if terminal)
    next_masks: np.ndarray      # (B, num_actions) bool


class ReplayBuffer:
    """Fixed-capacity circular buffer with uniform random sampling."""

    def __init__(
        self,
        capacity: int,
        state_dim: int = DEFAULT_STATE_DIM,
        num_actions: int = NUM_ACTIONS,
        seed: Optional[int] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")

        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.next_masks = np.ones(
            (self.capacity, self.num_actions), dtype=np.bool_
        )

        self._size = 0
        self._pos = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        """Insert one transition; overwrites oldest when full."""
        state = np.asarray(state, dtype=np.float32).reshape(self.state_dim)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(self.state_dim)
        next_mask = np.asarray(next_mask, dtype=np.bool_).reshape(self.num_actions)

        action = int(action)
        if not (0 <= action < self.num_actions):
            raise ValueError(
                f"action must be in 0..{self.num_actions - 1}, got {action}"
            )

        idx = self._pos
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = float(reward)
        self.next_states[idx] = next_state
        self.dones[idx] = 1.0 if done else 0.0
        self.next_masks[idx] = next_mask

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        """Uniformly sample a batch of transitions."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")
        if batch_size > self._size:
            raise ValueError(
                f"batch_size {batch_size} > buffer size {self._size}"
            )

        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return TransitionBatch(
            states=self.states[indices].copy(),
            actions=self.actions[indices].copy(),
            rewards=self.rewards[indices].copy(),
            next_states=self.next_states[indices].copy(),
            dones=self.dones[indices].copy(),
            next_masks=self.next_masks[indices].copy(),
        )

    def sample_tensors(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Sample a batch and convert to torch tensors.

        Returns:
            states, actions, rewards, next_states, dones, next_masks
        """
        batch = self.sample(batch_size)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)
        next_states = torch.as_tensor(
            batch.next_states, dtype=torch.float32, device=device
        )
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=device)
        next_masks = torch.as_tensor(
            batch.next_masks, dtype=torch.bool, device=device
        )
        return states, actions, rewards, next_states, dones, next_masks

    def clear(self) -> None:
        self._size = 0
        self._pos = 0
