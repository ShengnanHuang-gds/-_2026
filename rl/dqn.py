"""
DQN Q-network for the retail MDP.

Input:  encoded state vector (default dim=94)
Output: Q-values for 18 discrete actions

Supports standard MLP and Dueling architectures.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from mdp.actions import NUM_ACTIONS


DEFAULT_STATE_DIM = 94
DEFAULT_HIDDEN = (128, 128)

QNetLike = Union["QNetwork", "DuelingQNetwork"]


class QNetwork(nn.Module):
    """
    Multilayer perceptron Q(s, ·).

    Architecture (default):
        94 → 128 → ReLU → 128 → ReLU → 18
    """

    def __init__(
        self,
        state_dim: int = DEFAULT_STATE_DIM,
        num_actions: int = NUM_ACTIONS,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must be non-empty")

        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)

        layers: list[nn.Module] = []
        in_dim = self.state_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(hidden)))
            layers.append(nn.ReLU())
            in_dim = int(hidden)
        layers.append(nn.Linear(in_dim, self.num_actions))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        squeezed = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeezed = True
        if state.dim() != 2 or state.size(-1) != self.state_dim:
            raise ValueError(
                f"expected state shape (..., {self.state_dim}), got {tuple(state.shape)}"
            )
        q_values = self.net(state)
        if squeezed:
            q_values = q_values.squeeze(0)
        return q_values

    @torch.no_grad()
    def select_greedy(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> int:
        q_values = self.forward(state)
        if q_values.dim() == 2:
            if q_values.size(0) != 1:
                raise ValueError("select_greedy expects a single state")
            q_values = q_values.squeeze(0)

        if action_mask is not None:
            mask = action_mask.to(device=q_values.device, dtype=torch.bool)
            if mask.shape != (self.num_actions,):
                raise ValueError(
                    f"action_mask shape must be ({self.num_actions},), "
                    f"got {tuple(mask.shape)}"
                )
            if not bool(mask.any()):
                raise ValueError("action_mask has no legal actions")
            q_values = q_values.masked_fill(~mask, float("-inf"))

        return int(torch.argmax(q_values).item())


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a).

    Shared trunk → value stream + advantage stream.
    """

    def __init__(
        self,
        state_dim: int = DEFAULT_STATE_DIM,
        num_actions: int = NUM_ACTIONS,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must be non-empty")

        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)

        trunk_layers: list[nn.Module] = []
        in_dim = self.state_dim
        # Use all but last hidden size for trunk when multiple layers.
        trunk_sizes = (
            list(hidden_sizes[:-1]) if len(hidden_sizes) > 1 else list(hidden_sizes)
        )
        for hidden in trunk_sizes:
            trunk_layers.append(nn.Linear(in_dim, int(hidden)))
            trunk_layers.append(nn.ReLU())
            in_dim = int(hidden)
        self.trunk = nn.Sequential(*trunk_layers)

        head_hidden = int(hidden_sizes[-1])
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, self.num_actions),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        squeezed = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeezed = True
        if state.dim() != 2 or state.size(-1) != self.state_dim:
            raise ValueError(
                f"expected state shape (..., {self.state_dim}), got {tuple(state.shape)}"
            )
        features = self.trunk(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        if squeezed:
            q_values = q_values.squeeze(0)
        return q_values

    @torch.no_grad()
    def select_greedy(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> int:
        q_values = self.forward(state)
        if q_values.dim() == 2:
            if q_values.size(0) != 1:
                raise ValueError("select_greedy expects a single state")
            q_values = q_values.squeeze(0)
        if action_mask is not None:
            mask = action_mask.to(device=q_values.device, dtype=torch.bool)
            if mask.shape != (self.num_actions,):
                raise ValueError(
                    f"action_mask shape must be ({self.num_actions},), "
                    f"got {tuple(mask.shape)}"
                )
            if not bool(mask.any()):
                raise ValueError("action_mask has no legal actions")
            q_values = q_values.masked_fill(~mask, float("-inf"))
        return int(torch.argmax(q_values).item())


def states_to_tensor(
    states: np.ndarray,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert float state array to torch tensor on device."""
    tensor = torch.as_tensor(states, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def uses_dueling_network(algorithm: str) -> bool:
    return str(algorithm).lower() in {"dueling", "d3qn"}


def uses_double_q(algorithm: str) -> bool:
    return str(algorithm).lower() in {"ddqn", "d3qn"}


def build_q_network(
    state_dim: int = DEFAULT_STATE_DIM,
    num_actions: int = NUM_ACTIONS,
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN,
    device: Optional[torch.device] = None,
    *,
    dueling: bool = False,
    algorithm: Optional[str] = None,
) -> QNetLike:
    """Factory helper used by training scripts."""
    if algorithm is not None:
        dueling = uses_dueling_network(algorithm)
    cls = DuelingQNetwork if dueling else QNetwork
    net = cls(
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_sizes=hidden_sizes,
    )
    if device is not None:
        net = net.to(device)
    return net
