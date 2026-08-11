"""
Masked discrete Actor-Critic for PPO on the retail MDP.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from mdp.actions import NUM_ACTIONS
from rl.dqn import DEFAULT_HIDDEN, DEFAULT_STATE_DIM, states_to_tensor


class ActorCritic(nn.Module):
    """Shared trunk → policy logits + state value."""

    def __init__(
        self,
        state_dim: int = DEFAULT_STATE_DIM,
        num_actions: int = NUM_ACTIONS,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or num_actions <= 0 or not hidden_sizes:
            raise ValueError("invalid ActorCritic dimensions")
        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)

        layers: list[nn.Module] = []
        in_dim = self.state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.ReLU())
            in_dim = int(h)
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_dim, self.num_actions)
        self.value_head = nn.Linear(in_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        features = self.trunk(state)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def masked_logits(
        self, logits: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        # action_mask: (batch, num_actions) bool
        return logits.masked_fill(~action_mask, float("-inf"))

    def act(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample (or greedy) one action.

        Returns action_id, log_prob (scalar tensor), value (scalar tensor).
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)
        logits, value = self.forward(state)
        logits = self.masked_logits(logits, action_mask)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.squeeze(0), value.squeeze(0)

    @torch.no_grad()
    def select_greedy(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> int:
        if action_mask is None:
            action_mask = torch.ones(self.num_actions, dtype=torch.bool, device=state.device)
        action, _, _ = self.act(state, action_mask, deterministic=True)
        return action


def build_actor_critic(
    state_dim: int = DEFAULT_STATE_DIM,
    num_actions: int = NUM_ACTIONS,
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN,
    device: Optional[torch.device] = None,
) -> ActorCritic:
    net = ActorCritic(state_dim, num_actions, hidden_sizes)
    if device is not None:
        net = net.to(device)
    return net


__all__ = [
    "ActorCritic",
    "build_actor_critic",
    "states_to_tensor",
]
