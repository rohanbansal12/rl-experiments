from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCriticMLP(nn.Module):
    """Simple actor-critic network for discrete-action control tasks."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("ActorCriticMLP currently supports Discrete action spaces only")
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("ActorCriticMLP currently supports Box observation spaces only")

        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(action_space.n)
        act_layer = nn.Tanh if activation == "tanh" else nn.ReLU

        torso_layers: list[nn.Module] = []
        prev_dim = obs_dim
        for hidden_dim in hidden_sizes:
            torso_layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            torso_layers.append(act_layer())
            prev_dim = int(hidden_dim)

        self.torso = nn.Sequential(*torso_layers)
        self.policy_head = nn.Linear(prev_dim, act_dim)
        self.value_head = nn.Linear(prev_dim, 1)
        self.obs_shape = observation_space.shape

        self.apply(_orthogonal_init)
        nn.init.constant_(self.policy_head.bias, 0.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        # obs: [batch, *obs_shape]
        x = obs.float().view(obs.shape[0], -1)
        features = self.torso(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return Categorical(logits=logits), value

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprob, entropy, value


def _orthogonal_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        nn.init.constant_(module.bias, 0.0)
