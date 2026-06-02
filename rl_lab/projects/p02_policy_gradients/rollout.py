from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    @property
    def num_steps(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def num_envs(self) -> int:
        return int(self.rewards.shape[1])

    def flatten(self) -> dict[str, torch.Tensor]:
        """Flatten [time, env, ...] tensors into [time * env, ...]."""

        data = {
            "obs": _flatten_time_env(self.obs),
            "actions": _flatten_time_env(self.actions),
            "logprobs": _flatten_time_env(self.logprobs),
            "rewards": _flatten_time_env(self.rewards),
            "dones": _flatten_time_env(self.dones),
            "values": _flatten_time_env(self.values),
        }
        if self.advantages is not None:
            data["advantages"] = _flatten_time_env(self.advantages)
        if self.returns is not None:
            data["returns"] = _flatten_time_env(self.returns)
        return data


def _flatten_time_env(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError(f"Expected at least [time, env] dims, got shape {tuple(x.shape)}")
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


class RolloutStorage:
    """Preallocated rollout storage for on-policy algorithms."""

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        device: torch.device,
    ) -> None:
        self.rollout_steps = int(rollout_steps)
        self.num_envs = int(num_envs)
        self.device = device
        self.step = 0

        self.obs = torch.zeros(
            (rollout_steps, num_envs, *obs_shape), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros((rollout_steps, num_envs), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if self.step >= self.rollout_steps:
            raise IndexError("RolloutStorage is full; call reset() before adding more transitions")

        self.obs[self.step].copy_(obs)
        self.actions[self.step].copy_(action)
        self.logprobs[self.step].copy_(logprob)
        self.rewards[self.step].copy_(reward)
        self.dones[self.step].copy_(done)
        self.values[self.step].copy_(value)
        self.step += 1

    def as_batch(self) -> RolloutBatch:
        if self.step != self.rollout_steps:
            raise RuntimeError(
                f"RolloutStorage is incomplete: {self.step}/{self.rollout_steps} steps collected"
            )
        return RolloutBatch(
            obs=self.obs,
            actions=self.actions,
            logprobs=self.logprobs,
            rewards=self.rewards,
            dones=self.dones,
            values=self.values,
        )

    def reset(self) -> None:
        self.step = 0
