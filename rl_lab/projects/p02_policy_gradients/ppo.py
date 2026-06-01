from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from rl_lab.projects.p02_policy_gradients.rollout import RolloutBatch


@dataclass
class PPOLossOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute generalized advantage estimates and returns.

    Args:
        rewards: [time, env]
        dones: [time, env], 1.0 when the transition ended the episode, else 0.0
        values: [time, env]
        next_value: [env]
        gamma: discount factor
        gae_lambda: GAE trace parameter

    Returns:
        advantages: [time, env]
        returns: [time, env]

    TODO(user): implement this manually.
    """

    raise NotImplementedError("TODO(user): implement GAE")


def ppo_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute PPO clipped policy loss.

    Args:
        new_logprobs: [batch]
        old_logprobs: [batch]
        advantages: [batch]
        clip_coef: epsilon in PPO clipping

    Returns:
        policy_loss: scalar tensor
        approx_kl: scalar tensor
        clip_fraction: scalar tensor

    TODO(user): implement this manually.
    """

    raise NotImplementedError("TODO(user): implement PPO policy loss")


def ppo_value_loss(
    new_values: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """Compute value function loss.

    TODO(user): implement this manually. Start with 0.5 * mean squared error.
    Later, optionally add PPO-style value clipping.
    """

    raise NotImplementedError("TODO(user): implement PPO value loss")


def ppo_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
) -> PPOLossOutput:
    """Compute the full PPO objective for one minibatch.

    This wrapper is intentionally thin. You should implement the math in
    `ppo_policy_loss` and `ppo_value_loss` first.
    """

    if "advantages" not in batch or "returns" not in batch:
        raise KeyError("Batch must include advantages and returns")

    new_logprobs, entropy, new_values = model.evaluate_actions(batch["obs"], batch["actions"])
    policy_loss, approx_kl, clip_fraction = ppo_policy_loss(
        new_logprobs=new_logprobs,
        old_logprobs=batch["logprobs"],
        advantages=batch["advantages"],
        clip_coef=clip_coef,
    )
    value_loss = ppo_value_loss(new_values=new_values, returns=batch["returns"])
    entropy_mean = entropy.mean()
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean
    return PPOLossOutput(
        loss=loss,
        policy_loss=policy_loss.detach(),
        value_loss=value_loss.detach(),
        entropy=entropy_mean.detach(),
        approx_kl=approx_kl.detach(),
        clip_fraction=clip_fraction.detach(),
    )


def update_ppo(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    next_value: torch.Tensor,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Run PPO minibatch updates over one rollout.

    TODO(user): implement this manually after `compute_gae` and loss functions.

    Suggested structure:
      1. compute advantages and returns
      2. normalize advantages
      3. flatten rollout
      4. shuffle minibatches for update_epochs
      5. compute loss
      6. backprop and clip grads
      7. early-stop on target_kl if configured
      8. return scalar metrics
    """

    raise NotImplementedError("TODO(user): implement PPO update loop")
