from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rl_lab.core.metrics import explained_variance
from rl_lab.projects.p02_policy_gradients.rollout import RolloutBatch


@dataclass
class PPOLossOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


@dataclass(frozen=True)
class PPOUpdateConfig:
    """Hyperparameters needed by one PPO update over one rollout."""

    gamma: float
    gae_lambda: float
    clip_coef: float
    ent_coef: float
    vf_coef: float
    target_kl: float | None
    max_grad_norm: float
    update_epochs: int
    minibatch_size: int
    normalize_advantages: bool = True

    @classmethod
    def from_config(cls, cfg: dict) -> PPOUpdateConfig:
        """Build the narrow update config from the experiment YAML mapping."""

        ppo_cfg = cfg.get("ppo", {})
        train_cfg = cfg.get("train", {})
        optim_cfg = cfg.get("optim", {})
        return cls(
            gamma=float(ppo_cfg.get("gamma", 0.99)),
            gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
            clip_coef=float(ppo_cfg.get("clip_coef", 0.2)),
            ent_coef=float(ppo_cfg.get("ent_coef", 0.0)),
            vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
            target_kl=_optional_float(ppo_cfg.get("target_kl")),
            max_grad_norm=float(optim_cfg.get("max_grad_norm", 0.5)),
            update_epochs=int(train_cfg.get("update_epochs", 1)),
            minibatch_size=int(train_cfg.get("minibatch_size", 64)),
            normalize_advantages=bool(ppo_cfg.get("normalize_advantages", True)),
        )


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
    T, E = rewards.shape
    next_nonterminal = 1.0 - dones
    next_values = torch.cat([values[1:], next_value.unsqueeze(0)], dim=0)

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * next_nonterminal[t] - values[t]
        gae = delta + gamma * gae_lambda * next_nonterminal[t] * gae
        advantages[t] = gae

    return advantages, advantages + values


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

    log_ratio = new_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)
    approx_kl = ((ratio - 1.0) - log_ratio).mean()

    weight = torch.exp(log_ratio)
    clipped_weight = torch.clip(weight, min=1 - clip_coef, max=1 + clip_coef)
    policy_loss = -torch.minimum(
        advantages * weight,
        advantages * clipped_weight,
    ).mean()

    clip_fraction = ((weight < 1 - clip_coef) | (weight > 1 + clip_coef)).float().mean()

    return policy_loss, approx_kl, clip_fraction


def ppo_value_loss(
    new_values: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """Compute value function loss.

    TODO(user): implement this manually. Start with 0.5 * mean squared error.
    Later, optionally add PPO-style value clipping.
    """

    return 0.5 * (new_values - returns).square().mean()


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
        old_logprobs=batch["logprobs"].detach(),
        advantages=batch["advantages"].detach(),
        clip_coef=clip_coef,
    )
    value_loss = ppo_value_loss(new_values=new_values, returns=batch["returns"].detach())
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
    cfg: PPOUpdateConfig,
) -> dict[str, float]:
    """Update the actor-critic model once using a complete rollout.

    Args:
        model: Actor-critic module exposing `evaluate_actions(obs, actions)`.
        optimizer: Optimizer for `model.parameters()`.
        rollout: Complete on-policy rollout with tensors shaped `[time, env, ...]`.
        next_value: Value prediction for the observation after the rollout, shaped `[env]`.
        cfg: Narrow PPO update hyperparameter bundle.

    Returns:
        Scalar metrics for logging. Expected keys include `policy_loss`, `value_loss`,
        `entropy`, `approx_kl`, `clip_fraction`, and `explained_variance`.

    This computes advantages/returns, optionally normalizes advantages, flattens the rollout,
    runs PPO minibatch epochs, clips gradients, optionally stops early on `cfg.target_kl`, and
    returns logging metrics averaged across executed minibatches.
    """

    raw_advantages, returns = compute_gae(
        rewards=rollout.rewards,
        dones=rollout.dones,
        values=rollout.values,
        next_value=next_value,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )
    advantage_mean = float(raw_advantages.mean().item())
    advantage_std = float(raw_advantages.std(unbiased=False).item())
    return_mean = float(returns.mean().item())
    return_std = float(returns.std(unbiased=False).item())
    advantages = raw_advantages
    if cfg.normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    batch = RolloutBatch(
        obs=rollout.obs,
        actions=rollout.actions,
        logprobs=rollout.logprobs,
        rewards=rollout.rewards,
        dones=rollout.dones,
        values=rollout.values,
        advantages=advantages,
        returns=returns,
    ).flatten()
    batch_size = int(batch["obs"].shape[0])

    metric_sums = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
    }
    minibatches_run = 0

    stop_update = False
    for _ in range(cfg.update_epochs):
        indices = torch.randperm(batch_size, device=batch["obs"].device)
        for start in range(0, batch_size, cfg.minibatch_size):
            mb_indices = indices[start : start + cfg.minibatch_size]
            minibatch = {key: value[mb_indices] for key, value in batch.items()}

            loss_out = ppo_loss(
                model=model,
                batch=minibatch,
                clip_coef=cfg.clip_coef,
                ent_coef=cfg.ent_coef,
                vf_coef=cfg.vf_coef,
            )

            optimizer.zero_grad(set_to_none=True)
            loss_out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            metric_sums["policy_loss"] += float(loss_out.policy_loss.item())
            metric_sums["value_loss"] += float(loss_out.value_loss.item())
            metric_sums["entropy"] += float(loss_out.entropy.item())
            metric_sums["approx_kl"] += float(loss_out.approx_kl.item())
            metric_sums["clip_fraction"] += float(loss_out.clip_fraction.item())
            minibatches_run += 1

            if cfg.target_kl is not None and loss_out.approx_kl.item() > cfg.target_kl:
                stop_update = True
                break
        if stop_update:
            break

    if minibatches_run == 0:
        raise RuntimeError("PPO update did not execute any minibatches")

    metrics = {key: value / minibatches_run for key, value in metric_sums.items()}
    metrics["explained_variance"] = explained_variance(rollout.values, returns)
    metrics["advantage_mean"] = advantage_mean
    metrics["advantage_std"] = advantage_std
    metrics["return_mean"] = return_mean
    metrics["return_std"] = return_std
    metrics["normalize_advantages"] = float(cfg.normalize_advantages)
    return metrics


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
