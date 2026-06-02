import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical

from rl_lab.projects.p02_policy_gradients import ppo as ppo_module
from rl_lab.projects.p02_policy_gradients.models import ActorCriticMLP
from rl_lab.projects.p02_policy_gradients.ppo import (
    PPOLossOutput,
    PPOUpdateConfig,
    compute_gae,
    ppo_loss,
    ppo_policy_loss,
    ppo_value_loss,
    update_ppo,
)
from rl_lab.projects.p02_policy_gradients.rollout import RolloutBatch


def test_compute_gae_known_values() -> None:
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    dones = torch.tensor([[0.0], [0.0], [1.0]])
    values = torch.tensor([[0.5], [0.5], [0.5]])
    next_value = torch.tensor([0.0])
    advantages, returns = compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        next_value=next_value,
        gamma=1.0,
        gae_lambda=1.0,
    )
    expected_returns = torch.tensor([[3.0], [2.0], [1.0]])
    expected_advantages = expected_returns - values
    torch.testing.assert_close(returns, expected_returns)
    torch.testing.assert_close(advantages, expected_advantages)


def test_compute_gae_bootstraps_from_next_value_when_not_done() -> None:
    rewards = torch.tensor([[1.0, 2.0]])
    dones = torch.tensor([[0.0, 0.0]])
    values = torch.tensor([[0.5, 1.0]])
    next_value = torch.tensor([10.0, 20.0])

    advantages, returns = compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        next_value=next_value,
        gamma=0.9,
        gae_lambda=0.95,
    )

    expected_returns = torch.tensor([[10.0, 20.0]])
    expected_advantages = expected_returns - values
    torch.testing.assert_close(returns, expected_returns)
    torch.testing.assert_close(advantages, expected_advantages)


def test_compute_gae_masks_next_value_when_done() -> None:
    rewards = torch.tensor([[1.0, 2.0]])
    dones = torch.tensor([[0.0, 1.0]])
    values = torch.tensor([[0.5, 1.0]])
    next_value = torch.tensor([10.0, 20.0])

    advantages, returns = compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        next_value=next_value,
        gamma=0.9,
        gae_lambda=0.95,
    )

    expected_returns = torch.tensor([[10.0, 2.0]])
    expected_advantages = expected_returns - values
    torch.testing.assert_close(returns, expected_returns)
    torch.testing.assert_close(advantages, expected_advantages)


def test_compute_gae_does_not_cross_done_boundaries() -> None:
    rewards = torch.tensor([[1.0], [1.0], [10.0]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    values = torch.zeros_like(rewards)
    next_value = torch.tensor([0.0])

    advantages, returns = compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        next_value=next_value,
        gamma=1.0,
        gae_lambda=1.0,
    )

    expected_returns = torch.tensor([[2.0], [1.0], [10.0]])
    torch.testing.assert_close(returns, expected_returns)
    torch.testing.assert_close(advantages, expected_returns)


def test_compute_gae_lambda_zero_matches_one_step_td_residuals() -> None:
    rewards = torch.tensor([[1.0, -1.0], [2.0, 3.0]])
    dones = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    values = torch.tensor([[0.5, -0.5], [1.5, 2.5]])
    next_value = torch.tensor([4.0, 5.0])

    advantages, returns = compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        next_value=next_value,
        gamma=0.9,
        gae_lambda=0.0,
    )

    next_values = torch.tensor([[1.5, 2.5], [4.0, 5.0]])
    expected_advantages = rewards + 0.9 * next_values * (1.0 - dones) - values
    torch.testing.assert_close(advantages, expected_advantages)
    torch.testing.assert_close(returns, expected_advantages + values)


def test_ppo_policy_loss_known_values_without_clipping() -> None:
    old_logprobs = torch.log(torch.tensor([0.5, 0.25]))
    new_logprobs = torch.log(torch.tensor([0.5, 0.5]))
    advantages = torch.tensor([1.0, -2.0])

    policy_loss, approx_kl, clip_fraction = ppo_policy_loss(
        new_logprobs=new_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        clip_coef=0.2,
    )

    ratios = torch.tensor([1.0, 2.0])
    expected_policy_loss = -torch.min(
        ratios * advantages,
        torch.clamp(ratios, 0.8, 1.2) * advantages,
    ).mean()
    expected_approx_kl = ((ratios - 1.0) - torch.log(ratios)).mean()
    expected_clip_fraction = torch.tensor(0.5)

    torch.testing.assert_close(policy_loss, expected_policy_loss)
    torch.testing.assert_close(approx_kl, expected_approx_kl)
    torch.testing.assert_close(clip_fraction, expected_clip_fraction)


def test_ppo_policy_loss_clips_positive_advantage() -> None:
    old_logprobs = torch.log(torch.tensor([0.5]))
    new_logprobs = torch.log(torch.tensor([1.0]))
    advantages = torch.tensor([3.0])

    policy_loss, _, clip_fraction = ppo_policy_loss(
        new_logprobs=new_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        clip_coef=0.2,
    )

    torch.testing.assert_close(policy_loss, torch.tensor(-3.6))
    torch.testing.assert_close(clip_fraction, torch.tensor(1.0))


def test_ppo_policy_loss_clips_negative_advantage() -> None:
    old_logprobs = torch.log(torch.tensor([1.0]))
    new_logprobs = torch.log(torch.tensor([0.5]))
    advantages = torch.tensor([-3.0])

    policy_loss, _, clip_fraction = ppo_policy_loss(
        new_logprobs=new_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        clip_coef=0.2,
    )

    torch.testing.assert_close(policy_loss, torch.tensor(2.4))
    torch.testing.assert_close(clip_fraction, torch.tensor(1.0))


def test_policy_loss_gradient_direction_tracks_advantage_sign() -> None:
    action = torch.tensor([0])

    positive_logits = torch.zeros(1, 2, requires_grad=True)
    positive_dist = Categorical(logits=positive_logits)
    positive_logprob = positive_dist.log_prob(action)
    positive_loss, _, _ = ppo_policy_loss(
        new_logprobs=positive_logprob,
        old_logprobs=positive_logprob.detach(),
        advantages=torch.tensor([1.0]),
        clip_coef=0.2,
    )
    positive_loss.backward()

    assert positive_logits.grad is not None
    assert positive_logits.grad[0, 0] < 0.0
    assert positive_logits.grad[0, 1] > 0.0

    negative_logits = torch.zeros(1, 2, requires_grad=True)
    negative_dist = Categorical(logits=negative_logits)
    negative_logprob = negative_dist.log_prob(action)
    negative_loss, _, _ = ppo_policy_loss(
        new_logprobs=negative_logprob,
        old_logprobs=negative_logprob.detach(),
        advantages=torch.tensor([-1.0]),
        clip_coef=0.2,
    )
    negative_loss.backward()

    assert negative_logits.grad is not None
    assert negative_logits.grad[0, 0] > 0.0
    assert negative_logits.grad[0, 1] < 0.0


def test_ppo_value_loss_is_half_mean_squared_error() -> None:
    new_values = torch.tensor([1.0, 3.0, -1.0])
    returns = torch.tensor([2.0, 1.0, -1.0])

    value_loss = ppo_value_loss(new_values=new_values, returns=returns)

    torch.testing.assert_close(value_loss, torch.tensor(5.0 / 6.0))


class FixedEvalModel(nn.Module):
    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del obs, actions
        return (
            torch.log(torch.tensor([0.5, 0.5])),
            torch.tensor([0.2, 0.4]),
            torch.tensor([1.0, 3.0]),
        )


def test_ppo_loss_combines_policy_value_and_entropy_terms() -> None:
    batch = {
        "obs": torch.zeros(2, 4),
        "actions": torch.tensor([0, 1]),
        "logprobs": torch.log(torch.tensor([0.5, 0.25])),
        "advantages": torch.tensor([1.0, -2.0]),
        "returns": torch.tensor([2.0, 1.0]),
    }

    out = ppo_loss(
        model=FixedEvalModel(),
        batch=batch,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
    )

    expected_policy_loss = torch.tensor(1.5)
    expected_value_loss = torch.tensor(1.25)
    expected_entropy = torch.tensor(0.3)
    expected_loss = expected_policy_loss + 0.5 * expected_value_loss - 0.01 * expected_entropy

    torch.testing.assert_close(out.loss, expected_loss)
    torch.testing.assert_close(out.policy_loss, expected_policy_loss)
    torch.testing.assert_close(out.value_loss, expected_value_loss)
    torch.testing.assert_close(out.entropy, expected_entropy)
    assert out.approx_kl.ndim == 0
    assert out.clip_fraction.ndim == 0


class GradientProbeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logprob_delta = nn.Parameter(torch.tensor(0.0))
        self.value = nn.Parameter(torch.tensor(0.0))

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del actions
        batch_size = obs.shape[0]
        return (
            self.logprob_delta.expand(batch_size),
            torch.zeros(batch_size),
            self.value.expand(batch_size),
        )


def test_ppo_loss_detaches_rollout_targets_from_gradients() -> None:
    model = GradientProbeModel()
    old_logprobs = torch.zeros(2, requires_grad=True)
    advantages = torch.tensor([1.0, 2.0], requires_grad=True)
    returns = torch.tensor([1.0, -1.0], requires_grad=True)
    batch = {
        "obs": torch.zeros(2, 4),
        "actions": torch.tensor([0, 1]),
        "logprobs": old_logprobs,
        "advantages": advantages,
        "returns": returns,
    }

    out = ppo_loss(
        model=model,
        batch=batch,
        clip_coef=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
    )
    out.loss.backward()

    assert model.logprob_delta.grad is not None
    assert model.value.grad is not None
    leaked = {
        "old_logprobs": old_logprobs.grad,
        "advantages": advantages.grad,
        "returns": returns.grad,
    }
    assert {
        name
        for name, grad in leaked.items()
        if grad is not None and torch.count_nonzero(grad).item() > 0
    } == set()


def test_ppo_update_config_from_experiment_config() -> None:
    cfg = PPOUpdateConfig.from_config(
        {
            "ppo": {
                "gamma": 0.9,
                "gae_lambda": 0.8,
                "clip_coef": 0.1,
                "ent_coef": 0.02,
                "vf_coef": 0.7,
                "target_kl": None,
                "normalize_advantages": False,
            },
            "optim": {"max_grad_norm": 0.25},
            "train": {"update_epochs": 3, "minibatch_size": 16},
        }
    )

    assert cfg == PPOUpdateConfig(
        gamma=0.9,
        gae_lambda=0.8,
        clip_coef=0.1,
        ent_coef=0.02,
        vf_coef=0.7,
        target_kl=None,
        max_grad_norm=0.25,
        update_epochs=3,
        minibatch_size=16,
        normalize_advantages=False,
    )


def test_update_ppo_returns_required_metrics_and_updates_parameters() -> None:
    torch.manual_seed(0)
    model = ActorCriticMLP(
        observation_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)),
        action_space=gym.spaces.Discrete(2),
        hidden_sizes=(8,),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rollout = RolloutBatch(
        obs=torch.zeros(4, 2, 4),
        actions=torch.zeros(4, 2, dtype=torch.long),
        logprobs=torch.zeros(4, 2),
        rewards=torch.ones(4, 2),
        dones=torch.zeros(4, 2),
        values=torch.zeros(4, 2),
    )
    before = [param.detach().clone() for param in model.parameters()]

    metrics = update_ppo(
        model=model,
        optimizer=optimizer,
        rollout=rollout,
        next_value=torch.zeros(2),
        cfg=PPOUpdateConfig(
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            target_kl=0.03,
            max_grad_norm=0.5,
            update_epochs=1,
            minibatch_size=4,
            normalize_advantages=True,
        ),
    )

    assert {
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "advantage_mean",
        "advantage_std",
        "return_mean",
        "return_std",
        "normalize_advantages",
    } <= set(metrics)
    assert metrics["normalize_advantages"] == 1.0
    assert any(
        not torch.equal(param, old_param)
        for param, old_param in zip(model.parameters(), before, strict=True)
    )


class MinibatchRecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))


def test_update_ppo_minibatches_cover_all_rollout_elements_once_per_epoch(monkeypatch) -> None:
    seen_minibatches: list[list[int]] = []

    def record_loss(
        model: MinibatchRecordingModel,
        batch: dict[str, torch.Tensor],
        clip_coef: float,
        ent_coef: float,
        vf_coef: float,
    ) -> PPOLossOutput:
        del clip_coef, ent_coef, vf_coef
        seen_minibatches.append(batch["obs"].reshape(-1).int().tolist())
        zero = model.anchor * 0.0
        return PPOLossOutput(
            loss=zero,
            policy_loss=zero.detach(),
            value_loss=zero.detach(),
            entropy=zero.detach(),
            approx_kl=zero.detach(),
            clip_fraction=zero.detach(),
        )

    monkeypatch.setattr(ppo_module, "ppo_loss", record_loss)
    torch.manual_seed(0)
    model = MinibatchRecordingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    rollout = RolloutBatch(
        obs=torch.arange(6, dtype=torch.float32).reshape(3, 2, 1),
        actions=torch.zeros(3, 2, dtype=torch.long),
        logprobs=torch.zeros(3, 2),
        rewards=torch.ones(3, 2),
        dones=torch.zeros(3, 2),
        values=torch.zeros(3, 2),
    )

    update_ppo(
        model=model,
        optimizer=optimizer,
        rollout=rollout,
        next_value=torch.zeros(2),
        cfg=PPOUpdateConfig(
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            target_kl=None,
            max_grad_norm=0.5,
            update_epochs=2,
            minibatch_size=2,
            normalize_advantages=True,
        ),
    )

    minibatches_per_epoch = 3
    assert len(seen_minibatches) == 6
    for epoch_start in range(0, len(seen_minibatches), minibatches_per_epoch):
        epoch_ids = [
            item
            for minibatch in seen_minibatches[epoch_start : epoch_start + minibatches_per_epoch]
            for item in minibatch
        ]
        assert sorted(epoch_ids) == list(range(6))
