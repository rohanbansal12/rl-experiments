import pytest
import torch

from rl_lab.projects.p02_policy_gradients.ppo import compute_gae


@pytest.mark.xfail(reason="TODO(user): implement compute_gae")
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
