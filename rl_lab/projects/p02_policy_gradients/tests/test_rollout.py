import torch

from rl_lab.projects.p02_policy_gradients.rollout import RolloutStorage


def test_rollout_storage_shapes_and_flatten() -> None:
    storage = RolloutStorage(
        rollout_steps=3,
        num_envs=2,
        obs_shape=(4,),
        device=torch.device("cpu"),
    )
    for _ in range(3):
        storage.add(
            obs=torch.zeros(2, 4),
            action=torch.zeros(2, dtype=torch.long),
            logprob=torch.zeros(2),
            reward=torch.ones(2),
            done=torch.zeros(2),
            value=torch.zeros(2),
        )
    batch = storage.as_batch()
    flat = batch.flatten()
    assert flat["obs"].shape == (6, 4)
    assert flat["actions"].shape == (6,)
    assert flat["rewards"].shape == (6,)
