import json
import subprocess
import sys
from pathlib import Path

import torch

from rl_lab.core.registry import get_train_fn
from rl_lab.projects.p02_policy_gradients import train as train_module
from rl_lab.projects.p02_policy_gradients.models import ActorCriticMLP
from rl_lab.projects.p02_policy_gradients.rollout import RolloutStorage

CONFIG = Path("rl_lab/projects/p02_policy_gradients/configs/ppo_cartpole_debug.yaml")


def test_smoke_mode_does_not_call_update_ppo(monkeypatch, tmp_path) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("smoke mode should not call update_ppo")

    monkeypatch.setattr(train_module, "update_ppo", fail_if_called)

    result = train_module.train(
        config_path=CONFIG,
        overrides=["train.rollout_steps=4", "train.num_envs=2", "logging.mode=disabled"],
        output_dir=tmp_path,
        smoke=True,
    )

    assert result["ok"] is True
    assert result["mode"] == "smoke"


def test_collect_rollout_fills_storage_and_returns_metrics() -> None:
    device = torch.device("cpu")
    envs = train_module.make_vector_env(env_id="CartPole-v1", num_envs=2, seed=123)
    try:
        model = ActorCriticMLP(
            observation_space=envs.single_observation_space,
            action_space=envs.single_action_space,
            hidden_sizes=(8,),
        ).to(device)
        storage = RolloutStorage(
            rollout_steps=4,
            num_envs=2,
            obs_shape=tuple(envs.single_observation_space.shape),
            device=device,
        )
        obs_np, _ = envs.reset(seed=123)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

        next_obs, metrics = train_module.collect_rollout(
            envs=envs,
            model=model,
            storage=storage,
            obs=obs,
            device=device,
        )

        batch = storage.as_batch()
        assert storage.step == 4
        assert batch.obs.shape == (4, 2, *envs.single_observation_space.shape)
        assert batch.actions.shape == (4, 2)
        assert batch.logprobs.shape == (4, 2)
        assert batch.values.shape == (4, 2)
        assert next_obs.shape == (2, *envs.single_observation_space.shape)
        assert set(metrics) == {
            "rollout_reward_mean",
            "rollout_done_fraction",
            "episodic_return_mean",
            "episodic_return_std",
            "episodic_length_mean",
            "episodic_length_std",
        }
    finally:
        envs.close()


def test_masked_episode_info_ignores_nonterminal_placeholders() -> None:
    values = train_module._masked_info_values(
        episode_info={
            "r": [14.0, 0.0, 22.0],
            "_r": [True, False, True],
        },
        key="r",
        fallback_mask=None,
    )

    assert values == [14.0, 22.0]


def test_run_identity_config_ignores_total_timesteps_only() -> None:
    base = {
        "seed": 0,
        "train": {"total_timesteps": 1000, "rollout_steps": 128},
        "ppo": {"clip_coef": 0.2},
    }
    longer = {
        "seed": 0,
        "train": {"total_timesteps": 5000, "rollout_steps": 128},
        "ppo": {"clip_coef": 0.2},
    }
    changed_hparams = {
        "seed": 0,
        "train": {"total_timesteps": 5000, "rollout_steps": 256},
        "ppo": {"clip_coef": 0.2},
    }

    assert train_module.run_identity_config(base) == train_module.run_identity_config(longer)
    assert train_module.run_identity_config(base) != train_module.run_identity_config(
        changed_hparams
    )


def test_project_registry_resolves_p02_train() -> None:
    assert get_train_fn("p02_policy_gradients") is train_module.train


def test_cli_smoke_writes_artifacts(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rl_lab.projects.p02_policy_gradients.train",
            "--config",
            str(CONFIG),
            "--output-dir",
            str(tmp_path),
            "--smoke",
            "--overrides",
            "logging.mode=disabled,train.rollout_steps=4,train.num_envs=2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "'mode': 'smoke'" in result.stdout

    run_dirs = list(tmp_path.glob("p02_cartpole_debug_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metadata.json").exists()

    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["step"] == 8
    assert records[0]["smoke"] is True
