import json
from pathlib import Path

import gymnasium as gym
import torch

from rl_lab.core.checkpointing import load_checkpoint, save_checkpoint
from rl_lab.projects.p02_policy_gradients.eval import evaluate
from rl_lab.projects.p02_policy_gradients.models import ActorCriticMLP
from rl_lab.projects.p02_policy_gradients.train import train

CONFIG = Path("rl_lab/projects/p02_policy_gradients/configs/ppo_cartpole_debug.yaml")


def test_checkpoint_roundtrip(tmp_path) -> None:
    model = ActorCriticMLP(
        observation_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)),
        action_space=gym.spaces.Discrete(2),
        hidden_sizes=(8,),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        config={"seed": 3},
        global_step=12,
        seed=3,
        metrics={"policy_loss": 1.0},
    )

    payload = load_checkpoint(path)

    assert payload["global_step"] == 12
    assert payload["seed"] == 3
    assert payload["config"] == {"seed": 3}
    assert payload["metrics"]["policy_loss"] == 1.0
    assert set(payload["model"]) == set(model.state_dict())
    assert payload["optimizer"] is not None
    assert "git_commit" in payload


def test_evaluate_writes_output_json(tmp_path) -> None:
    env = gym.make("CartPole-v1")
    try:
        model = ActorCriticMLP(
            observation_space=env.observation_space,
            action_space=env.action_space,
            hidden_sizes=(64, 64),
        )
    finally:
        env.close()

    checkpoint_path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=None,
        config={},
        global_step=0,
        seed=0,
    )
    output_path = tmp_path / "eval_results.json"

    metrics = evaluate(
        config_path=CONFIG,
        checkpoint_path=checkpoint_path,
        overrides=["logging.mode=disabled"],
        episodes=1,
        output_path=output_path,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == metrics
    assert {"eval_return_mean", "eval_return_std", "eval_length_mean"} <= set(metrics)


def test_tiny_train_writes_checkpoints_and_eval_artifacts(tmp_path) -> None:
    result = train(
        config_path=CONFIG,
        overrides=[
            "logging.mode=disabled",
            "train.total_timesteps=16",
            "train.rollout_steps=8",
            "train.num_envs=2",
            "train.update_epochs=1",
            "train.minibatch_size=8",
            "eval.enabled=true",
            "eval.episodes=1",
        ],
        output_dir=tmp_path,
    )

    run_dir = Path(result["run_dir"])
    final_checkpoint = run_dir / "checkpoint_final.pt"
    last_checkpoint = run_dir / "checkpoint_last.pt"
    eval_path = run_dir / "eval_results.json"

    assert result["ok"] is True
    assert result["checkpoint_path"] == str(final_checkpoint)
    assert result["eval_path"] == str(eval_path)
    assert final_checkpoint.exists()
    assert last_checkpoint.exists()
    assert eval_path.exists()

    checkpoint = load_checkpoint(final_checkpoint)
    assert checkpoint["global_step"] == 16
    assert checkpoint["seed"] == 0
    assert {"eval_return_mean", "eval_return_std", "eval_length_mean"} <= set(result)


def test_longer_same_seed_run_reuses_directory_and_overwrites_metrics(tmp_path) -> None:
    common_overrides = [
        "logging.mode=disabled",
        "train.rollout_steps=4",
        "train.num_envs=2",
        "train.update_epochs=1",
        "train.minibatch_size=4",
    ]
    first = train(
        config_path=CONFIG,
        overrides=[*common_overrides, "train.total_timesteps=8"],
        output_dir=tmp_path,
    )
    first_metrics_path = Path(first["run_dir"]) / "metrics.jsonl"
    first_records = first_metrics_path.read_text(encoding="utf-8").splitlines()

    second = train(
        config_path=CONFIG,
        overrides=[*common_overrides, "train.total_timesteps=16"],
        output_dir=tmp_path,
    )
    second_metrics_path = Path(second["run_dir"]) / "metrics.jsonl"
    second_records = second_metrics_path.read_text(encoding="utf-8").splitlines()

    assert first["run_dir"] == second["run_dir"]
    assert len(first_records) == 1
    assert len(second_records) == 2
