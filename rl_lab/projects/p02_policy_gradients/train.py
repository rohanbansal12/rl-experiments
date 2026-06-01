from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from rl_lab.core.config import config_hash, load_config, parse_overrides, require
from rl_lab.core.logging import MetricLogger
from rl_lab.core.seeding import set_seed
from rl_lab.projects.p02_policy_gradients.models import ActorCriticMLP
from rl_lab.projects.p02_policy_gradients.ppo import update_ppo
from rl_lab.projects.p02_policy_gradients.rollout import RolloutStorage


def train(
    config_path: str | Path,
    overrides: list[str] | None = None,
    output_dir: str | Path | None = None,
    remote: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    """Project-level train entrypoint used by both local CLI and Modal."""

    cfg = load_config(config_path, overrides)
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    env_id = str(require(cfg, "env_id"))
    train_cfg = require(cfg, "train")
    logging_cfg = cfg.get("logging", {})
    device = torch.device(str(cfg.get("device", "cpu")))

    base_output = Path(output_dir or "experiments/runs")
    run_name = str(logging_cfg.get("run_name", f"{env_id}_seed{seed}"))
    run_dir = base_output / f"{run_name}_{config_hash(cfg)}"
    logger = MetricLogger(
        output_dir=run_dir,
        config={**cfg, "config_path": str(config_path), "remote": remote},
        backend=str(logging_cfg.get("backend", "jsonl")),
        run_name=run_name,
    )

    num_envs = int(train_cfg.get("num_envs", 1))
    rollout_steps = int(train_cfg.get("rollout_steps", 128))
    envs = make_vector_env(env_id=env_id, num_envs=num_envs, seed=seed)

    model_cfg = cfg.get("model", {})
    model = ActorCriticMLP(
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        hidden_sizes=tuple(model_cfg.get("hidden_sizes", [64, 64])),
        activation=str(model_cfg.get("activation", "tanh")),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(require(cfg, "optim", "lr")))

    obs_np, _ = envs.reset(seed=seed)
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    storage = RolloutStorage(
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        obs_shape=tuple(envs.single_observation_space.shape),
        device=device,
    )

    global_step = 0
    start_time = time.time()
    try:
        if smoke:
            obs, smoke_metrics = collect_rollout(
                envs=envs,
                model=model,
                storage=storage,
                obs=obs,
                device=device,
            )
            global_step += rollout_steps * num_envs
            logger.log({**smoke_metrics, "smoke": True}, step=global_step)
            return {"ok": True, "mode": "smoke", "run_dir": str(run_dir), **smoke_metrics}

        total_timesteps = int(train_cfg.get("total_timesteps", 0))
        if total_timesteps <= 0:
            raise ValueError("train.total_timesteps must be positive")

        while global_step < total_timesteps:
            obs, rollout_metrics = collect_rollout(
                envs=envs,
                model=model,
                storage=storage,
                obs=obs,
                device=device,
            )
            global_step += rollout_steps * num_envs

            with torch.no_grad():
                _, next_value = model.forward(obs)

            update_metrics = update_ppo(
                model=model,
                optimizer=optimizer,
                rollout=storage.as_batch(),
                next_value=next_value,
                cfg=cfg,
            )
            storage.reset()

            elapsed = max(time.time() - start_time, 1e-8)
            metrics = {
                **rollout_metrics,
                **update_metrics,
                "steps_per_second": global_step / elapsed,
            }
            logger.log(metrics, step=global_step)

        return {"ok": True, "mode": "train", "run_dir": str(run_dir), "global_step": global_step}
    finally:
        logger.close()
        envs.close()


def collect_rollout(
    envs: gym.vector.VectorEnv,
    model: ActorCriticMLP,
    storage: RolloutStorage,
    obs: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Collect one on-policy rollout.

    This function intentionally does not compute advantages or update the model.
    """

    storage.reset()
    episode_returns: list[float] = []
    episode_lengths: list[float] = []

    for _ in range(storage.rollout_steps):
        with torch.no_grad():
            action, logprob, value = model.act(obs)

        next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(action.cpu().numpy())
        done_np = np.logical_or(terminated_np, truncated_np)

        reward = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
        done = torch.as_tensor(done_np.astype(np.float32), dtype=torch.float32, device=device)
        storage.add(obs, action, logprob, reward, done, value)

        # Gymnasium vector envs expose final episode info under different keys depending on wrappers.
        # This captures common vectorized RecordEpisodeStatistics outputs if added later.
        if isinstance(infos, dict) and "episode" in infos:
            ep_info = infos["episode"]
            if "r" in ep_info:
                episode_returns.extend(np.asarray(ep_info["r"]).reshape(-1).astype(float).tolist())
            if "l" in ep_info:
                episode_lengths.extend(np.asarray(ep_info["l"]).reshape(-1).astype(float).tolist())

        obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)

    metrics = {
        "rollout_reward_mean": float(storage.rewards.mean().item()),
        "rollout_done_fraction": float(storage.dones.mean().item()),
        "episodic_return_mean": float(np.mean(episode_returns)) if episode_returns else float("nan"),
        "episodic_length_mean": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
    }
    return obs, metrics


def make_vector_env(env_id: str, num_envs: int, seed: int) -> gym.vector.VectorEnv:
    def thunk(rank: int):
        def _make() -> gym.Env:
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed + rank)
            env.observation_space.seed(seed + rank)
            return env

        return _make

    return gym.vector.SyncVectorEnv([thunk(i) for i in range(num_envs)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overrides", default="")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    result = train(
        config_path=args.config,
        overrides=parse_overrides(args.overrides),
        output_dir=args.output_dir,
        remote=False,
        smoke=args.smoke,
    )
    print(result)


if __name__ == "__main__":
    main()
