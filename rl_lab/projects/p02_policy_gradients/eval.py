from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_lab.core.config import load_config, parse_overrides, require
from rl_lab.projects.p02_policy_gradients.models import ActorCriticMLP


@torch.no_grad()
def evaluate(
    config_path: str | Path,
    checkpoint_path: str | Path,
    overrides: list[str] | None = None,
    episodes: int = 10,
) -> dict[str, float]:
    cfg = load_config(config_path, overrides)
    env_id = str(require(cfg, "env_id"))
    device = torch.device(str(cfg.get("device", "cpu")))

    env = gym.make(env_id)
    model_cfg = cfg.get("model", {})
    model = ActorCriticMLP(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_sizes=tuple(model_cfg.get("hidden_sizes", [64, 64])),
        activation=str(model_cfg.get("activation", "tanh")),
    ).to(device)

    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("model", payload)
    model.load_state_dict(state_dict)
    model.eval()

    returns: list[float] = []
    lengths: list[int] = []
    for ep in range(episodes):
        obs_np, _ = env.reset(seed=int(cfg.get("seed", 0)) + ep)
        done = False
        total_reward = 0.0
        length = 0
        while not done:
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            dist, _ = model.forward(obs)
            action = torch.argmax(dist.logits, dim=-1).item()
            obs_np, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            length += 1
        returns.append(total_reward)
        lengths.append(length)
    env.close()

    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_length_mean": float(np.mean(lengths)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--overrides", default="")
    args = parser.parse_args()
    print(
        evaluate(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            overrides=parse_overrides(args.overrides),
            episodes=args.episodes,
        )
    )


if __name__ == "__main__":
    main()
