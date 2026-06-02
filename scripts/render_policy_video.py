from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import torch

from rl_lab.core.checkpointing import load_checkpoint
from rl_lab.core.config import load_config, parse_overrides, require
from rl_lab.core.seeding import set_seed
from rl_lab.projects.p02_policy_gradients.models import ActorCriticMLP


@torch.no_grad()
def render_policy_video(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    overrides: list[str] | None = None,
    seed: int | None = None,
    max_steps: int | None = None,
    fps: int | None = None,
    deterministic: bool = True,
) -> dict[str, Any]:
    cfg = load_config(config_path, overrides)
    env_id = str(require(cfg, "env_id"))
    run_seed = int(cfg.get("seed", 0) if seed is None else seed)
    set_seed(run_seed)

    # Pygame can render rgb arrays headlessly when SDL uses the dummy driver.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    device = torch.device(str(cfg.get("device", "cpu")))
    env = gym.make(env_id, render_mode="rgb_array")
    try:
        model_cfg = cfg.get("model", {})
        model = ActorCriticMLP(
            observation_space=env.observation_space,
            action_space=env.action_space,
            hidden_sizes=tuple(model_cfg.get("hidden_sizes", [64, 64])),
            activation=str(model_cfg.get("activation", "tanh")),
        ).to(device)

        payload = load_checkpoint(checkpoint_path, map_location=device)
        state_dict = payload.get("model", payload)
        model.load_state_dict(state_dict)
        model.eval()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        video_fps = int(fps or env.metadata.get("render_fps", 30))
        step_limit = int(max_steps or getattr(env.spec, "max_episode_steps", 1000) or 1000)

        obs_np, _ = env.reset(seed=run_seed)
        frame = env.render()
        total_reward = 0.0
        length = 0

        with imageio.get_writer(
            output, fps=video_fps, codec="libx264", macro_block_size=1
        ) as writer:
            writer.append_data(frame)
            done = False
            while not done and length < step_limit:
                obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                dist, _ = model.forward(obs)
                if deterministic:
                    action = torch.argmax(dist.logits, dim=-1).item()
                else:
                    action = dist.sample().item()
                obs_np, reward, terminated, truncated, _ = env.step(action)
                frame = env.render()
                writer.append_data(frame)
                total_reward += float(reward)
                length += 1
                done = bool(terminated or truncated)

        metadata = {
            "env_id": env_id,
            "seed": run_seed,
            "checkpoint_path": str(checkpoint_path),
            "output_path": str(output),
            "deterministic": deterministic,
            "return": total_reward,
            "length": length,
            "fps": video_fps,
        }
        metadata_path = output.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return metadata
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a trained policy checkpoint to MP4.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overrides", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument(
        "--sample", action="store_true", help="Sample actions instead of using argmax."
    )
    args = parser.parse_args()

    metadata = render_policy_video(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        overrides=parse_overrides(args.overrides),
        seed=args.seed,
        max_steps=args.max_steps,
        fps=args.fps,
        deterministic=not args.sample,
    )
    print(metadata)


if __name__ == "__main__":
    main()
