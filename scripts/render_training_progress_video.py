from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from rl_lab.core.config import load_config, parse_overrides, require
from rl_lab.core.seeding import set_seed
from rl_lab.projects.p02_policy_gradients.models import ActorCriticMLP
from rl_lab.projects.p02_policy_gradients.ppo import PPOUpdateConfig, update_ppo
from rl_lab.projects.p02_policy_gradients.rollout import RolloutStorage
from rl_lab.projects.p02_policy_gradients.train import collect_rollout, make_vector_env


@torch.no_grad()
def render_policy_clip(
    *,
    model: ActorCriticMLP,
    cfg: dict[str, Any],
    seed: int,
    global_step: int,
    max_steps: int,
    deterministic: bool,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env_id = str(require(cfg, "env_id"))
    device = next(model.parameters()).device
    env = gym.make(env_id, render_mode="rgb_array")
    frames: list[np.ndarray] = []
    total_reward = 0.0
    length = 0
    try:
        obs_np, _ = env.reset(seed=seed)
        frames.append(_annotate_frame(env.render(), global_step=global_step, length=0, reward=0.0))
        done = False
        while not done and length < max_steps:
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            dist, _ = model.forward(obs)
            action = (
                torch.argmax(dist.logits, dim=-1).item() if deterministic else dist.sample().item()
            )
            obs_np, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            length += 1
            frames.append(
                _annotate_frame(
                    env.render(),
                    global_step=global_step,
                    length=length,
                    reward=total_reward,
                )
            )
            done = bool(terminated or truncated)
    finally:
        env.close()

    metadata = {
        "global_step": int(global_step),
        "seed": int(seed),
        "return": float(total_reward),
        "length": int(length),
        "deterministic": bool(deterministic),
    }
    return frames, metadata


def render_training_progress_video(
    *,
    config_path: str | Path,
    output_path: str | Path,
    overrides: list[str] | None = None,
    snapshot_steps: list[int],
    eval_seed: int | None,
    clip_max_steps: int,
    fps: int,
    speedup: int,
    transition_frames: int,
    hold_frames: int,
    deterministic: bool,
) -> dict[str, Any]:
    cfg = load_config(config_path, overrides)
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    env_id = str(require(cfg, "env_id"))
    train_cfg = require(cfg, "train")
    update_cfg = PPOUpdateConfig.from_config(cfg)
    device = torch.device(str(cfg.get("device", "cpu")))

    num_envs = int(train_cfg.get("num_envs", 1))
    rollout_steps = int(train_cfg.get("rollout_steps", 128))
    envs = make_vector_env(env_id=env_id, num_envs=num_envs, seed=seed)
    try:
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

        targets = sorted(set(int(step) for step in snapshot_steps))
        if not targets or targets[0] != 0:
            targets.insert(0, 0)
        final_target = targets[-1]

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        eval_base_seed = int(seed if eval_seed is None else eval_seed)
        all_frames: list[np.ndarray] = []
        snapshots: list[dict[str, Any]] = []
        next_snapshot_idx = 0
        global_step = 0

        while next_snapshot_idx < len(targets):
            target = targets[next_snapshot_idx]
            if global_step >= target:
                frames, metadata = render_policy_clip(
                    model=model,
                    cfg=cfg,
                    seed=eval_base_seed,
                    global_step=global_step,
                    max_steps=clip_max_steps,
                    deterministic=deterministic,
                )
                snapshots.append(metadata)
                _append_clip(
                    all_frames,
                    _subsample_frames(frames, speedup=speedup),
                    transition_frames=transition_frames,
                    hold_frames=hold_frames,
                )
                next_snapshot_idx += 1
                continue

            obs, _ = collect_rollout(
                envs=envs,
                model=model,
                storage=storage,
                obs=obs,
                device=device,
            )
            global_step += rollout_steps * num_envs
            with torch.no_grad():
                _, next_value = model.forward(obs)
            update_ppo(
                model=model,
                optimizer=optimizer,
                rollout=storage.as_batch(),
                next_value=next_value,
                cfg=update_cfg,
            )
            storage.reset()

            if global_step >= final_target and next_snapshot_idx >= len(targets):
                break

        with imageio.get_writer(output, fps=fps, codec="libx264", macro_block_size=1) as writer:
            for frame in all_frames:
                writer.append_data(frame)

        metadata = {
            "env_id": env_id,
            "train_seed": seed,
            "eval_seed": eval_base_seed,
            "output_path": str(output),
            "config_path": str(config_path),
            "snapshot_steps": targets,
            "clip_max_steps": int(clip_max_steps),
            "fps": int(fps),
            "speedup": int(speedup),
            "transition_frames": int(transition_frames),
            "hold_frames": int(hold_frames),
            "deterministic": bool(deterministic),
            "snapshots": snapshots,
        }
        metadata_path = output.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return metadata
    finally:
        envs.close()


def _append_clip(
    output_frames: list[np.ndarray],
    clip_frames: list[np.ndarray],
    *,
    transition_frames: int,
    hold_frames: int,
) -> None:
    if not clip_frames:
        return
    if not output_frames:
        output_frames.extend(clip_frames)
    else:
        blend_count = min(max(0, transition_frames), len(output_frames), len(clip_frames))
        if blend_count:
            previous = output_frames[-1]
            for idx in range(blend_count):
                alpha = (idx + 1) / (blend_count + 1)
                output_frames.append(_blend_frames(previous, clip_frames[idx], alpha))
            output_frames.extend(clip_frames[blend_count:])
        else:
            output_frames.extend(clip_frames)
    if hold_frames > 0:
        output_frames.extend([clip_frames[-1]] * hold_frames)


def _blend_frames(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return ((1.0 - alpha) * a.astype(np.float32) + alpha * b.astype(np.float32)).astype(np.uint8)


def _subsample_frames(frames: list[np.ndarray], speedup: int) -> list[np.ndarray]:
    stride = max(1, int(speedup))
    sampled = frames[::stride]
    if sampled[-1] is not frames[-1]:
        sampled.append(frames[-1])
    return sampled


def _annotate_frame(
    frame: np.ndarray,
    *,
    global_step: int,
    length: int,
    reward: float,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    label = f"train step {global_step:,} | episode step {length} | return {reward:.0f}"
    margin = 8
    bbox = draw.textbbox((margin, margin), label, font=font)
    draw.rectangle(
        (bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3),
        fill=(255, 255, 255),
    )
    draw.text((margin, margin), label, fill=(0, 0, 0), font=font)
    return np.asarray(image)


def _parse_snapshot_steps(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO and render a sped-up progress video.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overrides", default="")
    parser.add_argument("--snapshot-steps", default="0,8192,32768,131072,262144,501760")
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--clip-max-steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--speedup", type=int, default=8)
    parser.add_argument("--transition-frames", type=int, default=8)
    parser.add_argument("--hold-frames", type=int, default=0)
    parser.add_argument(
        "--sample", action="store_true", help="Sample actions instead of using argmax."
    )
    args = parser.parse_args()

    metadata = render_training_progress_video(
        config_path=args.config,
        output_path=args.output,
        overrides=parse_overrides(args.overrides),
        snapshot_steps=_parse_snapshot_steps(args.snapshot_steps),
        eval_seed=args.eval_seed,
        clip_max_steps=args.clip_max_steps,
        fps=args.fps,
        speedup=args.speedup,
        transition_frames=args.transition_frames,
        hold_frames=args.hold_frames,
        deterministic=not args.sample,
    )
    print(metadata)


if __name__ == "__main__":
    main()
