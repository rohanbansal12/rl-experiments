from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from rl_lab.core.logging import git_commit_hash


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    global_step: int,
    seed: int,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save a training checkpoint with enough metadata to audit a run."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
        "global_step": int(global_step),
        "seed": int(seed),
        "git_commit": git_commit_hash(),
        "metrics": metrics or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint payload saved by `save_checkpoint`."""

    return torch.load(Path(path), map_location=map_location)
