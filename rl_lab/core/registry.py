from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

TrainFn = Callable[..., Any]


def get_train_fn(project: str) -> TrainFn:
    """Import a project's train function by project directory name."""

    module = importlib.import_module(f"rl_lab.projects.{project}.train")
    train_fn = getattr(module, "train", None)
    if train_fn is None:
        raise AttributeError(f"Project {project!r} does not expose train(...)")
    return train_fn
