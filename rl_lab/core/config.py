from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


Config = dict[str, Any]


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config and apply dot-path overrides.

    Override examples:
        train.total_timesteps=10000
        logging.backend=tensorboard
        ppo.clip_coef=0.1
        smoke=true
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise TypeError(f"Expected top-level mapping in config: {path}")

    cfg = copy.deepcopy(cfg)
    for override in overrides or []:
        if not override:
            continue
        apply_override(cfg, override)
    return cfg


def parse_overrides(raw: str | list[str] | None) -> list[str]:
    """Parse CLI override strings.

    Supports either comma-separated strings or lists. Empty values are ignored.
    """

    if raw is None:
        return []
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            out.extend(parse_overrides(item))
        return out
    return [item.strip() for item in raw.split(",") if item.strip()]


def apply_override(cfg: Config, override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must have form key.path=value, got: {override!r}")

    key_path, raw_value = override.split("=", 1)
    keys = [k.strip() for k in key_path.split(".") if k.strip()]
    if not keys:
        raise ValueError(f"Invalid override key path: {override!r}")

    value = yaml.safe_load(raw_value)
    cursor: Config = cfg
    for key in keys[:-1]:
        child = cursor.get(key)
        if child is None:
            child = {}
            cursor[key] = child
        if not isinstance(child, dict):
            raise TypeError(f"Cannot set {override!r}: {key!r} is not a mapping")
        cursor = child
    cursor[keys[-1]] = value


def require(cfg: Config, *keys: str) -> Any:
    """Fetch a nested config value or raise a helpful KeyError."""

    cursor: Any = cfg
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        if not isinstance(cursor, dict) or key not in cursor:
            dotted = ".".join(traversed)
            raise KeyError(f"Missing required config key: {dotted}")
        cursor = cursor[key]
    return cursor


def config_hash(cfg: Config) -> str:
    """Stable short hash for experiment tracking."""

    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
