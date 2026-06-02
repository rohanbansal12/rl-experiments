from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


class MetricLogger:
    """Small JSONL + optional W&B metric logger.

    This intentionally avoids becoming an experiment framework. It writes a config
    snapshot and append-only metrics that are easy for agents/scripts to parse.
    """

    def __init__(
        self,
        output_dir: str | Path,
        config: dict[str, Any],
        backend: str = "wandb",
        run_name: str | None = None,
        project: str | None = None,
        entity: str | None = None,
        tags: list[str] | None = None,
        mode: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.run_name = run_name or self.output_dir.name
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.metrics_path.write_text("", encoding="utf-8")
        self.start_time = time.time()
        self._wandb_run = None

        metadata = {
            "run_name": self.run_name,
            "git_commit": git_commit_hash(),
            "created_unix": self.start_time,
        }
        write_json(self.output_dir / "config.json", config)
        write_json(self.output_dir / "metadata.json", metadata)

        if backend == "wandb":
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project=project or "rl-frontier-lab",
                    entity=entity,
                    name=self.run_name,
                    dir=str(self.output_dir),
                    config=_jsonable(config),
                    tags=tags or [],
                    mode=mode or os.environ.get("WANDB_MODE", "offline"),
                )
                self._wandb_run.config.update(
                    {
                        "run_dir": str(self.output_dir),
                        "git_commit": metadata["git_commit"],
                    },
                    allow_val_change=True,
                )
            except Exception as exc:  # pragma: no cover - optional dependency path
                raise RuntimeError(
                    "W&B backend requested but wandb is unavailable or could not initialize. "
                    "Install with `uv sync --group logging` or run via `uv run --group logging ...`."
                ) from exc
        elif backend not in {"jsonl", "none"}:
            raise ValueError(f"Unknown logging backend: {backend}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        record = {
            "step": int(step),
            "elapsed_seconds": time.time() - self.start_time,
            **_jsonable(metrics),
        }
        if self.backend != "none":
            with self.metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        if self._wandb_run is not None:
            wandb_record = dict(record)
            wandb_record.pop("step", None)
            self._wandb_run.log(wandb_record, step=step)

    def close(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    return str(value)
