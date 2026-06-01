from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any


class MetricLogger:
    """Small JSONL + optional TensorBoard metric logger.

    This intentionally avoids becoming an experiment framework. It writes a config
    snapshot and append-only metrics that are easy for agents/scripts to parse.
    """

    def __init__(
        self,
        output_dir: str | Path,
        config: dict[str, Any],
        backend: str = "jsonl",
        run_name: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.run_name = run_name or self.output_dir.name
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.start_time = time.time()
        self._tb = None

        metadata = {
            "run_name": self.run_name,
            "git_commit": git_commit_hash(),
            "created_unix": self.start_time,
        }
        write_json(self.output_dir / "config.json", config)
        write_json(self.output_dir / "metadata.json", metadata)

        if backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(self.output_dir / "tb"))
            except Exception as exc:  # pragma: no cover - optional dependency path
                raise RuntimeError(
                    "TensorBoard backend requested but tensorboard is unavailable. "
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
        if self._tb is not None:
            for key, value in metrics.items():
                if isinstance(value, int | float):
                    self._tb.add_scalar(key, value, step)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()


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
