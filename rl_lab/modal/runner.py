from __future__ import annotations

import modal

from rl_lab.modal.app import app, cpu_image, cuda_image, runs_volume


@app.function(
    image=cpu_image,
    cpu=2.0,
    memory=4096,
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=60 * 60 * 4,
    volumes={"/runs": runs_volume},
)
def train_remote_cpu(project: str, config_path: str, overrides: list[str], smoke: bool) -> dict:
    return _train_remote(project=project, config_path=config_path, overrides=overrides, smoke=smoke)


@app.function(
    image=cuda_image,
    gpu="A10G",
    cpu=4.0,
    memory=16384,
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=60 * 60 * 4,
    volumes={"/runs": runs_volume},
)
def train_remote_cuda(project: str, config_path: str, overrides: list[str], smoke: bool) -> dict:
    return _train_remote(project=project, config_path=config_path, overrides=overrides, smoke=smoke)


def _train_remote(project: str, config_path: str, overrides: list[str], smoke: bool) -> dict:
    from rl_lab.core.registry import get_train_fn

    train_fn = get_train_fn(project)
    result = train_fn(
        config_path=config_path,
        overrides=overrides,
        output_dir="/runs",
        remote=True,
        smoke=smoke,
    )
    return dict(result or {})


@app.local_entrypoint()
def main(
    project: str,
    config: str,
    overrides: str = "",
    smoke: bool = False,
    runtime: str = "cpu",
) -> None:
    override_list = [item.strip() for item in overrides.split(",") if item.strip()]
    if runtime == "cpu":
        result = train_remote_cpu.remote(project, config, override_list, smoke)
    elif runtime == "cuda":
        result = train_remote_cuda.remote(project, config, override_list, smoke)
    else:
        raise ValueError(f"Unknown runtime {runtime!r}; expected 'cpu' or 'cuda'")
    print(result)
