from __future__ import annotations

from rl_lab.modal.app import app, image, runs_volume


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 4,
    volumes={"/runs": runs_volume},
)
def train_remote(project: str, config_path: str, overrides: list[str], smoke: bool) -> dict:
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
def main(project: str, config: str, overrides: str = "", smoke: bool = False) -> None:
    override_list = [item.strip() for item in overrides.split(",") if item.strip()]
    result = train_remote.remote(project, config, override_list, smoke)
    print(result)
