from __future__ import annotations

import modal

app = modal.App("rl-frontier-lab")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "numpy>=1.26",
        "torch>=2.2",
        "gymnasium[classic-control]>=0.29",
        "pyyaml>=6.0",
        "tensorboard>=2.16",
    )
    .add_local_dir("rl_lab", remote_path="/root/rl_lab")
)

runs_volume = modal.Volume.from_name("rl-frontier-lab-runs", create_if_missing=True)
