from __future__ import annotations

import modal

app = modal.App("rl-frontier-lab")

BASE_PACKAGES = (
    "numpy>=1.26",
    "gymnasium[box2d,classic-control]>=0.29",
    "pyyaml>=6.0",
    "swig>=4.4.1",
    "wandb>=0.17",
)

LLM_PACKAGES = (
    "transformers>=4.44",
    "datasets>=2.20",
    "accelerate>=0.33",
)

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(*BASE_PACKAGES, *LLM_PACKAGES)
    .uv_pip_install("torch>=2.2", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir("rl_lab", remote_path="/root/rl_lab")
)

cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        *BASE_PACKAGES,
        *LLM_PACKAGES,
        # PyPI Torch wheels include CUDA dependencies on Linux. Keep this image for later GPU runs.
        "torch>=2.2",
    )
    .add_local_dir("rl_lab", remote_path="/root/rl_lab")
)

# Backward-compatible alias for imports that expect `image`.
image = cpu_image

runs_volume = modal.Volume.from_name("rl-frontier-lab-runs", create_if_missing=True)
