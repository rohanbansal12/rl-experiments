# Modal Notes

The repo uses Modal only as a thin remote execution layer.

Local training and remote training should call the same project-level `train(...)` function. Modal changes the hardware and output path; it should not change algorithm code.

## Local Modal client commands

Run Modal through `uv`:

```bash
make modal-smoke PROJECT=p02_policy_gradients CONFIG=configs/ppo_cartpole_debug.yaml
make modal-train PROJECT=p02_policy_gradients CONFIG=configs/ppo_cartpole_debug.yaml
```

Equivalent direct command:

```bash
uv run --group modal modal run -m rl_lab.modal.runner \
  --project p02_policy_gradients \
  --config rl_lab/projects/p02_policy_gradients/configs/ppo_cartpole_debug.yaml \
  --runtime cpu \
  --smoke
```

## Remote image dependencies

Use Modal image methods backed by uv when available, such as `Image.uv_pip_install`. Keep Modal environment construction isolated to `rl_lab/modal/`.

## CPU resources

Project 2 runs CPU-only on Modal by default. The CPU remote train function requests:

- `cpu=2.0`, meaning two physical CPU cores.
- `memory=4096`, meaning 4 GiB.

Use `RUNTIME=cuda` to choose the CUDA/GPU image and A10G function:

```bash
make modal-train PROJECT=p02_policy_gradients \
  CONFIG=configs/ppo_cartpole_debug.yaml \
  RUNTIME=cuda \
  OVERRIDES="logging.mode=online"
```

Adjust CPU/GPU resources in `rl_lab/modal/runner.py` after checking observed usage in the Modal dashboard.

For Project 5, the Modal images include the LLM dependency group packages
(`transformers`, `datasets`, and `accelerate`) so the same generic runner can
launch the small RLHF pipeline:

```bash
make modal-train PROJECT=p05_mini_rlhf \
  CONFIG=configs/rlhf_imdb_debug.yaml \
  RUNTIME=cuda \
  OVERRIDES="logging.mode=online"
```

## Secrets

Create secrets in Modal for external services instead of hard-coding credentials.

Suggested secret names:

- `wandb-secret`
- `hf-secret`

The runner attaches `wandb-secret`, so create that secret before running Modal jobs. For online W&B logging, include `WANDB_API_KEY` in the secret and run with `OVERRIDES="logging.mode=online"`.

## Volumes

The starter runner writes remote outputs to `/runs`. Configure a Modal Volume in `rl_lab/modal/app.py`.
