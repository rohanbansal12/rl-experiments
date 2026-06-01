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
  --smoke true
```

## Remote image dependencies

Use Modal image methods backed by uv when available, such as `Image.uv_pip_install`. Keep Modal environment construction isolated to `rl_lab/modal/`.

## Secrets

Create secrets in Modal for external services instead of hard-coding credentials.

Suggested secret names:

- `wandb-secret`
- `hf-secret`

## Volumes

The starter runner writes remote outputs to `/runs`. Configure a Modal Volume in `rl_lab/modal/app.py`.
