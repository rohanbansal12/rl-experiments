# Logging

This repo uses W&B for experiment logging and keeps local JSONL artifacts for quick inspection, plotting, and smoke-test assertions.

## Local modes

The debug configs default to:

```yaml
logging:
  backend: wandb
  mode: offline
```

Use `offline` for local development when you do not want network logging. Use `disabled` in tests. Use `online` for real experiment runs after setting `WANDB_API_KEY`.

```bash
make train-local PROJECT=p02_policy_gradients \
  CONFIG=configs/ppo_cartpole_debug.yaml \
  OVERRIDES="logging.mode=online"
```

## Modal

Remote jobs should use the same project `train(...)` entrypoint as local jobs. For online W&B logging on Modal, create a Modal Secret named `wandb-secret` containing `WANDB_API_KEY`, then run with `logging.mode=online`.

```bash
uv run --group modal modal secret create wandb-secret WANDB_API_KEY=...
make modal-train PROJECT=p02_policy_gradients \
  CONFIG=configs/ppo_cartpole_debug.yaml \
  OVERRIDES="logging.mode=online"
```

If `logging.mode=offline`, W&B writes local offline run files under the run directory but does not upload them. If `logging.mode=disabled`, W&B is disabled and only local JSONL/config/metadata artifacts are written.

Do not hard-code W&B keys in configs, code, or docs.

## Local artifacts

Each run directory contains:

- `config.json`: resolved config snapshot.
- `metadata.json`: run name, git commit, and creation timestamp.
- `metrics.jsonl`: append-only metrics records.
- `wandb/`: W&B local run files when W&B is enabled.
