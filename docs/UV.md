# uv Notes

This repo uses `uv` as the single tool for local Python environment management, dependency management, locking, and command execution.

## First setup

```bash
uv venv --python 3.11
uv sync --group dev --group logging --group modal
```

`uv sync` installs the project and selected dependency groups into `.venv`. You can activate the environment for editor integration, but normal commands should use `uv run` or Make targets.

```bash
source .venv/bin/activate  # optional
```

## Daily commands

```bash
uv run --group dev --group logging pytest -q
uv run --group dev ruff check rl_lab scripts
uv run --group logging python -m rl_lab.projects.p02_policy_gradients.train \
  --config rl_lab/projects/p02_policy_gradients/configs/ppo_cartpole_debug.yaml \
  --smoke
```

## Dependency groups

- Base project dependencies: core runtime for Project 2.
- `dev`: pytest, ruff, and development tooling.
- `logging`: TensorBoard, W&B, and experiment logging adapters.
- `modal`: Modal client for remote execution.
- `llm`: heavier dependencies for future LLM/RLHF projects.

## Adding dependencies

```bash
uv add package-name
uv add --group dev pytest
uv add --group logging tensorboard
uv add --group modal modal
uv add --group llm transformers
uv lock
```

Do not create `requirements.txt` unless the user explicitly asks for one.
