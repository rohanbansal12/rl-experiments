# AGENTS.md

## Role

You are assisting with an RL research-learning repository. Your default job is to scaffold infrastructure, tests, configs, logging, plotting, documentation, and Modal execution.

Do **not** implement or rewrite core algorithm math unless the user explicitly asks.

## Protected learning files

The user wants to personally implement the core methods. Treat these files and symbols as protected unless explicitly instructed otherwise:

- `rl_lab/projects/*/ppo.py`
- `rl_lab/projects/*/dqn.py`
- `rl_lab/projects/*/sac.py`
- `rl_lab/projects/*/grpo.py`
- `rl_lab/projects/*/losses.py`
- `rl_lab/projects/*/mcts.py`
- any function, class, or block marked `TODO(user)`

You may add type hints, tests, shape assertions, docstrings, and comments around these files, but do not fill in algorithmic TODOs.

## Environment and dependency rules

Use `uv` for all Python environment and dependency work wherever possible.

- Create the local virtual environment with `uv venv --python 3.11`.
- Sync dependencies with `uv sync ...`.
- Run commands with `uv run ...` or the Makefile targets.
- Add dependencies with `uv add ...`, not by hand-editing requirements files.
- Add dev tools with `uv add --group dev ...`.
- Add logging tools with `uv add --group logging ...`.
- Add Modal tools with `uv add --group modal ...`.
- Add LLM-only tools with `uv add --group llm ...`.
- Update the lockfile with `uv lock` after dependency changes.
- Do not use `pip`, `python -m venv`, Conda, Poetry, or requirements files unless the user explicitly asks.
- Do not commit `.venv/`.

## Agent-safe tasks

You may freely work on:

- config schemas and config loading
- CLI wrappers
- Modal runners
- logging integrations
- checkpoint save/load helpers
- plotting scripts
- smoke tests
- unit tests for math functions
- README/SPEC updates
- environment wrappers
- dataset download/cache scripts
- experiment table generation
- code formatting and import cleanup

## Commands

Use these commands instead of inventing alternatives:

```bash
make venv
make sync
make format
make lint
make test
make smoke PROJECT=p02_policy_gradients CONFIG=configs/ppo_cartpole_debug.yaml
make train-local PROJECT=p02_policy_gradients CONFIG=configs/ppo_cartpole_debug.yaml
make modal-smoke PROJECT=p02_policy_gradients CONFIG=configs/ppo_cartpole_debug.yaml
make modal-train PROJECT=p02_policy_gradients CONFIG=configs/ppo_cartpole_debug.yaml
```

These Make targets call `uv` internally. Direct Python commands should be run as `uv run ...`.

## Coding style

- Python 3.11+.
- Use PyTorch for neural nets.
- Prefer small, explicit functions.
- Keep algorithm update steps readable, even if less abstract.
- Add shape comments for tensors in RL losses.
- Avoid clever framework abstractions.
- Avoid global mutable state except simple registries.
- All randomness must flow through explicit seeds.
- Keep Modal code out of algorithm files.

## Testing expectations

For every algorithm project, maintain:

- shape tests
- deterministic tiny-env tests where possible
- loss sanity tests
- serialization/checkpoint tests
- one CPU smoke test

Do not claim a training run is successful unless there is a logged metric, saved config, seed, git commit hash, and evaluation output.

## Modal rules

- Modal code lives only under `rl_lab/modal/`.
- Do not add Modal decorators inside algorithm files.
- Remote jobs must call the same train/eval functions as local jobs.
- All remote outputs go under `/runs`.
- All local outputs go under `experiments/runs` unless overridden.
- Use Modal Secrets for external credentials such as W&B or Hugging Face.
- Do not hard-code secrets or tokens.
- Use Modal image methods backed by uv when available, e.g. `Image.uv_pip_install`, rather than `pip_install`.

## Dependency rules

- Do not add new dependencies unless justified in the final summary.
- Prefer standard library, numpy, torch, gymnasium, PyYAML, pytest, ruff, tensorboard, wandb, transformers, datasets, accelerate, and modal.
- Keep dependencies optional/grouped when they are only used by later projects.

## When uncertain

Prefer adding a TODO, a test, or a narrow helper over making a broad architectural decision. Ask before changing protected learning files.
