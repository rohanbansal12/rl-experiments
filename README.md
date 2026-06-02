# RL Frontier Lab Starter

A small, agent-friendly starter repo for learning reinforcement learning methods relevant to modern frontier AI/ML research.

This starter intentionally grows one project at a time. The repo is designed so a coding agent can scaffold infrastructure, logging, tests, configs, and Modal execution while leaving the core algorithmic math for you to implement.

## Philosophy

- Keep the repo small until one project is working.
- Keep each algorithm's mathematical core in an obvious file.
- Let agents automate boring glue work.
- Do not let agents silently implement the parts you are trying to learn.
- Prefer readable implementations over clever abstractions.
- Use `uv` for Python, virtualenvs, dependency syncing, locking, and command execution.

## Current scope

```text
rl-frontier-lab-starter/
  .python-version
  AGENTS.md
  README.md
  pyproject.toml
  Makefile
  docs/
  rl_lab/
    core/
    modal/
    projects/
      p02_policy_gradients/
      p05_mini_rlhf/
```

Project 2 (`p02_policy_gradients`) now has a complete PPO implementation and
supporting infrastructure for local and Modal runs. Project 5
(`p05_mini_rlhf`) is the current protected learning surface for RLHF work.

## Setup

Install `uv`, then from the repo root run:

```bash
uv venv --python 3.11
uv sync --group dev --group logging --group modal
```

You usually do **not** need to activate the environment. Prefer `uv run ...` or the Make targets below. If your editor wants an interpreter path, point it at:

```text
.venv/bin/python
```

For later LLM projects, include the heavier LLM group only when needed:

```bash
uv sync --group dev --group logging --group modal --group llm
```

Update the lockfile after dependency changes:

```bash
uv lock
```

## Common commands

All commands route through `uv` via the Makefile:

```bash
make sync
make test
make lint
make format
make smoke PROJECT=p02_policy_gradients CONFIG=configs/ppo_cartpole_debug.yaml
make train-local PROJECT=p02_policy_gradients CONFIG=configs/ppo_cartpole_debug.yaml
make modal-smoke PROJECT=p02_policy_gradients CONFIG=configs/ppo_cartpole_debug.yaml
make smoke PROJECT=p05_mini_rlhf CONFIG=configs/rlhf_imdb_debug.yaml
```

Equivalent direct commands look like:

```bash
uv run --group dev --group logging pytest -q
uv run --group logging python -m rl_lab.projects.p02_policy_gradients.train \
  --config rl_lab/projects/p02_policy_gradients/configs/ppo_cartpole_debug.yaml \
  --smoke
```

For Project 2, `make smoke` runs the quick PPO infrastructure path and
`make train-local` runs local PPO training from the same project entrypoint.
The smoke path checks config loading, env creation, model creation, rollout
collection, logging, and output directories without requiring a long training
job.

For Project 5 full local training, use the LLM group:

```bash
make sync-llm
make train-local-llm PROJECT=p05_mini_rlhf CONFIG=configs/rlhf_imdb_debug.yaml
```

The Project 5 smoke path avoids external model/dataset downloads and only checks
dataset construction, artifacts, logging, and protected TODO boundaries.

## Logging

Experiment logging uses W&B. The checked-in debug config runs W&B in `offline` mode so smoke tests and early local runs do not need credentials, while still creating W&B run files and local JSONL artifacts under `experiments/runs`.

For online W&B logging, set `WANDB_API_KEY` in your environment or configure a Modal Secret named `wandb-secret`, then override `logging.mode=online` as needed.

## Dependency policy

- Add runtime dependencies with `uv add <package>`.
- Add dev-only tooling with `uv add --group dev <package>`.
- Add logging dependencies with `uv add --group logging <package>`.
- Add Modal dependencies with `uv add --group modal <package>`.
- Add LLM-only dependencies with `uv add --group llm <package>`.
- Commit `pyproject.toml`, `.python-version`, and `uv.lock` once generated.
- Never commit `.venv/`.

## Agent workflow

1. Read `AGENTS.md`.
2. Read the relevant project `SPEC.md`.
3. Ask the agent for scaffolding only: configs, tests, logging, plotting, Modal wrappers.
4. Personally implement functions marked `TODO(user)`.
5. Run unit tests and smoke tests.
6. Run local training.
7. Run Modal training.
8. Add an experiment note under `experiments/reports/`.

Example agent prompt:

> Add W&B logging and local JSONL artifacts for p02 without modifying `ppo.py`. Use only `uv`/Makefile commands. Log return, episode length, entropy, approx KL, clip fraction, value loss, policy loss, explained variance, FPS, seed, config hash, and git commit. Add tests for logger output and config overrides.

## Adding the next project

Do not add all projects at once. Once a project is working, clone the pattern:

```text
rl_lab/projects/p03_dqn_rainbow/
  SPEC.md
  README.md
  configs/
  dqn.py          # TODO(user) core math
  replay.py
  models.py
  train.py
  eval.py
  tests/
```

Keep reusable infrastructure in `rl_lab/core`. Keep algorithm-specific code inside the project directory.
