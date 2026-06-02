# Policy Gradients

This project starts with PPO because it exercises the implementation skills that
transfer to later LLM RL work: rollout collection, log-prob ratios, advantage
estimation, clipping, entropy, KL diagnostics, value heads, and seed-sensitive
debugging.

## What's Implemented

- PPO for discrete-action Gymnasium environments with a shared actor-critic MLP.
- Vectorized rollout collection with stored observations, actions, log-probs,
  rewards, dones, and value predictions.
- Generalized advantage estimation, clipped PPO policy loss, value loss,
  entropy bonus, approximate KL diagnostics, clip fraction, gradient clipping,
  minibatch updates, advantage-normalization toggling, and optional target-KL
  early stopping.
- Config loading with command-line overrides for training, logging, optimizer,
  model, PPO, and evaluation settings.
- Local JSONL artifact logging, optional W&B logging, saved configs, checkpoint
  save/load helpers, and deterministic evaluation from checkpoints.
- Local and Modal training paths that call the same project-level `train(...)`
  entrypoint.
- Tests for PPO math, rollout shapes, config overrides, logging artifacts,
  checkpoint/eval behavior, train infrastructure, and CPU smoke runs.

## Report

The project report contains the PPO/GAE derivation, CartPole baseline,
ablation results, generalization checks, conclusions, and future work:

- [REPORT.md](REPORT.md)

## Demo Video

This short CartPole video shows PPO progressing from early balance failures to
later stable control.

<video src="../../../experiments/videos/p02_cartpole_training_progress_seed1_smooth_short.mp4" controls width="560"></video>

## Commands

Use the root Makefile; the targets call `uv` internally.

Run a quick local CPU smoke test:

```bash
make smoke PROJECT=p02_policy_gradients CONFIG=configs/ppo_cartpole_debug.yaml
```

Run a local CPU training job:

```bash
make train-local PROJECT=p02_policy_gradients CONFIG=configs/ppo_cartpole_debug.yaml
```

Run a Modal CPU smoke job:

```bash
make modal-smoke PROJECT=p02_policy_gradients CONFIG=configs/ppo_cartpole_debug.yaml
```

Run a Modal CPU training job:

```bash
make modal-train PROJECT=p02_policy_gradients CONFIG=configs/ppo_cartpole_debug.yaml
```

Run a Modal CUDA/GPU training job:

```bash
make modal-train PROJECT=p02_policy_gradients \
  CONFIG=configs/ppo_cartpole_debug.yaml \
  RUNTIME=cuda
```

Pass config overrides as a comma-separated list:

```bash
make modal-train PROJECT=p02_policy_gradients \
  CONFIG=configs/ppo_cartpole_debug.yaml \
  OVERRIDES="logging.mode=online,eval.enabled=true,eval.episodes=10"
```

Direct Modal equivalent:

```bash
uv run --group modal modal run -m rl_lab.modal.runner \
  --project p02_policy_gradients \
  --config rl_lab/projects/p02_policy_gradients/configs/ppo_cartpole_debug.yaml \
  --runtime cpu \
  --no-smoke
```

## Protected Math

Core PPO math files remain protected learning surfaces for this repo. Prefer
adding tests, shape checks, documentation, and infrastructure around them unless
the user explicitly asks for algorithm changes.
