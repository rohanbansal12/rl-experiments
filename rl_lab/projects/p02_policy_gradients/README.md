# p02 Policy Gradients

This project starts with PPO because it exercises the implementation skills that transfer to later LLM RL work: rollout collection, log-prob ratios, advantage estimation, clipping, entropy, KL diagnostics, value heads, and seed-sensitive debugging.

## First milestone

1. Run the smoke test.
2. Implement `compute_gae` in `ppo.py`.
3. Add/enable a known-values test for GAE.
4. Implement PPO policy and value losses.
5. Implement `update_ppo`.
6. Train CartPole.
7. Add ablations.

## Commands

Use the root Makefile; the targets call `uv` internally.

```bash
make smoke PROJECT=p02_policy_gradients CONFIG=configs/ppo_cartpole_debug.yaml
make train-local PROJECT=p02_policy_gradients CONFIG=configs/ppo_cartpole_debug.yaml
```

Direct equivalent:

```bash
uv run --group logging python -m rl_lab.projects.p02_policy_gradients.train \
  --config rl_lab/projects/p02_policy_gradients/configs/ppo_cartpole_debug.yaml \
  --smoke
```

## Protected math

Functions marked `TODO(user)` are intentionally not implemented in the starter.
