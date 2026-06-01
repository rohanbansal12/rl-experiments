# Project 2 SPEC: Policy Gradients → PPO

## Goal

Implement REINFORCE, actor-critic, GAE, and PPO from scratch on small control tasks. Infrastructure may be assisted; algorithmic update functions are written by the user.

## Core files

- `ppo.py`: PPO loss, GAE, and update loop. User-owned math.
- `rollout.py`: rollout storage.
- `models.py`: policy/value networks.
- `train.py`: local training entrypoint.
- `eval.py`: deterministic evaluation scaffold.

## User-owned TODOs

- `compute_gae`
- `ppo_policy_loss`
- `ppo_value_loss`
- `update_ppo`

## Agent-safe TODOs

- CLI argument parsing
- config loading
- JSONL/TensorBoard/W&B logging
- checkpoint save/load
- plotting
- smoke tests
- Modal launch config
- config sweeps

## Required metrics

- `episodic_return_mean`
- `episodic_return_std`
- `episodic_length_mean`
- `policy_loss`
- `value_loss`
- `entropy`
- `approx_kl`
- `clip_fraction`
- `explained_variance`
- `steps_per_second`

## Required ablations

- GAE lambda sweep
- PPO clip coefficient sweep
- entropy coefficient sweep
- value loss coefficient sweep
- learning-rate failure case
- no-KL-early-stop failure case

## Done criteria

- CartPole solved across at least 3 seeds.
- LunarLander shows learning across at least 3 seeds.
- CPU smoke test passes.
- README contains derivation and failure analysis.
- Experiment report includes plots, qualitative failure notes, and exact configs.
