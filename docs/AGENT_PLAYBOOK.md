# Agent Playbook

Use this file for recurring prompts and repo-maintenance patterns.

## Safe scaffolding prompt

```text
Read AGENTS.md and rl_lab/projects/p02_policy_gradients/SPEC.md.
Scaffold the requested infrastructure without implementing any TODO(user) code.
Use uv/Makefile commands only; do not use pip, python -m venv, Poetry, or Conda.
Run make test and make smoke. Summarize files changed and remaining TODOs.
```

## Useful first tasks

1. Add richer config validation.
2. Add checkpoint save/load helpers.
3. Add plotting from JSONL logs.
4. Add W&B logging tests.
5. Add Modal GPU override support.
6. Add seed-sweep scripts.
7. Add Project 5 RLHF ablation configs after the protected losses/PPO methods
   are implemented by the user.

## Review checklist

- Did the agent modify protected algorithmic TODOs?
- Did it use `uv` for all environment/dependency commands?
- Did it add dependencies unnecessarily?
- Did it run the expected tests?
- Did it preserve local/remote train path equivalence?
- Did it log seed, config hash, and output directory?
