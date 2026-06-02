# p02 Policy Gradients Agent Instructions

This project teaches REINFORCE, actor-critic, GAE, and PPO.

Protected user-owned implementation points:

- `ppo.py::compute_gae`
- `ppo.py::ppo_policy_loss`
- `ppo.py::ppo_value_loss`
- `ppo.py::update_ppo`

Agent-safe work:

- improve config files
- improve logger plumbing
- add tests around expected math behavior
- add plotting scripts
- add env wrappers
- improve smoke tests
- improve W&B adapters
- improve CLI ergonomics

When adding tests for protected functions, use explicit small tensors with known expected values. It is okay for these tests to fail or be marked xfail until the user implements the functions.
