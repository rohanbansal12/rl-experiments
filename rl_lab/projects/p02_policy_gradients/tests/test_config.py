from pathlib import Path

from rl_lab.core.config import config_hash, load_config, parse_overrides


CONFIG = Path("rl_lab/projects/p02_policy_gradients/configs/ppo_cartpole_debug.yaml")


def test_load_config_and_overrides() -> None:
    cfg = load_config(CONFIG, parse_overrides("train.total_timesteps=123,ppo.clip_coef=0.15"))
    assert cfg["train"]["total_timesteps"] == 123
    assert cfg["ppo"]["clip_coef"] == 0.15
    assert cfg["env_id"] == "CartPole-v1"


def test_config_hash_is_stable() -> None:
    cfg1 = load_config(CONFIG)
    cfg2 = load_config(CONFIG)
    assert config_hash(cfg1) == config_hash(cfg2)
