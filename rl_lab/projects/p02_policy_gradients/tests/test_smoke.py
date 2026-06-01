from pathlib import Path

from rl_lab.projects.p02_policy_gradients.train import train


CONFIG = Path("rl_lab/projects/p02_policy_gradients/configs/ppo_cartpole_debug.yaml")


def test_smoke_run(tmp_path) -> None:
    result = train(
        config_path=CONFIG,
        overrides=["train.rollout_steps=8", "train.num_envs=2", "logging.backend=jsonl"],
        output_dir=tmp_path,
        smoke=True,
    )
    assert result["ok"] is True
    assert result["mode"] == "smoke"
