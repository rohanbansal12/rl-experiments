import json
from pathlib import Path

from rl_lab.projects.p02_policy_gradients.train import train

CONFIG = Path("rl_lab/projects/p02_policy_gradients/configs/ppo_cartpole_debug.yaml")


def test_smoke_run(tmp_path) -> None:
    result = train(
        config_path=CONFIG,
        overrides=["train.rollout_steps=8", "train.num_envs=2", "logging.mode=disabled"],
        output_dir=tmp_path,
        smoke=True,
    )
    assert result["ok"] is True
    assert result["mode"] == "smoke"

    run_dir = Path(result["run_dir"])
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metadata.json").exists()

    saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert saved_config["logging"]["backend"] == "wandb"
    assert saved_config["logging"]["mode"] == "disabled"
    assert saved_config["train"]["rollout_steps"] == 8
    assert saved_config["train"]["num_envs"] == 2

    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["step"] == 16
    assert records[0]["smoke"] is True
    assert "rollout_reward_mean" in records[0]
    assert "rollout_done_fraction" in records[0]
