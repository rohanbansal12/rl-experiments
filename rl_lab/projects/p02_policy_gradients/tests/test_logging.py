import json

from rl_lab.core.logging import MetricLogger


def test_wandb_disabled_backend_writes_local_artifacts(tmp_path) -> None:
    logger = MetricLogger(
        output_dir=tmp_path,
        config={"seed": 7, "logging": {"backend": "wandb", "mode": "disabled"}},
        backend="wandb",
        run_name="unit-test",
        project="rl-frontier-lab-tests",
        tags=["unit"],
        mode="disabled",
    )
    logger.log({"loss": 1.5, "ok": True}, step=3)
    logger.close()

    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert config["seed"] == 7
    assert metadata["run_name"] == "unit-test"
    assert "git_commit" in metadata
    assert records == [
        {
            "elapsed_seconds": records[0]["elapsed_seconds"],
            "loss": 1.5,
            "ok": True,
            "step": 3,
        }
    ]
