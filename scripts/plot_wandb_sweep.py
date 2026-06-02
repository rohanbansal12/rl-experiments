from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb


def plot_wandb_sweep(
    *,
    project_path: str,
    runs: dict[str, str],
    output_dir: str | Path,
    sweep_name: str,
    title_prefix: str = "Sweep",
    legend_title: str = "value",
    x_key: str = "_step",
) -> list[Path]:
    """Fetch W&B histories and write aggregate sweep plots."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()

    histories: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for label, run_id in runs.items():
        run = api.run(f"{project_path}/{run_id}")
        history = pd.DataFrame(run.history(samples=1000, pandas=False))
        if history.empty:
            continue
        for key in [
            "_step",
            "advantage_mean",
            "advantage_std",
            "approx_kl",
            "clip_fraction",
            "entropy",
            "episodic_return_mean",
            "episodic_return_std",
            "eval_return_mean",
            "eval_return_std",
            "explained_variance",
            "normalize_advantages",
            "policy_loss",
            "return_mean",
            "return_std",
            "value_loss",
        ]:
            if key in history:
                history[key] = pd.to_numeric(history[key], errors="coerce")
        history["sweep_value"] = label
        history["run_id"] = run_id
        histories.append(history)

        summary = dict(run.summary)
        finite_train = history["episodic_return_mean"].dropna()
        summaries.append(
            {
                "sweep_value": label,
                "run_id": run_id,
                "eval_return_mean": summary.get("eval_return_mean"),
                "eval_return_std": summary.get("eval_return_std"),
                "train_return_first": finite_train.iloc[0] if not finite_train.empty else None,
                "train_return_last": finite_train.iloc[-1] if not finite_train.empty else None,
                "train_return_best": finite_train.max() if not finite_train.empty else None,
                "advantage_mean_last": _last_finite(history, "advantage_mean"),
                "advantage_std_last": _last_finite(history, "advantage_std"),
                "approx_kl_last": _last_finite(history, "approx_kl"),
                "clip_fraction_last": _last_finite(history, "clip_fraction"),
                "return_mean_last": _last_finite(history, "return_mean"),
                "return_std_last": _last_finite(history, "return_std"),
            }
        )

    if not histories:
        raise RuntimeError("No W&B histories were fetched")

    history_df = pd.concat(histories, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    history_df.to_csv(output / f"{sweep_name}_history.csv", index=False)
    summary_df.to_csv(output / f"{sweep_name}_summary.csv", index=False)

    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("viridis", n_colors=len(runs))
    paths: list[Path] = []

    _append_if_present(
        paths,
        _line_plot(
            history_df,
            output / f"{sweep_name}_training_return.png",
            x_key=x_key,
            y_key="episodic_return_mean",
            title=f"{title_prefix}: training return",
            y_label="episodic return mean",
            legend_title=legend_title,
            palette=palette,
        ),
    )
    _append_if_present(
        paths,
        _line_plot(
            history_df,
            output / f"{sweep_name}_episodic_return_std.png",
            x_key=x_key,
            y_key="episodic_return_std",
            title=f"{title_prefix}: training return variability",
            y_label="episodic return std",
            legend_title=legend_title,
            palette=palette,
        ),
    )
    _append_if_present(
        paths,
        _line_plot(
            history_df,
            output / f"{sweep_name}_explained_variance.png",
            x_key=x_key,
            y_key="explained_variance",
            title=f"{title_prefix}: critic explained variance",
            y_label="explained variance",
            legend_title=legend_title,
            palette=palette,
        ),
    )
    _append_if_present(
        paths,
        _line_plot(
            history_df,
            output / f"{sweep_name}_approx_kl.png",
            x_key=x_key,
            y_key="approx_kl",
            title=f"{title_prefix}: approximate KL",
            y_label="approx KL",
            legend_title=legend_title,
            palette=palette,
        ),
    )
    _append_if_present(
        paths,
        _line_plot(
            history_df,
            output / f"{sweep_name}_clip_fraction.png",
            x_key=x_key,
            y_key="clip_fraction",
            title=f"{title_prefix}: PPO clip fraction",
            y_label="clip fraction",
            legend_title=legend_title,
            palette=palette,
        ),
    )
    _append_if_present(
        paths,
        _line_plot(
            history_df,
            output / f"{sweep_name}_value_loss.png",
            x_key=x_key,
            y_key="value_loss",
            title=f"{title_prefix}: value loss",
            y_label="value loss",
            legend_title=legend_title,
            palette=palette,
        ),
    )
    _append_if_present(
        paths,
        _line_plot(
            history_df,
            output / f"{sweep_name}_policy_loss.png",
            x_key=x_key,
            y_key="policy_loss",
            title=f"{title_prefix}: policy loss",
            y_label="policy loss",
            legend_title=legend_title,
            palette=palette,
        ),
    )
    _append_if_present(
        paths,
        _line_plot(
            history_df,
            output / f"{sweep_name}_advantage_mean.png",
            x_key=x_key,
            y_key="advantage_mean",
            title=f"{title_prefix}: raw advantage mean",
            y_label="advantage mean",
            legend_title=legend_title,
            palette=palette,
        ),
    )
    _append_if_present(
        paths,
        _line_plot(
            history_df,
            output / f"{sweep_name}_advantage_std.png",
            x_key=x_key,
            y_key="advantage_std",
            title=f"{title_prefix}: raw advantage variability",
            y_label="advantage std",
            legend_title=legend_title,
            palette=palette,
        ),
    )
    _append_if_present(
        paths,
        _line_plot(
            history_df,
            output / f"{sweep_name}_return_mean.png",
            x_key=x_key,
            y_key="return_mean",
            title=f"{title_prefix}: target return mean",
            y_label="GAE return mean",
            legend_title=legend_title,
            palette=palette,
        ),
    )
    _append_if_present(
        paths,
        _line_plot(
            history_df,
            output / f"{sweep_name}_return_std.png",
            x_key=x_key,
            y_key="return_std",
            title=f"{title_prefix}: target return variability",
            y_label="GAE return std",
            legend_title=legend_title,
            palette=palette,
        ),
    )
    paths.append(
        _summary_plot(
            summary_df,
            output / f"{sweep_name}_final_vs_best.png",
            title=f"{title_prefix}: final checkpoint vs training progress",
            x_label=legend_title,
        )
    )
    return paths


def _line_plot(
    df: pd.DataFrame,
    path: Path,
    *,
    x_key: str,
    y_key: str,
    title: str,
    y_label: str,
    legend_title: str,
    palette: list[tuple[float, float, float]],
) -> Path | None:
    if y_key not in df:
        return None
    plot_df = df[[x_key, y_key, "sweep_value"]].dropna().copy()
    if plot_df.empty:
        return None
    plot_df[y_key] = plot_df.groupby("sweep_value", sort=False)[y_key].transform(
        lambda values: values.rolling(window=7, min_periods=1).mean()
    )

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.lineplot(
        data=plot_df,
        x=x_key,
        y=y_key,
        hue="sweep_value",
        palette=palette,
        linewidth=2.5,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(y_label)
    ax.legend(title=legend_title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _append_if_present(paths: list[Path], maybe_path: Path | None) -> None:
    if maybe_path is not None:
        paths.append(maybe_path)


def _last_finite(history: pd.DataFrame, key: str) -> float | None:
    if key not in history:
        return None
    values = history[key].dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _summary_plot(
    summary_df: pd.DataFrame,
    path: Path,
    *,
    title: str,
    x_label: str,
) -> Path:
    long_df = summary_df.melt(
        id_vars=["sweep_value"],
        value_vars=["eval_return_mean", "train_return_best", "train_return_last"],
        var_name="metric",
        value_name="return",
    )
    label_map = {
        "eval_return_mean": "final eval",
        "train_return_best": "best train",
        "train_return_last": "last train",
    }
    long_df["metric"] = long_df["metric"].map(label_map)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.barplot(
        data=long_df,
        x="sweep_value",
        y="return",
        hue="metric",
        palette=["#377eb8", "#999999", "#4daf4a"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("return")
    ax.axhline(500, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _parse_runs(raw: str) -> dict[str, str]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise TypeError("--runs must be a JSON object mapping labels to W&B run ids")
    return {str(key): str(value) for key, value in parsed.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a W&B sweep from run histories.")
    parser.add_argument("--project-path", default="robansal/rl-frontier-lab")
    parser.add_argument(
        "--runs", required=True, help="JSON mapping from sweep label to W&B run id."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sweep-name", required=True)
    parser.add_argument("--title-prefix", default="Sweep")
    parser.add_argument("--legend-title", default="value")
    args = parser.parse_args()

    paths = plot_wandb_sweep(
        project_path=args.project_path,
        runs=_parse_runs(args.runs),
        output_dir=args.output_dir,
        sweep_name=args.sweep_name,
        title_prefix=args.title_prefix,
        legend_title=args.legend_title,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
