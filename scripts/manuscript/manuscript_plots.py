"""Plot generation for manuscript comparison outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .manuscript_metrics import REGION_ORDER
except ImportError:  # Direct script execution.
    from manuscript_metrics import REGION_ORDER


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def dataset_figures(rows: pd.DataFrame, figures_dir: Path) -> None:
    """Create the three dataset-composition figures."""
    source_counts = rows["source_dataset"].astype(str).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    source_counts.plot.bar(ax=ax, color="#4472C4")
    ax.set(ylabel="Rows", xlabel="Source dataset", title="Dataset rows by source")
    _save(fig, figures_dir / "dataset_rows_by_source.png")

    targets = ["absorption_nm", "emission_nm", "quantum_yield"]
    coverage = rows.groupby("source_dataset")[targets].count()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    coverage.plot.bar(ax=ax)
    ax.set(ylabel="Rows with target", xlabel="Source dataset", title="Target coverage by source")
    _save(fig, figures_dir / "target_coverage_by_source.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for source, subset in rows.dropna(subset=["emission_nm"]).groupby("source_dataset"):
        ax.hist(
            subset["emission_nm"],
            bins=np.arange(250, 1001, 20),
            alpha=0.45,
            label=str(source),
        )
    ax.set(
        xlabel="Emission wavelength (nm)",
        ylabel="Rows",
        title="Emission wavelength distribution by source",
    )
    ax.legend()
    _save(fig, figures_dir / "emission_distribution_by_source.png")


def result_figures(
    metrics: pd.DataFrame,
    region: pd.DataFrame,
    predictions: list[pd.DataFrame],
    figures_dir: Path,
) -> None:
    """Create model/split comparison and best-model diagnostic figures."""
    emission = metrics[metrics["target"] == "emission_nm"].copy()
    if emission.empty:
        return

    split_summary = emission.groupby("split", as_index=False)["mae"].mean()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(split_summary["split"], split_summary["mae"], color="#70AD47")
    ax.set(ylabel="Mean MAE (nm)", xlabel="Split", title="Emission MAE by split")
    _save(fig, figures_dir / "emission_mae_by_split.png")

    model_summary = emission.groupby("model", as_index=False)["mae"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(model_summary["model"], model_summary["mae"], color="#ED7D31")
    ax.set(ylabel="Mean MAE (nm)", xlabel="Model", title="Emission model comparison")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, figures_dir / "emission_mae_by_model.png")

    if not region.empty:
        heat = region.pivot_table(
            index="model", columns="region", values="mae", aggfunc="mean"
        ).reindex(columns=REGION_ORDER)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.55 * len(heat))))
        image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="magma")
        ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(heat.index)), heat.index)
        ax.set(title="Emission MAE by wavelength region", xlabel="Region", ylabel="Model")
        fig.colorbar(image, ax=ax, label="MAE (nm)")
        _save(fig, figures_dir / "emission_region_mae_heatmap.png")

    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions = all_predictions[all_predictions["target"] == "emission_nm"]
    best = (
        emission.groupby(["split", "model"], as_index=False)["mae"]
        .mean()
        .sort_values(["split", "mae"])
        .groupby("split", as_index=False)
        .first()
    )
    for _, choice in best.iterrows():
        subset = all_predictions[
            (all_predictions["split"] == choice["split"])
            & (all_predictions["model"] == choice["model"])
        ]
        if subset.empty:
            continue
        label = f"{choice['split']}__{choice['model']}"
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(subset["y_true"], subset["y_pred"], s=12, alpha=0.45)
        limits = [
            min(subset["y_true"].min(), subset["y_pred"].min()),
            max(subset["y_true"].max(), subset["y_pred"].max()),
        ]
        ax.plot(limits, limits, "--", color="black", linewidth=1)
        ax.set(
            xlabel="Actual emission (nm)",
            ylabel="Predicted emission (nm)",
            title=f"Predicted vs actual: {choice['split']} / {choice['model']}",
        )
        _save(fig, figures_dir / f"predicted_vs_actual__{label}.png")

        residual = subset["y_true"] - subset["y_pred"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(residual, bins=30, color="#5B9BD5", alpha=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set(
            xlabel="Residual: actual - predicted (nm)",
            ylabel="Rows",
            title=f"Residuals: {choice['split']} / {choice['model']}",
        )
        _save(fig, figures_dir / f"residual_histogram__{label}.png")
