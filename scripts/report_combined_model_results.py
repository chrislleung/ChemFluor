"""Create a professor-ready report for combined ChemFluor + Deep4Chem models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_MODEL_DIR = Path("models/chemfluor_combined")
DEFAULT_REPORT_DIR = Path("outputs/combined_model_report")

METRIC_COLUMNS = [
    "target",
    "mae",
    "rmse",
    "r2",
    "train_rows",
    "test_rows",
    "unique_train_chromophores",
    "unique_test_chromophores",
]

TARGET_DESCRIPTIONS = {
    "absorption_nm": "Absorption maximum wavelength in nanometers; this approximates where the chromophore most strongly absorbs incoming light.",
    "emission_nm": "Emission maximum wavelength in nanometers; this approximates the observed fluorescence color after excitation.",
    "quantum_yield": "Photoluminescence quantum yield; this is the fraction of absorbed photons that are re-emitted as fluorescence.",
    "lifetime_ns": "Excited-state lifetime in nanoseconds; this reflects how long the emissive excited state persists before decay.",
    "log_extinction": "Log molar extinction coefficient; this is related to absorption strength and future brightness estimates.",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create summary tables, figures, and markdown for combined model results."
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        type=Path,
        help=f"Directory containing metrics and prediction CSVs. Defaults to {DEFAULT_MODEL_DIR}.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_REPORT_DIR,
        type=Path,
        help=f"Directory for report outputs. Defaults to {DEFAULT_REPORT_DIR}.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file with a helpful missing-file error."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(model_dir: Path) -> dict[str, dict[str, Any]]:
    """Load metrics.json as a target-keyed dictionary."""
    raw_metrics = load_json(model_dir / "metrics.json")
    if isinstance(raw_metrics, list):
        return {row["target"]: row for row in raw_metrics}
    if isinstance(raw_metrics, dict):
        return raw_metrics
    raise ValueError("metrics.json must contain either a dictionary or a list of metric rows.")


def load_feature_metadata(model_dir: Path) -> dict[str, Any]:
    """Load feature metadata if present."""
    metadata_path = model_dir / "feature_metadata.json"
    if not metadata_path.exists():
        return {}
    return load_json(metadata_path)


def metrics_to_table(metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Convert metrics to the requested CSV table shape."""
    rows = []
    for target, values in metrics.items():
        row = {column: values.get(column) for column in METRIC_COLUMNS}
        row["target"] = values.get("target", target)
        rows.append(row)
    return pd.DataFrame(rows, columns=METRIC_COLUMNS).sort_values("target")


def load_predictions(model_dir: Path, targets: list[str]) -> dict[str, pd.DataFrame]:
    """Load prediction CSVs for all metric targets."""
    predictions: dict[str, pd.DataFrame] = {}
    required_columns = {"y_true", "y_pred", "residual"}

    for target in targets:
        prediction_path = model_dir / f"predictions_{target}.csv"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {prediction_path}")
        df = pd.read_csv(prediction_path)
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(
                f"{prediction_path} is missing required column(s): {sorted(missing)}"
            )
        predictions[target] = df

    return predictions


def save_predicted_vs_actual(target: str, df: pd.DataFrame, figures_dir: Path) -> Path:
    """Save a predicted-vs-actual scatter plot."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["y_true"], df["y_pred"], s=12, alpha=0.6)
    lower = min(df["y_true"].min(), df["y_pred"].min())
    upper = max(df["y_true"].max(), df["y_pred"].max())
    ax.plot([lower, upper], [lower, upper], linewidth=1)
    ax.set_title(f"{target}: predicted vs actual")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = figures_dir / f"predicted_vs_actual_{target}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_residual_histogram(target: str, df: pd.DataFrame, figures_dir: Path) -> Path:
    """Save a residual histogram."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["residual"].dropna(), bins=40)
    ax.set_title(f"{target}: residual histogram")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = figures_dir / f"residual_histogram_{target}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_residual_vs_predicted(target: str, df: pd.DataFrame, figures_dir: Path) -> Path:
    """Save a residual-vs-predicted scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["y_pred"], df["residual"], s=12, alpha=0.6)
    ax.axhline(0, linewidth=1)
    ax.set_title(f"{target}: residual vs predicted")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = figures_dir / f"residual_vs_predicted_{target}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def create_figures(
    predictions: dict[str, pd.DataFrame], figures_dir: Path
) -> dict[str, dict[str, Path]]:
    """Create all requested figures for each target."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, dict[str, Path]] = {}
    for target, df in predictions.items():
        figure_paths[target] = {
            "predicted_vs_actual": save_predicted_vs_actual(target, df, figures_dir),
            "residual_histogram": save_residual_histogram(target, df, figures_dir),
            "residual_vs_predicted": save_residual_vs_predicted(target, df, figures_dir),
        }
    return figure_paths


def summarize_datasets(model_dir: Path) -> str:
    """Summarize source datasets from standardized rows when available."""
    standardized_path = model_dir / "combined_standardized_training_rows.csv"
    if not standardized_path.exists():
        return "The report uses the combined ChemFluor + Deep4Chem model outputs."

    rows = pd.read_csv(standardized_path, usecols=["source_dataset"])
    counts = rows["source_dataset"].value_counts().sort_index()
    count_text = ", ".join(f"{dataset}: {count}" for dataset, count in counts.items())
    return (
        "The models were trained from standardized ChemFluor and Deep4Chem rows. "
        f"The combined standardized table contains {len(rows)} rows ({count_text})."
    )


def format_metrics_markdown(metrics_table: pd.DataFrame) -> str:
    """Render the metrics table as markdown."""
    rounded = metrics_table.copy()
    for column in ["mae", "rmse", "r2"]:
        rounded[column] = rounded[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
        )
    headers = list(rounded.columns)
    rows = [[str(value) for value in row] for row in rounded.to_numpy()]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


def build_markdown_report(
    model_dir: Path,
    metrics_table: pd.DataFrame,
    metadata: dict[str, Any],
    figure_paths: dict[str, dict[str, Path]],
) -> str:
    """Build the professor-ready markdown report."""
    descriptor_columns = metadata.get("solvent_descriptor_columns_used", [])
    fingerprint_bits = metadata.get("fingerprint_n_bits", "unknown")
    fingerprint_radius = metadata.get("fingerprint_radius", "unknown")
    model_type = metadata.get("model_type", "unknown")

    descriptor_text = (
        ", ".join(descriptor_columns)
        if descriptor_columns
        else "solvent descriptor columns recorded in feature_metadata.json"
    )

    target_lines = [
        f"- `{target}`: {TARGET_DESCRIPTIONS.get(target, 'Optical-property target.')}"
        for target in metrics_table["target"]
    ]

    figure_lines = []
    for target, paths in figure_paths.items():
        figure_lines.append(f"- `{target}`")
        for label, path in paths.items():
            relative_path = path.as_posix()
            figure_lines.append(f"  - {label}: `{relative_path}`")

    return f"""# Combined ChemFluor + Deep4Chem Model Summary

## Data Used

{summarize_datasets(model_dir)}

The prediction outputs and metrics were read from `{model_dir}`. The training path combined the original ChemFluor dataset with the Deep4Chem chromophore dataset after standardizing molecule SMILES, solvent labels, target names, and dataset provenance.

## Features Used

The models used Morgan fingerprint bits for the canonical chromophore SMILES plus numeric solvent descriptor features. The fingerprint radius was `{fingerprint_radius}` and the fingerprint length was `{fingerprint_bits}` bits. The model type recorded in metadata is `{model_type}`.

Numeric solvent descriptors used: {descriptor_text}.

## Split Strategy

The train/test split was grouped by `canonical_chromophore_smiles`. This means the same chromophore scaffold was not allowed to appear in both train and test sets, giving a more realistic estimate of generalization to new chromophores.

## Target Meanings

{chr(10).join(target_lines)}

## Metrics

MAE is reported in the native unit of each target. For wavelength targets, lower MAE means fewer nanometers of average prediction error. For `quantum_yield`, MAE is on the 0-1 quantum-yield scale. For `lifetime_ns`, MAE is in nanoseconds. For `log_extinction`, MAE is in log units.

R² measures how much target variance is explained by the model on held-out chromophore groups. Higher R² indicates stronger predictive structure, while low or moderate R² suggests noisier labels, missing mechanisms, outliers, or insufficient feature coverage.

{format_metrics_markdown(metrics_table)}

## Required Interpretation

- absorption_nm is the strongest model
- emission_nm is good enough for first-pass fluorescence color screening
- quantum_yield is noisier and should be treated as a ranking signal, not an exact prediction
- lifetime_ns has fewer labels and is affected by outliers
- log_extinction is promising for future brightness estimation

## Figures

Each target has three diagnostic plots: predicted vs actual, residual histogram, and residual vs predicted.

{chr(10).join(figure_lines)}

## Main Takeaway

The combined model is most reliable for spectral position tasks, especially absorption and emission wavelength prediction. Quantum yield and lifetime remain useful but should be interpreted more cautiously because they are more sensitive to experimental conditions, measurement protocol, solvent effects, and label noise. The log-extinction result suggests that brightness-related modeling is feasible and worth expanding as more curated labels become available.
"""


def write_outputs(
    out_dir: Path,
    metrics_table: pd.DataFrame,
    markdown_report: str,
) -> None:
    """Write report files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(out_dir / "metrics_table.csv", index=False)
    (out_dir / "model_summary.md").write_text(markdown_report, encoding="utf-8")


def main() -> int:
    """Run report generation."""
    args = parse_args()
    try:
        metrics = load_metrics(args.model_dir)
        metadata = load_feature_metadata(args.model_dir)
        metrics_table = metrics_to_table(metrics)
        predictions = load_predictions(args.model_dir, metrics_table["target"].tolist())

        figures_dir = args.out_dir / "figures"
        figure_paths = create_figures(predictions, figures_dir)
        markdown_report = build_markdown_report(
            model_dir=args.model_dir,
            metrics_table=metrics_table,
            metadata=metadata,
            figure_paths=figure_paths,
        )
        write_outputs(args.out_dir, metrics_table, markdown_report)

        print(f"Saved markdown report to: {args.out_dir / 'model_summary.md'}")
        print(f"Saved metrics table to: {args.out_dir / 'metrics_table.csv'}")
        print(f"Saved figures to: {figures_dir}")
        return 0
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
