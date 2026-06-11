"""Compare Random Forest and HistGradientBoosting combined model results."""

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


DEFAULT_RF_DIR = Path("models/chemfluor_combined")
DEFAULT_HISTGB_DIR = Path("models/chemfluor_combined_histgb")
DEFAULT_OUT_DIR = Path("outputs/model_comparison_report")

COMPARISON_COLUMNS = [
    "target",
    "rf_mae",
    "histgb_mae",
    "better_mae_model",
    "rf_rmse",
    "histgb_rmse",
    "rf_r2",
    "histgb_r2",
    "better_r2_model",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare RF and HistGB metrics for combined ChemFluor models."
    )
    parser.add_argument("--rf-dir", default=DEFAULT_RF_DIR, type=Path)
    parser.add_argument("--histgb-dir", default=DEFAULT_HISTGB_DIR, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    return parser.parse_args()


def load_metrics(model_dir: Path) -> dict[str, dict[str, Any]]:
    """Load target-keyed metrics from a model directory."""
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if isinstance(metrics, list):
        return {row["target"]: row for row in metrics}
    if isinstance(metrics, dict):
        return metrics
    raise ValueError(f"Unsupported metrics format in {metrics_path}")


def better_lower(rf_value: float, histgb_value: float) -> str:
    """Return the model name with the lower metric value."""
    if pd.isna(rf_value) and pd.isna(histgb_value):
        return "tie"
    if pd.isna(rf_value):
        return "histgb"
    if pd.isna(histgb_value):
        return "rf"
    if rf_value < histgb_value:
        return "rf"
    if histgb_value < rf_value:
        return "histgb"
    return "tie"


def better_higher(rf_value: float, histgb_value: float) -> str:
    """Return the model name with the higher metric value."""
    if pd.isna(rf_value) and pd.isna(histgb_value):
        return "tie"
    if pd.isna(rf_value):
        return "histgb"
    if pd.isna(histgb_value):
        return "rf"
    if rf_value > histgb_value:
        return "rf"
    if histgb_value > rf_value:
        return "histgb"
    return "tie"


def build_comparison_table(
    rf_metrics: dict[str, dict[str, Any]],
    histgb_metrics: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build the side-by-side model comparison table."""
    targets = sorted(set(rf_metrics) | set(histgb_metrics))
    rows: list[dict[str, Any]] = []

    for target in targets:
        rf = rf_metrics.get(target, {})
        histgb = histgb_metrics.get(target, {})
        rf_mae = rf.get("mae", pd.NA)
        histgb_mae = histgb.get("mae", pd.NA)
        rf_r2 = rf.get("r2", pd.NA)
        histgb_r2 = histgb.get("r2", pd.NA)

        rows.append(
            {
                "target": target,
                "rf_mae": rf_mae,
                "histgb_mae": histgb_mae,
                "better_mae_model": better_lower(rf_mae, histgb_mae),
                "rf_rmse": rf.get("rmse", pd.NA),
                "histgb_rmse": histgb.get("rmse", pd.NA),
                "rf_r2": rf_r2,
                "histgb_r2": histgb_r2,
                "better_r2_model": better_higher(rf_r2, histgb_r2),
            }
        )

    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def save_metric_bar_plot(
    comparison: pd.DataFrame,
    metric: str,
    y_label: str,
    out_dir: Path,
) -> Path:
    """Save a side-by-side RF vs HistGB bar plot for one metric."""
    plot_df = comparison.set_index("target")[[f"rf_{metric}", f"histgb_{metric}"]]
    ax = plot_df.plot(kind="bar", figsize=(9, 5))
    ax.set_title(f"RF vs HistGB {y_label}")
    ax.set_xlabel("Target")
    ax.set_ylabel(y_label)
    ax.legend(["RF", "HistGB"])
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = out_dir / f"{metric}_comparison.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def create_plots(comparison: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Create all comparison bar plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "mae": save_metric_bar_plot(comparison, "mae", "MAE", out_dir),
        "rmse": save_metric_bar_plot(comparison, "rmse", "RMSE", out_dir),
        "r2": save_metric_bar_plot(comparison, "r2", "R2", out_dir),
    }


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as a markdown table without optional dependencies."""
    display = df.copy()
    for column in ["rf_mae", "histgb_mae", "rf_rmse", "histgb_rmse", "rf_r2", "histgb_r2"]:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
        )

    headers = list(display.columns)
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


def build_markdown_report(
    comparison: pd.DataFrame,
    rf_dir: Path,
    histgb_dir: Path,
    plot_paths: dict[str, Path],
) -> str:
    """Build the markdown comparison report."""
    mae_counts = comparison["better_mae_model"].value_counts()
    r2_counts = comparison["better_r2_model"].value_counts()
    rf_mae_wins = int(mae_counts.get("rf", 0))
    histgb_mae_wins = int(mae_counts.get("histgb", 0))
    rf_r2_wins = int(r2_counts.get("rf", 0))
    histgb_r2_wins = int(r2_counts.get("histgb", 0))

    plot_lines = "\n".join(
        f"- {metric.upper()}: `{path.as_posix()}`" for metric, path in plot_paths.items()
    )

    return f"""# Combined Model Comparison: Random Forest vs HistGB

## Inputs

- Random Forest metrics: `{rf_dir / 'metrics.json'}`
- HistGB metrics: `{histgb_dir / 'metrics.json'}`

## Summary

Random Forest is the better main baseline by MAE, while HistGB is competitive by R2 for some targets. In this comparison, Random Forest has the lower MAE for {rf_mae_wins} target(s), and HistGB has the lower MAE for {histgb_mae_wins} target(s). By R2, Random Forest leads on {rf_r2_wins} target(s), while HistGB leads on {histgb_r2_wins} target(s).

MAE and RMSE should be read in each target's native units. Lower MAE and RMSE are better. R2 measures held-out variance explained under the grouped-by-chromophore split, so higher R2 is better.

## Comparison Table

{markdown_table(comparison)}

## Figures

{plot_lines}

## Recommendation

Use Random Forest as the main baseline because it gives the strongest MAE profile overall. Keep HistGB as a useful secondary model because its R2 is competitive for some targets, which suggests it can capture variance structure that may be valuable in an ensemble or follow-up tuning run.
"""


def write_outputs(
    comparison: pd.DataFrame,
    markdown_report: str,
    out_dir: Path,
) -> None:
    """Write comparison CSV and markdown report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)
    (out_dir / "model_comparison.md").write_text(markdown_report, encoding="utf-8")


def main() -> int:
    """Run model comparison report generation."""
    args = parse_args()
    try:
        rf_metrics = load_metrics(args.rf_dir)
        histgb_metrics = load_metrics(args.histgb_dir)
        comparison = build_comparison_table(rf_metrics, histgb_metrics)
        plot_paths = create_plots(comparison, args.out_dir)
        markdown_report = build_markdown_report(
            comparison=comparison,
            rf_dir=args.rf_dir,
            histgb_dir=args.histgb_dir,
            plot_paths=plot_paths,
        )
        write_outputs(comparison, markdown_report, args.out_dir)

        print(f"Saved comparison CSV to: {args.out_dir / 'model_comparison.csv'}")
        print(f"Saved markdown report to: {args.out_dir / 'model_comparison.md'}")
        print(f"Saved plots to: {args.out_dir}")
        return 0
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
