"""Create a manuscript-ready predicted-versus-actual PLQY scatterplot."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from scipy.stats import pearsonr
except ImportError:  # pragma: no cover - exercised only in environments without scipy
    pearsonr = None


DEFAULT_PREDICTION_CSV = Path(
    "outputs/paper_comparison/predictions/"
    "quantum_yield__extratrees__random__seed0.csv"
)
DEFAULT_OUT_DIR = Path("outputs/paper_comparison/qy_predicted_vs_actual")
ACTUAL_COLUMN_CANDIDATES = (
    "actual",
    "actual_value",
    "y_true",
    "true",
    "experimental",
    "experimental_quantum_yield",
    "quantum_yield",
)
PREDICTED_COLUMN_CANDIDATES = (
    "predicted",
    "prediction",
    "y_pred",
    "predicted_value",
    "predicted_quantum_yield",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-csv", type=Path, default=DEFAULT_PREDICTION_CSV
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--title", default="Predicted vs Actual Quantum Yield RF Scaffold Split")
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--allow-out-of-range-predictions", action="store_true")
    return parser.parse_args()


def _find_column(columns: Iterable[object], candidates: tuple[str, ...]) -> str | None:
    """Find the first candidate using case-insensitive, whitespace-tolerant matching."""
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def detect_qy_columns(data: pd.DataFrame) -> tuple[str, str]:
    """Detect actual and predicted PLQY columns or raise a descriptive error."""
    actual = _find_column(data.columns, ACTUAL_COLUMN_CANDIDATES)
    predicted = _find_column(data.columns, PREDICTED_COLUMN_CANDIDATES)
    if actual is None or predicted is None:
        available = ", ".join(map(str, data.columns)) or "(none)"
        missing = []
        if actual is None:
            missing.append("actual QY")
        if predicted is None:
            missing.append("predicted QY")
        raise ValueError(
            f"Could not detect {' and '.join(missing)} column(s). "
            f"Available columns: {available}"
        )
    return actual, predicted


def prepare_plot_data(
    data: pd.DataFrame,
    actual_column: str,
    predicted_column: str,
    allow_out_of_range_predictions: bool = False,
    max_points: int | None = None,
) -> pd.DataFrame:
    """Coerce, filter, and optionally reproducibly sample prediction pairs."""
    if max_points is not None and max_points < 1:
        raise ValueError("--max-points must be a positive integer")
    pairs = data[[actual_column, predicted_column]].copy()
    pairs.columns = ["actual", "predicted"]
    pairs["actual"] = pd.to_numeric(pairs["actual"], errors="coerce")
    pairs["predicted"] = pd.to_numeric(pairs["predicted"], errors="coerce")
    pairs = pairs.dropna(subset=["actual", "predicted"])
    pairs = pairs.loc[pairs["actual"].between(0.0, 1.0, inclusive="both")]
    if not allow_out_of_range_predictions:
        pairs = pairs.loc[pairs["predicted"].between(0.0, 1.0, inclusive="both")]
    if max_points is not None and len(pairs) > max_points:
        pairs = pairs.sample(n=max_points, random_state=42)
    return pairs.reset_index(drop=True)


def calculate_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> dict[str, float | int | str]:
    """Calculate regression performance, correlation, and linear-fit metrics."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted arrays must have the same shape")
    if actual_values.ndim != 1 or len(actual_values) < 2:
        raise ValueError("At least two actual/predicted pairs are required")

    model = LinearRegression().fit(actual_values.reshape(-1, 1), predicted_values)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    if np.std(actual_values) == 0 or np.std(predicted_values) == 0:
        correlation = float("nan")
    elif pearsonr is not None:
        correlation = float(pearsonr(actual_values, predicted_values)[0])
    else:  # pragma: no cover - scipy is installed in the test environment
        correlation = float(np.corrcoef(actual_values, predicted_values)[0, 1])

    return {
        "n_points": int(len(actual_values)),
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "r2": float(r2_score(actual_values, predicted_values)),
        "pearson_r": correlation,
        "slope": slope,
        "intercept": intercept,
        "regression_equation": f"predicted = {slope:.3f} × actual {intercept:+.3f}",
    }


def _metric_text(metrics: dict[str, float | int | str]) -> str:
    """Format metrics for the plot annotation."""
    return "\n".join(
        [
            f"MAE = {metrics['mae']:.3f}",
            f"RMSE = {metrics['rmse']:.3f}",
            f"R² = {metrics['r2']:.3f}",
            f"r = {metrics['pearson_r']:.3f}",
            f"Predicted = {metrics['slope']:.3f} × Actual {metrics['intercept']:+.3f}",
        ]
    )


def create_plot(
    pairs: pd.DataFrame,
    metrics: dict[str, float | int | str],
    title: str,
    alpha: float,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Render and save the scatterplot in raster and vector formats."""
    if not 0 < alpha <= 1:
        raise ValueError("--alpha must be greater than 0 and at most 1")
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(
        pairs["actual"],
        pairs["predicted"],
        s=18,
        alpha=alpha,
        color="#2878a6",
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    axis_values = np.array([0.0, 1.0])
    ax.plot(
        axis_values,
        axis_values,
        linestyle="--",
        color="#555555",
        linewidth=1.4,
        label="Perfect prediction",
        zorder=1,
    )
    fitted = float(metrics["slope"]) * axis_values + float(metrics["intercept"])
    ax.plot(
        axis_values,
        fitted,
        color="#c44e52",
        linewidth=2.0,
        label="Linear fit",
        zorder=3,
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Actual PLQY",
        ylabel="Predicted PLQY",
        title=title,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#d9d9d9", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    ax.text(
        0.035,
        0.965,
        _metric_text(metrics),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.92},
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(
    path: Path,
    prediction_csv: Path,
    metrics: dict[str, float | int | str],
) -> None:
    """Write a concise Markdown summary of the plotted results."""
    text = f"""# Predicted vs actual quantum yield

## Results

- Input prediction file: `{prediction_csv}`
- Number of plotted points: **{metrics['n_points']:,}**
- MAE: **{metrics['mae']:.4f}**
- RMSE: **{metrics['rmse']:.4f}**
- R²: **{metrics['r2']:.4f}**
- Pearson r: **{metrics['pearson_r']:.4f}**
- Linear regression: **{metrics['regression_equation']}**

## Interpretation

The spread of points around the y = x line shows that quantum-yield predictions have substantial uncertainty. Even when the overall trend is positive, the scatter indicates that exact PLQY regression is noisy and should be interpreted more cautiously than wavelength prediction.
"""
    path.write_text(text, encoding="utf-8")


def run_analysis(
    prediction_csv: Path,
    out_dir: Path,
    title: str = "Predicted vs Actual Quantum Yield",
    max_points: int | None = None,
    alpha: float = 0.35,
    allow_out_of_range_predictions: bool = False,
) -> dict[str, float | int | str]:
    """Read predictions and produce all manuscript artifacts."""
    data = pd.read_csv(prediction_csv)
    try:
        actual_column, predicted_column = detect_qy_columns(data)
    except ValueError:
        print(f"Available columns: {', '.join(map(str, data.columns))}")
        raise
    pairs = prepare_plot_data(
        data,
        actual_column,
        predicted_column,
        allow_out_of_range_predictions,
        max_points,
    )
    if len(pairs) < 2:
        raise ValueError(
            "Fewer than two valid actual/predicted QY pairs remain after filtering"
        )
    metrics = calculate_metrics(pairs["actual"], pairs["predicted"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(
        out_dir / "qy_predicted_vs_actual_metrics.csv", index=False
    )
    create_plot(
        pairs,
        metrics,
        title,
        alpha,
        out_dir / "qy_predicted_vs_actual_scatter.png",
        out_dir / "qy_predicted_vs_actual_scatter.pdf",
    )
    write_report(out_dir / "qy_predicted_vs_actual_report.md", prediction_csv, metrics)
    print(f"Plotted {len(pairs):,} QY predictions to {out_dir}")
    return metrics


def main() -> None:
    args = parse_args()
    run_analysis(
        args.prediction_csv,
        args.out_dir,
        args.title,
        args.max_points,
        args.alpha,
        args.allow_out_of_range_predictions,
    )


if __name__ == "__main__":
    main()
