"""Metrics and bootstrap summaries for manuscript experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


REGION_ORDER = ["UV", "blue", "green", "yellow/orange", "red/NIR"]


def wavelength_nm_to_ev(values: Any) -> np.ndarray:
    """Convert positive wavelengths in nm to photon energies in eV."""
    array = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("Wavelengths must be finite and greater than zero.")
    return 1240.0 / array


def emission_region(value: float) -> str:
    """Bin an emission wavelength using the manuscript's explicit boundaries."""
    value = float(value)
    if value < 400:
        return "UV"
    if value < 500:
        return "blue"
    if value < 560:
        return "green"
    if value < 620:
        return "yellow/orange"
    return "red/NIR"


def regression_metrics(
    y_true: Any,
    y_pred: Any,
    target: str,
) -> dict[str, float]:
    """Compute manuscript regression metrics."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    result = {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)) if len(truth) > 1 else np.nan,
        "median_absolute_error": float(median_absolute_error(truth, prediction)),
    }
    if target in {"absorption_nm", "emission_nm"}:
        truth_ev = wavelength_nm_to_ev(truth)
        if np.all(np.isfinite(prediction)) and np.all(prediction > 0):
            prediction_ev = wavelength_nm_to_ev(prediction)
            result["mae_ev"] = float(mean_absolute_error(truth_ev, prediction_ev))
            result["rmse_ev"] = float(
                np.sqrt(mean_squared_error(truth_ev, prediction_ev))
            )
        else:
            result["mae_ev"] = np.nan
            result["rmse_ev"] = np.nan
    else:
        result["mae_ev"] = np.nan
        result["rmse_ev"] = np.nan
    return result


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_probability: Any | None = None,
) -> dict[str, float | int]:
    """Compute binary bright/dim quantum-yield metrics."""
    truth = np.asarray(y_true, dtype=int)
    prediction = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    result: dict[str, float | int] = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "roc_auc": np.nan,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    if y_probability is not None and len(np.unique(truth)) == 2:
        result["roc_auc"] = float(roc_auc_score(truth, y_probability))
    return result


def region_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute region-wise emission errors from a prediction table."""
    working = predictions.copy()
    working["region"] = working["y_true"].map(emission_region)
    working["residual"] = working["y_true"] - working["y_pred"]
    rows = []
    for region in REGION_ORDER:
        subset = working[working["region"] == region]
        if subset.empty:
            continue
        residual = subset["residual"].to_numpy(dtype=float)
        rows.append(
            {
                "region": region,
                "rows": int(len(subset)),
                "mae": float(np.mean(np.abs(residual))),
                "median_absolute_error": float(np.median(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "mean_residual": float(np.mean(residual)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_regression_metrics(
    predictions: pd.DataFrame,
    target: str,
    seed: int,
    n_bootstrap: int = 500,
) -> pd.DataFrame:
    """Bootstrap prediction rows and return percentile confidence intervals."""
    truth = predictions["y_true"].to_numpy(dtype=float)
    predicted = predictions["y_pred"].to_numpy(dtype=float)
    point = regression_metrics(truth, predicted, target)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in point}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(truth), size=len(truth))
        values = regression_metrics(truth[indices], predicted[indices], target)
        for name, value in values.items():
            if np.isfinite(value):
                samples[name].append(float(value))
    rows = []
    for name, estimate in point.items():
        values = samples[name]
        rows.append(
            {
                "metric": name,
                "estimate": estimate,
                "ci_lower": float(np.percentile(values, 2.5)) if values else np.nan,
                "ci_upper": float(np.percentile(values, 97.5)) if values else np.nan,
                "bootstrap_samples": n_bootstrap,
            }
        )
    return pd.DataFrame(rows)
