"""Tests for the manuscript predicted-versus-actual QY plot."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.manuscript.plot_qy_predicted_vs_actual import (
    calculate_metrics,
    detect_qy_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "manuscript" / "plot_qy_predicted_vs_actual.py"


def test_detect_qy_columns_uses_likely_names() -> None:
    data = pd.DataFrame({"metadata": [1], "y_true": [0.2], "y_pred": [0.3]})
    assert detect_qy_columns(data) == ("y_true", "y_pred")


def test_calculate_metrics_on_tiny_dataset() -> None:
    metrics = calculate_metrics([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["pearson_r"] == pytest.approx(1.0)
    assert metrics["slope"] == pytest.approx(1.0)
    assert metrics["intercept"] == pytest.approx(0.0)


def test_script_creates_all_outputs(tmp_path: Path) -> None:
    prediction_csv = tmp_path / "predictions.csv"
    out_dir = tmp_path / "figures"
    pd.DataFrame(
        {
            "actual_value": [0.1, 0.25, 0.5, 0.8, None],
            "predicted_value": [0.15, 0.2, 0.45, 0.7, 0.3],
        }
    ).to_csv(prediction_csv, index=False)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prediction-csv",
            str(prediction_csv),
            "--out-dir",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for filename in [
        "qy_predicted_vs_actual_scatter.png",
        "qy_predicted_vs_actual_scatter.pdf",
        "qy_predicted_vs_actual_metrics.csv",
        "qy_predicted_vs_actual_report.md",
    ]:
        assert (out_dir / filename).exists()
