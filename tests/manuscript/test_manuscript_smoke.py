"""CLI smoke test for the manuscript results pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_manuscript_pipeline_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "paper_comparison_smoke"
    command = [
        sys.executable,
        "scripts/manuscript/run_paper_comparison_experiments.py",
        "--max-rows",
        "200",
        "--models",
        "rf",
        "--targets",
        "emission_nm",
        "--splits",
        "random",
        "--seeds",
        "0",
        "--out-dir",
        str(out_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    expected = [
        "dataset_audit.csv",
        "dataset_audit.md",
        "split_leakage_report.csv",
        "metrics_by_split_model_target.csv",
        "metrics_with_bootstrap_ci.csv",
        "region_metrics_by_split_model.csv",
        "qy_classifier_metrics.csv",
        "paper_tables.md",
    ]
    for filename in expected:
        assert (out_dir / filename).exists()
    assert list((out_dir / "predictions").glob("*.csv"))
    assert list((out_dir / "figures").glob("*.png"))
