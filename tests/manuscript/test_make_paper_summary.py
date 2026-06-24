"""Tests for the manuscript paper-summary generator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "manuscript" / "make_paper_summary.py"


def test_make_paper_summary_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0
    assert "--paper-results-dir" in completed.stdout
    assert "--original-chemfluor-reference" in completed.stdout


def write_minimal_audit(results_dir: Path) -> None:
    """Write enough audit data to populate the current-workflow scale row."""
    pd.DataFrame(
        [
            ["overall", "all", "total_rows", 100],
            ["overall", "all", "unique_canonical_chromophores", 80],
            ["overall", "all", "unique_canonical_solvents", 12],
            ["rows_by_source", "chemfluor", "rows", 100],
            ["target_coverage", "all", "emission_nm", 90],
            ["target_coverage", "all", "quantum_yield", 70],
            ["target_coverage", "all", "absorption_nm", 85],
            ["deduplication", "all", "rows_before_deduplication", 110],
        ],
        columns=["section", "source_dataset", "metric", "value"],
    ).to_csv(results_dir / "dataset_audit.csv", index=False)


def test_missing_optional_files_warn_instead_of_crashing(tmp_path: Path) -> None:
    results_dir = tmp_path / "partial_results"
    results_dir.mkdir()
    write_minimal_audit(results_dir)
    out_md = tmp_path / "summary.md"
    out_csv = tmp_path / "tables"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--paper-results-dir",
            str(results_dir),
            "--out-md",
            str(out_md),
            "--out-csv-dir",
            str(out_csv),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "WARNING:" in completed.stderr
    assert "model metrics not found" in completed.stderr
    assert out_md.exists()
    markdown = out_md.read_text(encoding="utf-8")
    assert "Main manuscript claims supported by results" in markdown
    assert "does not establish superiority" in markdown
    assert "beats the original" not in markdown.lower()


def test_requested_csv_tables_are_written(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    write_minimal_audit(results_dir)
    pd.DataFrame(
        [
            {
                "target": "emission_nm",
                "model": "rf",
                "split": "scaffold",
                "seed": 0,
                "mae": 45.0,
                "rmse": 60.0,
                "r2": 0.5,
                "mae_ev": 0.2,
                "train_rows": 80,
                "test_rows": 20,
            },
            {
                "target": "quantum_yield",
                "model": "rf",
                "split": "scaffold",
                "seed": 0,
                "mae": 0.1,
                "rmse": 0.15,
                "r2": 0.4,
                "train_rows": 60,
                "test_rows": 15,
            },
        ]
    ).to_csv(results_dir / "metrics_by_split_model_target.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "rf",
                "split": "scaffold",
                "seed": 0,
                "region": "red/NIR",
                "rows": 5,
                "mae": 80.0,
            },
            {
                "model": "rf",
                "split": "scaffold",
                "seed": 0,
                "region": "green",
                "rows": 8,
                "mae": 30.0,
            },
        ]
    ).to_csv(results_dir / "region_metrics_by_split_model.csv", index=False)
    out_csv = tmp_path / "custom_tables"
    out_md = tmp_path / "custom_summary.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--paper-results-dir",
            str(results_dir),
            "--out-md",
            str(out_md),
            "--out-csv-dir",
            str(out_csv),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    expected = [
        "table_1_dataset_scale.csv",
        "table_2_methodology.csv",
        "table_3_emission_models.csv",
        "table_4_quantum_yield_models.csv",
        "table_5_wavelength_regions.csv",
        "table_6_graph_models.csv",
    ]
    for filename in expected:
        assert (out_csv / filename).exists()
    region_table = pd.read_csv(out_csv / "table_5_wavelength_regions.csv")
    assert region_table.loc[0, "worst_region"] == "red_NIR"
    markdown = out_md.read_text(encoding="utf-8")
    assert "Red/NIR emission remains" in markdown
    assert "Figure 7. Applicability-domain benchmark prediction." in markdown
