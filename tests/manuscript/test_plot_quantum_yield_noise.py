"""Tests for the manuscript quantum-yield noise analysis."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.manuscript.plot_quantum_yield_noise import (
    compute_group_statistics,
    prepare_qy_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "manuscript" / "plot_quantum_yield_noise.py"


def fixture_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "canonical_chromophore_smiles": ["A", "A", "A", "B", "B", "C"],
            "canonical_solvent_smiles": ["O", "O", "N", "O", "O", "O"],
            "quantum_yield": [0.1, 0.3, 0.8, 0.4, 1.2, -0.1],
            "source_dataset": ["one", "two", "one", "one", "one", "two"],
        }
    )


def test_grouping_computes_range_and_sample_standard_deviation() -> None:
    valid, _, columns = prepare_qy_data(fixture_data())
    summary = compute_group_statistics(valid, columns)
    group = summary.query("canonical_chromophore_smiles == 'A' and canonical_solvent_smiles == 'O'").iloc[0]
    assert group["n_qy_records"] == 2
    assert group["qy_range"] == pytest.approx(0.2)
    assert group["std_qy"] == pytest.approx(np.std([0.1, 0.3], ddof=1))


def test_invalid_qy_values_are_excluded() -> None:
    valid, excluded, _ = prepare_qy_data(fixture_data())
    assert valid["quantum_yield"].between(0, 1).all()
    assert sorted(excluded["quantum_yield"].tolist()) == [-0.1, 1.2]


def test_repeated_molecule_solvent_groups_are_detected() -> None:
    valid, _, columns = prepare_qy_data(fixture_data())
    summary = compute_group_statistics(valid, columns)
    repeated = summary.loc[summary["n_qy_records"] >= 2]
    assert len(repeated) == 1
    assert repeated.iloc[0]["canonical_chromophore_smiles"] == "A"


def test_script_smoke_creates_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "tiny.csv"
    out_dir = tmp_path / "outputs"
    fixture_data().to_csv(input_path, index=False)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(input_path), "--out-dir", str(out_dir)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for filename in [
        "qy_replicate_noise_summary.csv",
        "top_noisy_qy_groups.csv",
        "qy_noise_report.md",
    ]:
        assert (out_dir / filename).exists()
    assert list(out_dir.glob("*.png"))
