from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "diagnose_external_benchmark.py"

spec = importlib.util.spec_from_file_location("diagnose_external_benchmark", SCRIPT_PATH)
diagnostics = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["diagnose_external_benchmark"] = diagnostics
spec.loader.exec_module(diagnostics)


def write_prediction(path: Path, molecule: str, expected: float, rows: list[tuple[str, str, float, float]]) -> None:
    pd.DataFrame(
        [
            {
                "molecule": molecule,
                "molecule_smiles": "CCO" if molecule != "Novel" else "CCCCCCCC",
                "solvent": "water",
                "solvent_smiles": "O",
                "expected_emission_nm": expected,
                "expected_qy": 0.5,
                "model": model,
                "model_family": family,
                "predicted_emission_nm": pred,
                "predicted_quantum_yield": qy,
                "nearest_training_similarity": 1.0 if molecule != "Novel" else 0.25,
                "nearest_training_smiles": "CCO",
            }
            for model, family, pred, qy in rows
        ]
    ).to_csv(path, index=False)


def args(tmp_path: Path, prediction_dir: Path, training_csv: Path) -> object:
    return diagnostics.parse_args(
        [
            "--prediction-dir",
            str(prediction_dir),
            "--training-csv",
            str(training_csv),
            "--out-dir",
            str(tmp_path / "diagnostics"),
        ]
    )


def test_prediction_consolidation_and_errors(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    write_prediction(pred_dir / "MolA.csv", "MolA", 500, [("good", "tree", 505, 0.45)])
    write_prediction(pred_dir / "MolB.csv", "MolB", 600, [("bad", "tree", 650, 0.1)])
    training = tmp_path / "training.csv"
    pd.DataFrame({"smiles": ["CCO"], "solvent_smiles": ["O"], "emission_nm": [500]}).to_csv(training, index=False)

    combined = diagnostics.consolidate_predictions(args(tmp_path, pred_dir, training), [])

    assert len(combined) == 2
    assert combined.loc[combined["molecule"] == "MolA", "emission_abs_error_nm"].iloc[0] == pytest.approx(5)
    assert combined.loc[combined["molecule"] == "MolB", "quantum_yield_abs_error"].iloc[0] == pytest.approx(0.4)


def test_model_summary_ranks_lower_error_first(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "molecule": ["A", "A"],
            "model": ["better", "worse"],
            "model_family": ["tree", "tree"],
            "emission_abs_error_nm": [2.0, 20.0],
            "quantum_yield_abs_error": [0.1, 0.05],
        }
    )

    summary = diagnostics.aggregate_errors(table, ["model", "model_family"])

    assert summary.iloc[0]["model"] == "better"


def test_training_overlap_detects_molecule_and_pair(tmp_path: Path) -> None:
    combined = pd.DataFrame(
        {
            "molecule": ["Ethanol"],
            "input_smiles": ["CCO"],
            "input_solvent": ["water"],
            "input_solvent_smiles": ["O"],
        }
    )
    training = tmp_path / "training.csv"
    pd.DataFrame(
        {
            "molecule_smiles": ["OCC", "CCN"],
            "solvent": ["water", "ethanol"],
            "solvent_smiles": ["O", "CCO"],
            "emission_nm": [505, 600],
            "quantum_yield": [0.4, 0.2],
        }
    ).to_csv(training, index=False)

    overlap = diagnostics.training_overlap(combined, training, [])

    assert bool(overlap.loc[0, "exact_molecule_seen"])
    assert bool(overlap.loc[0, "exact_molecule_solvent_pair_seen"])
    assert overlap.loc[0, "training_emission_median_same_molecule_solvent"] == pytest.approx(505)


def classify_one(molecule_row: dict, overlap_row: dict, tmp_path: Path) -> str:
    parsed = diagnostics.parse_args(
        [
            "--prediction-dir",
            str(tmp_path),
            "--training-csv",
            str(tmp_path / "training.csv"),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    result = diagnostics.classify_failures(
        pd.DataFrame([molecule_row]), pd.DataFrame([overlap_row]), parsed
    )
    return str(result.loc[0, "failure_mode"])


def base_molecule(**updates: object) -> dict:
    row = {
        "molecule": "A",
        "expected_emission_nm": 500.0,
        "expected_quantum_yield": 0.5,
        "mean_emission_abs_error_nm": 5.0,
        "best_emission_abs_error_nm": 4.0,
        "mean_quantum_yield_abs_error": 0.05,
        "predicted_emission_mean": 505.0,
        "predicted_emission_std": 5.0,
        "predicted_emission_min": 500.0,
        "predicted_emission_max": 510.0,
        "max_nearest_training_similarity": 1.0,
    }
    row.update(updates)
    return row


def base_overlap(**updates: object) -> dict:
    row = {
        "molecule": "A",
        "exact_molecule_seen": True,
        "exact_solvent_seen": True,
        "exact_molecule_solvent_pair_seen": True,
        "training_emission_median_same_molecule": 500.0,
        "training_emission_median_same_molecule_solvent": 500.0,
        "training_emission_std_same_molecule_solvent": 2.0,
        "training_qy_median_same_molecule_solvent": 0.5,
    }
    row.update(updates)
    return row


def test_failure_mode_classification_core_cases(tmp_path: Path) -> None:
    assert classify_one(base_molecule(), base_overlap(), tmp_path) == "reasonable_prediction"
    assert (
        classify_one(
            base_molecule(mean_emission_abs_error_nm=70, predicted_emission_mean=560),
            base_overlap(training_emission_median_same_molecule_solvent=560),
            tmp_path,
        )
        == "benchmark_training_label_mismatch"
    )
    assert (
        classify_one(
            base_molecule(mean_emission_abs_error_nm=55),
            base_overlap(exact_molecule_solvent_pair_seen=False),
            tmp_path,
        )
        == "solvent_or_condition_mismatch"
    )
    assert (
        classify_one(
            base_molecule(mean_emission_abs_error_nm=80, max_nearest_training_similarity=0.2),
            base_overlap(exact_molecule_seen=False, exact_molecule_solvent_pair_seen=False),
            tmp_path,
        )
        == "structural_extrapolation"
    )


def test_script_runs_end_to_end(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    write_prediction(
        pred_dir / "MolA.csv",
        "MolA",
        500,
        [("good", "tree", 505, 0.45), ("bad", "tree", 550, 0.2)],
    )
    training = tmp_path / "training.csv"
    pd.DataFrame(
        {
            "molecule_smiles": ["CCO"],
            "solvent": ["water"],
            "solvent_smiles": ["O"],
            "emission_nm": [500],
            "quantum_yield": [0.5],
        }
    ).to_csv(training, index=False)

    parsed = args(tmp_path, pred_dir, training)
    paths = diagnostics.run(parsed)

    for key in [
        "all_predictions",
        "model_summary",
        "family_summary",
        "molecule_summary",
        "training_overlap",
        "failure_modes",
        "report",
    ]:
        assert paths[key].exists()
