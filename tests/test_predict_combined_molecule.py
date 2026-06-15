from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "predict_combined_molecule.py"

spec = importlib.util.spec_from_file_location("predict_combined_molecule", SCRIPT_PATH)
predict_script = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(predict_script)


class ConstantRegressor:
    def __init__(self, value: float, n_features: int = 33) -> None:
        self.value = value
        self.n_features_in_ = n_features

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert features.shape[1] == self.n_features_in_
        return np.full(features.shape[0], self.value, dtype=float)


def write_mock_model_dir(tmp_path: Path, *, include_reference: bool = True) -> Path:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    metadata = {
        "fingerprint_radius": 2,
        "fingerprint_n_bits": 32,
        "solvent_descriptor_columns_used": ["dielectric_constant"],
        "target_columns": [
            "absorption_nm",
            "emission_nm",
            "lifetime_ns",
            "quantum_yield",
            "log_extinction",
        ],
        "model_type": "rf",
        "median_values_used_for_imputation": {
            "absorption_nm": {"dielectric_constant": 1.0},
            "emission_nm": {"dielectric_constant": 1.0},
        },
    }
    (model_dir / "feature_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    joblib.dump(ConstantRegressor(321.0), model_dir / "absorption_nm_rf.joblib")
    joblib.dump(ConstantRegressor(456.0), model_dir / "emission_nm_rf.joblib")
    if include_reference:
        pd.DataFrame({"canonical_chromophore_smiles": ["CCO", "c1ccccc1"]}).to_csv(
            model_dir / "combined_modeling_rows_after_feature_merge.csv", index=False
        )
    return model_dir


def write_solvent_descriptors(tmp_path: Path) -> Path:
    path = tmp_path / "solvent_descriptors.csv"
    pd.DataFrame(
        {
            "canonical_solvent_smiles": ["O", "CCO"],
            "solvent_original": ["water", "ethanol"],
            "dielectric_constant": [80.1, 24.6],
        }
    ).to_csv(path, index=False)
    return path


def run_main(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["predict_combined_molecule.py", *argv])
    return int(predict_script.main())


def test_invalid_smiles_exits_with_helpful_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    model_dir = write_mock_model_dir(tmp_path)
    solvent_descriptors = write_solvent_descriptors(tmp_path)

    exit_code = run_main(
        monkeypatch,
        [
            "--smiles",
            "not_a_smiles",
            "--solvent-smiles",
            "O",
            "--model-dir",
            str(model_dir),
            "--solvent-descriptors",
            str(solvent_descriptors),
            "--out",
            str(tmp_path / "prediction.json"),
        ],
    )

    assert exit_code == 1
    assert "Invalid molecule SMILES" in capsys.readouterr().err


def test_missing_model_directory_exits_with_helpful_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    solvent_descriptors = write_solvent_descriptors(tmp_path)

    exit_code = run_main(
        monkeypatch,
        [
            "--smiles",
            "CCO",
            "--solvent-smiles",
            "O",
            "--model-dir",
            str(tmp_path / "missing_models"),
            "--solvent-descriptors",
            str(solvent_descriptors),
            "--out",
            str(tmp_path / "prediction.json"),
        ],
    )

    assert exit_code == 1
    assert "Model directory not found" in capsys.readouterr().err


def test_feature_building_works_for_mocked_metadata(tmp_path: Path) -> None:
    solvent_descriptors = write_solvent_descriptors(tmp_path)
    descriptors = predict_script.load_solvent_descriptors(solvent_descriptors)
    solvent_row = predict_script.get_solvent_descriptor_row(descriptors, "O", "O")

    features = predict_script.build_single_feature_matrix(
        canonical_smiles="CCO",
        solvent_descriptor_row=solvent_row,
        descriptor_columns=["dielectric_constant"],
        medians={"dielectric_constant": 1.0},
        radius=2,
        n_bits=32,
    )

    assert features.shape == (1, 33)
    assert features[0, -1] == pytest.approx(80.1)


def test_missing_target_model_is_skipped_and_json_contains_domain(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    model_dir = write_mock_model_dir(tmp_path)
    solvent_descriptors = write_solvent_descriptors(tmp_path)
    output_json = tmp_path / "prediction.json"

    exit_code = run_main(
        monkeypatch,
        [
            "--smiles",
            "CCO",
            "--solvent-smiles",
            "O",
            "--model-dir",
            str(model_dir),
            "--solvent-descriptors",
            str(solvent_descriptors),
            "--name",
            "ethanol_water",
            "--out",
            str(output_json),
        ],
    )

    captured = capsys.readouterr()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["name"] == "ethanol_water"
    assert payload["canonical_smiles"] == "CCO"
    assert payload["predictions"] == {"absorption_nm": 321.0, "emission_nm": 456.0}
    assert "quantum_yield" not in payload["predictions"]
    assert payload["applicability_domain"]["nearest_training_similarity"] == 1.0
    assert payload["applicability_domain"]["outside_applicability_domain"] is False
    assert "Model not found for quantum_yield; skipping" in captured.out


def test_out_csv_writes_one_row(tmp_path: Path, monkeypatch) -> None:
    model_dir = write_mock_model_dir(tmp_path)
    solvent_descriptors = write_solvent_descriptors(tmp_path)
    output_json = tmp_path / "prediction.json"
    output_csv = tmp_path / "prediction.csv"

    exit_code = run_main(
        monkeypatch,
        [
            "--smiles",
            "CCO",
            "--solvent-smiles",
            "O",
            "--model-dir",
            str(model_dir),
            "--solvent-descriptors",
            str(solvent_descriptors),
            "--out",
            str(output_json),
            "--out-csv",
            str(output_csv),
        ],
    )

    csv_rows = pd.read_csv(output_csv)
    assert exit_code == 0
    assert len(csv_rows) == 1
    assert csv_rows.loc[0, "absorption_nm"] == 321.0
    assert csv_rows.loc[0, "emission_nm"] == 456.0
    assert bool(csv_rows.loc[0, "outside_applicability_domain"]) is False
