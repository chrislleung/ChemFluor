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
combined_prediction = sys.modules["chemfluor.combined_prediction"]


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


@pytest.mark.parametrize(
    ("similarity", "label"),
    [
        (0.75, "high"),
        (0.55, "medium"),
        (0.45, "low-medium"),
        (0.20, "low"),
        (None, "unknown"),
        (float("nan"), "unknown"),
    ],
)
def test_similarity_confidence_label(similarity: float | None, label: str) -> None:
    assert combined_prediction.similarity_confidence_label(similarity) == label


def test_known_value_errors_are_computed_correctly() -> None:
    errors = predict_script.compute_prediction_errors(
        predictions={"emission_nm": 461.439, "quantum_yield": 0.135278},
        known_values={"emission_nm": 539.0, "quantum_yield": 0.196},
    )

    assert errors["emission_nm"]["residual"] == pytest.approx(-77.561)
    assert errors["emission_nm"]["absolute_error"] == pytest.approx(77.561)
    assert errors["quantum_yield"]["residual"] == pytest.approx(-0.060722)
    assert errors["quantum_yield"]["absolute_error"] == pytest.approx(0.060722)


def test_csv_flattening_includes_known_error_residual_and_confidence_columns() -> None:
    payload = {
        "name": "candidate",
        "predictions": {"emission_nm": 461.439},
        "known_values": {"emission_nm": 539.0},
        "errors": {
            "emission_nm": {
                "predicted": 461.439,
                "known": 539.0,
                "residual": -77.561,
                "absolute_error": 77.561,
            }
        },
        "applicability_domain": {
            "confidence_label": "low-medium",
            "confidence_interpretation": (
                "Candidate is only weakly similar; prediction may be extrapolative."
            ),
        },
    }

    row = predict_script.flatten_for_csv(payload)

    assert row.loc[0, "emission_nm"] == pytest.approx(461.439)
    assert row.loc[0, "known_emission_nm"] == pytest.approx(539.0)
    assert row.loc[0, "error_emission_nm"] == pytest.approx(77.561)
    assert row.loc[0, "residual_emission_nm"] == pytest.approx(-77.561)
    assert row.loc[0, "confidence_label"] == "low-medium"
    assert "extrapolative" in row.loc[0, "confidence_interpretation"]


def test_disabled_applicability_domain_does_not_crash(
    tmp_path: Path, monkeypatch
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
            "--no-applicability-domain",
            "--out",
            str(output_json),
        ],
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["applicability_domain"]["confidence_label"] == "unknown"


def test_exact_reference_match_returns_zero_without_reference(tmp_path: Path) -> None:
    model_dir = write_mock_model_dir(tmp_path, include_reference=False)

    matches = predict_script.exact_reference_matches_payload("CCO", model_dir)

    assert matches == {"count": 0, "rows": []}


def test_exact_reference_match_returns_zero_without_match(tmp_path: Path) -> None:
    model_dir = write_mock_model_dir(tmp_path)

    matches = predict_script.exact_reference_matches_payload("CCN", model_dir)

    assert matches == {"count": 0, "rows": []}


def test_exact_reference_match_finds_tiny_mock_reference(tmp_path: Path) -> None:
    model_dir = write_mock_model_dir(tmp_path)
    pd.DataFrame(
        {
            "canonical_chromophore_smiles": ["CCO", "CCO", "c1ccccc1"],
            "canonical_solvent_smiles": ["O", "CCO", "O"],
            "emission_nm": [539.0, 540.0, 300.0],
            "quantum_yield": [0.196, 0.2, 0.1],
            "source_dataset": ["mock", "mock2", "mock"],
            "fluodb_source": ["source_a", "source_b", "source_c"],
            "tag_name": ["tag_a", "tag_b", "tag_c"],
        }
    ).to_csv(model_dir / "combined_modeling_rows_after_feature_merge.csv", index=False)

    matches = predict_script.exact_reference_matches_payload("CCO", model_dir)

    assert matches["count"] == 2
    assert len(matches["rows"]) == 2
    assert matches["rows"][0]["canonical_chromophore_smiles"] == "CCO"
    assert matches["rows"][0]["emission_nm"] == 539.0
