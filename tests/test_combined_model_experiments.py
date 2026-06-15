from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


trainer = load_script("train_combined_predictors_for_tests", "scripts/train_combined_predictors.py")
predictor = load_script("predict_combined_molecule_for_tests", "scripts/predict_combined_molecule.py")
experiments = load_script(
    "run_combined_model_experiments_for_tests",
    "scripts/run_combined_model_experiments.py",
)


class ConstantRegressor:
    def __init__(self, value: float, n_features: int = 33) -> None:
        self.value = value
        self.n_features_in_ = n_features

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.value, dtype=float)


def write_feature_metadata(model_dir: Path, model_type: str = "histgb") -> None:
    metadata = {
        "fingerprint_radius": 2,
        "fingerprint_n_bits": 32,
        "solvent_descriptor_columns_used": ["dielectric_constant"],
        "target_columns": ["emission_nm", "quantum_yield"],
        "model_type": model_type,
        "median_values_used_for_imputation": {
            "emission_nm": {"dielectric_constant": 1.0},
            "quantum_yield": {"dielectric_constant": 1.0},
        },
    }
    (model_dir / "feature_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def test_train_script_accepts_new_model_names(monkeypatch) -> None:
    for model_name in trainer.MODEL_TYPES:
        monkeypatch.setattr(
            sys,
            "argv",
            ["train_combined_predictors.py", "--model", model_name],
        )
        args = trainer.parse_args()
        assert args.model == model_name


def test_optional_unavailable_model_library_is_skipped_gracefully(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("mock missing lightgbm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert trainer.make_model("lightgbm") is None


def test_prediction_script_loads_selected_model_type(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    joblib.dump(ConstantRegressor(500.0), model_dir / "emission_nm_histgb.joblib")

    models, warnings = predictor.load_available_models(model_dir, model_type="histgb")

    assert set(models) == {"emission_nm"}
    assert models["emission_nm"].predict(np.zeros((1, 33)))[0] == 500.0
    assert any("absorption_nm_histgb.joblib" in warning for warning in warnings)


def test_model_comparison_csv_is_created_from_mock_metrics_dirs(tmp_path: Path) -> None:
    rf_dir = tmp_path / "rf"
    histgb_dir = tmp_path / "histgb"
    rf_dir.mkdir()
    histgb_dir.mkdir()
    (rf_dir / "metrics.json").write_text(
        json.dumps(
            {
                "emission_nm": {
                    "target": "emission_nm",
                    "mae": 25.0,
                    "rmse": 35.0,
                    "r2": 0.4,
                    "train_rows": 10,
                    "test_rows": 5,
                }
            }
        ),
        encoding="utf-8",
    )
    (histgb_dir / "metrics.json").write_text(
        json.dumps(
            {
                "emission_nm": {
                    "target": "emission_nm",
                    "mae": 20.0,
                    "rmse": 30.0,
                    "r2": 0.5,
                    "train_rows": 10,
                    "test_rows": 5,
                }
            }
        ),
        encoding="utf-8",
    )

    comparison = experiments.collect_model_metrics(
        {"rf": rf_dir, "histgb": histgb_dir}
    )
    compare_out = tmp_path / "compare"
    experiments.write_outputs(compare_out, comparison, pd.DataFrame(), pd.DataFrame())

    saved = pd.read_csv(compare_out / "model_comparison.csv")
    assert saved.loc[0, "model"] == "histgb"
    assert saved.loc[0, "mae"] == 20.0
    assert (compare_out / "model_comparison.md").exists()


def test_wavelength_region_error_comparison_tiny_prediction_csv() -> None:
    predictions = pd.DataFrame(
        {
            "y_true": [390.0, 450.0, 525.0, 575.0, 650.0],
            "y_pred": [380.0, 470.0, 500.0, 600.0, 610.0],
        }
    )

    region_errors = experiments.compute_error_by_region(predictions, "rf")

    red = region_errors[region_errors["wavelength_region"] == "red/NIR"].iloc[0]
    blue = region_errors[region_errors["wavelength_region"] == "blue"].iloc[0]
    assert red["mean_absolute_error"] == pytest.approx(40.0)
    assert blue["mean_absolute_error"] == pytest.approx(20.0)
    assert set(region_errors["wavelength_region"]) == set(experiments.REGION_ORDER)


def test_benchmark_comparison_handles_missing_known_values(tmp_path: Path) -> None:
    model_dir = tmp_path / "histgb"
    model_dir.mkdir()
    write_feature_metadata(model_dir, model_type="histgb")
    joblib.dump(ConstantRegressor(500.0), model_dir / "emission_nm_histgb.joblib")
    pd.DataFrame({"canonical_chromophore_smiles": ["CCO"]}).to_csv(
        model_dir / "combined_modeling_rows_after_feature_merge.csv", index=False
    )
    solvent_descriptors = tmp_path / "solvents.csv"
    pd.DataFrame(
        {
            "canonical_solvent_smiles": ["O"],
            "solvent_original": ["water"],
            "dielectric_constant": [80.1],
        }
    ).to_csv(solvent_descriptors, index=False)
    args = type(
        "Args",
        (),
        {
            "benchmark_smiles": "CCO",
            "benchmark_solvent_smiles": "O",
            "known_emission_nm": None,
            "known_quantum_yield": None,
            "solvent_descriptors": solvent_descriptors,
        },
    )()

    comparison = experiments.collect_benchmark_predictions({"histgb": model_dir}, args)

    assert len(comparison) == 1
    assert comparison.loc[0, "predicted_emission_nm"] == pytest.approx(500.0)
    assert pd.isna(comparison.loc[0, "emission_absolute_error"])
