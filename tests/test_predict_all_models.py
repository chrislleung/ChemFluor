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
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "predict_all_models.py"

spec = importlib.util.spec_from_file_location("predict_all_models", SCRIPT_PATH)
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


def write_standardized_combined(tmp_path: Path) -> Path:
    path = tmp_path / "combined.csv"
    pd.DataFrame({"canonical_chromophore_smiles": ["CCO", "c1ccccc1"]}).to_csv(
        path, index=False
    )
    return path


def write_tree_model_root(
    tmp_path: Path,
    *,
    include_emission: bool = True,
    include_qy: bool = True,
) -> Path:
    root = tmp_path / "tree_models"
    model_dir = root / "rf"
    model_dir.mkdir(parents=True)
    metadata = {
        "fingerprint_radius": 2,
        "fingerprint_n_bits": 32,
        "solvent_descriptor_columns_used": ["dielectric_constant"],
        "median_values_used_for_imputation": {
            "emission_nm": {"dielectric_constant": 1.0},
            "quantum_yield": {"dielectric_constant": 1.0},
        },
        "model_type": "rf",
        "target_columns": ["emission_nm", "quantum_yield"],
    }
    (model_dir / "feature_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    pd.DataFrame({"canonical_chromophore_smiles": ["CCO", "c1ccccc1"]}).to_csv(
        model_dir / "combined_modeling_rows_after_feature_merge.csv", index=False
    )
    if include_emission:
        joblib.dump(ConstantRegressor(456.0), model_dir / "emission_nm_rf.joblib")
    if include_qy:
        joblib.dump(ConstantRegressor(0.42), model_dir / "quantum_yield_rf.joblib")
    return root


def write_graph_artifact_dir(
    root: Path,
    *,
    model_name: str = "graph_gin",
    seed: int | None = 0,
    targets: tuple[str, ...] = ("emission_nm",),
) -> Path:
    model_dir = root / model_name
    model_dir.mkdir(parents=True)
    metadata = {
        "model_type": model_name,
        "model_family": "graph_neural",
        "seed": seed,
        "solvent_descriptor_columns_used": ["dielectric_constant"],
        "median_values_used_for_imputation": {
            target: {"dielectric_constant": 1.0} for target in targets
        },
    }
    (model_dir / "feature_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    pd.DataFrame({"canonical_chromophore_smiles": ["CCO"]}).to_csv(
        model_dir / "combined_modeling_rows_after_feature_merge.csv", index=False
    )
    for target in targets:
        (model_dir / f"{target}_{model_name}.pt").write_text("mock", encoding="utf-8")
    return model_dir


class FakeGraphHelpers:
    @staticmethod
    def import_torch():
        return object()

    @staticmethod
    def predict_graph_target(model_dir, model_name, target, graph, vector, torch):
        values = {"emission_nm": 500.0, "quantum_yield": 0.33}
        return values[target]


def run_main(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["predict_all_models.py", *argv])
    return int(predict_script.main())


def base_args(tmp_path: Path, tree_root: Path) -> list[str]:
    return [
        "--solvent-descriptors",
        str(write_solvent_descriptors(tmp_path)),
        "--standardized-combined",
        str(write_standardized_combined(tmp_path)),
        "--tree-model-dir",
        str(tree_root),
        "--neural-model-dir",
        str(tmp_path / "missing_neural"),
        "--graph-model-dirs",
    ]


def test_cli_help(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["predict_all_models.py", "--help"])

    with pytest.raises(SystemExit) as exc:
        predict_script.parse_args()

    assert exc.value.code == 0
    assert "--smiles" in capsys.readouterr().out


def test_invalid_smiles(tmp_path: Path, monkeypatch, capsys) -> None:
    tree_root = write_tree_model_root(tmp_path)

    exit_code = run_main(
        monkeypatch,
        [
            "--smiles",
            "not_a_smiles",
            "--solvent-smiles",
            "O",
            *base_args(tmp_path, tree_root),
        ],
    )

    assert exit_code == 1
    assert "Invalid molecule SMILES" in capsys.readouterr().err


def test_missing_solvent(tmp_path: Path, monkeypatch, capsys) -> None:
    tree_root = write_tree_model_root(tmp_path)

    exit_code = run_main(
        monkeypatch,
        [
            "--smiles",
            "CCO",
            *base_args(tmp_path, tree_root),
        ],
    )

    assert exit_code == 1
    assert "Provide either --solvent or --solvent-smiles" in capsys.readouterr().err


def test_output_csv_creation(tmp_path: Path, monkeypatch) -> None:
    tree_root = write_tree_model_root(tmp_path)
    output_csv = tmp_path / "predictions.csv"

    exit_code = run_main(
        monkeypatch,
        [
            "--smiles",
            "CCO",
            "--solvent",
            "water",
            "--out",
            str(output_csv),
            *base_args(tmp_path, tree_root),
        ],
    )

    table = pd.read_csv(output_csv)
    assert exit_code == 0
    assert len(table) == 1
    assert table["predicted_emission_nm"].dropna().unique().tolist() == [456.0]
    assert table["predicted_quantum_yield"].dropna().unique().tolist() == [0.42]
    assert "target" not in table.columns


def test_disagreement_summary_correctness() -> None:
    table = pd.DataFrame(
        [
            {
                "model": "a",
                "model_family": "tree",
                "seed": None,
                "predicted_emission_nm": 400.0,
                "predicted_quantum_yield": 0.2,
            },
            {
                "model": "b",
                "model_family": "tree",
                "seed": None,
                "predicted_emission_nm": 500.0,
                "predicted_quantum_yield": 0.6,
            },
        ]
    )

    summaries = predict_script.compute_disagreement_summaries(table)

    assert summaries["all_emission"]["mean"] == pytest.approx(450.0)
    assert summaries["all_emission"]["median"] == pytest.approx(450.0)
    assert summaries["all_emission"]["std"] == pytest.approx(50.0)
    assert summaries["all_emission"]["min"] == pytest.approx(400.0)
    assert summaries["all_emission"]["max"] == pytest.approx(500.0)
    assert summaries["all_emission"]["range"] == pytest.approx(100.0)
    assert summaries["all_quantum_yield"]["range"] == pytest.approx(0.4)


def test_missing_model_files_are_skipped_with_warning(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    tree_root = write_tree_model_root(tmp_path, include_qy=False)

    exit_code = run_main(
        monkeypatch,
        [
            "--smiles",
            "CCO",
            "--solvent-smiles",
            "O",
            *base_args(tmp_path, tree_root),
        ],
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Model file not found; skipping" in captured.out
    assert "quantum_yield_rf.joblib" in captured.out
    assert "emission_nm" in captured.out


def test_direct_graph_artifact_directory_is_discovered(tmp_path: Path) -> None:
    graph_dir = write_graph_artifact_dir(tmp_path / "models" / "run" / "seed_0")

    assert predict_script.discover_graph_model_dirs([graph_dir]) == [graph_dir]


def test_parent_graph_model_directory_is_discovered(tmp_path: Path) -> None:
    parent = tmp_path / "models" / "run" / "seed_0"
    graph_dir = write_graph_artifact_dir(parent)

    assert predict_script.discover_graph_model_dirs([parent]) == [graph_dir]


def test_outputs_graph_directory_resolves_to_models_directory(tmp_path: Path) -> None:
    model_parent = tmp_path / "models" / "run" / "seed_0"
    output_parent = tmp_path / "outputs" / "run" / "seed_0"
    output_parent.mkdir(parents=True)
    graph_dir = write_graph_artifact_dir(model_parent)

    assert predict_script.discover_graph_model_dirs([output_parent]) == [graph_dir]


def test_graph_single_target_folder_has_no_missing_target_warning(
    tmp_path: Path, monkeypatch
) -> None:
    graph_dir = write_graph_artifact_dir(tmp_path / "models" / "run" / "seed_0")
    solvent_descriptors = pd.read_csv(write_solvent_descriptors(tmp_path))
    solvent_row = solvent_descriptors.iloc[0]
    monkeypatch.setattr(predict_script, "import_graph_helpers", lambda: FakeGraphHelpers)

    predictions, warnings = predict_script.predict_graph_targets(
        graph_dir, "graph_gin", "CCO", solvent_row
    )

    assert predictions == {"emission_nm": 500.0}
    assert not any("quantum_yield" in warning for warning in warnings)


def test_known_emission_error_calculation(tmp_path: Path, monkeypatch) -> None:
    tree_root = write_tree_model_root(tmp_path)
    output_csv = tmp_path / "predictions.csv"

    exit_code = run_main(
        monkeypatch,
        [
            "--smiles",
            "CCO",
            "--solvent",
            "water",
            "--known-emission-nm",
            "450",
            "--out",
            str(output_csv),
            *base_args(tmp_path, tree_root),
        ],
    )

    table = pd.read_csv(output_csv)
    assert exit_code == 0
    assert table.loc[0, "emission_abs_error_nm"] == pytest.approx(6.0)


def test_disagreement_summaries_by_model_family() -> None:
    table = pd.DataFrame(
        [
            {"model": "rf", "model_family": "tree", "seed": None, "predicted_emission_nm": 400.0, "predicted_quantum_yield": 0.2},
            {"model": "mlp", "model_family": "neural", "seed": None, "predicted_emission_nm": 410.0, "predicted_quantum_yield": 0.3},
            {"model": "graph_gin", "model_family": "graph_neural", "seed": 0, "predicted_emission_nm": 500.0, "predicted_quantum_yield": 0.4},
            {"model": "graph_gcn", "model_family": "graph_neural", "seed": 1, "predicted_emission_nm": 530.0, "predicted_quantum_yield": np.nan},
        ]
    )

    summaries = predict_script.compute_disagreement_summaries(table)

    assert summaries["all_emission"]["range"] == pytest.approx(130.0)
    assert summaries["tree_neural_emission"]["range"] == pytest.approx(10.0)
    assert summaries["graph_emission"]["range"] == pytest.approx(30.0)
    assert summaries["graph_gin_emission"]["mean"] == pytest.approx(500.0)
    assert summaries["graph_gcn_emission"]["mean"] == pytest.approx(530.0)
    assert summaries["graph_quantum_yield"]["mean"] == pytest.approx(0.4)
