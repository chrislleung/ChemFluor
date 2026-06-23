"""Run one FluorCast prediction job described by JSON."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback as traceback_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import predict_all_models  # noqa: E402


REQUIRED_FIELDS = (
    "job_id",
    "user_id",
    "molecule_smiles",
    "solvent_smiles",
    "model_choice",
    "requested_at",
)
MODEL_CHOICES = {
    "all",
    "rf",
    "extratrees",
    "gbdt",
    "histgb",
    "graph_model_later",
}
MODEL_AVAILABILITY = {
    "rf": {"artifact_dir": "rf", "experimental": False},
    "extratrees": {"artifact_dir": "extratrees", "experimental": False},
    "gbdt": {"artifact_dir": "gbdt", "experimental": True},
    "histgb": {"artifact_dir": "histgb", "experimental": True},
    "graph_model_later": {"artifact_dir": None, "experimental": True},
}
PredictionBackend = Callable[
    [dict[str, Any]], tuple[list[dict[str, Any]], list[str], str, str]
]


class JobError(Exception):
    """An expected job failure with a stable machine-readable code."""

    def __init__(
        self, code: str, message: str, warnings: list[str] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.warnings = warnings or []


def read_input(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise JobError("INVALID_INPUT", "Input JSON must be an object.")
    return payload


def validate_input(payload: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if not payload.get(field)]
    if missing:
        raise JobError("INVALID_INPUT", f"Missing required field(s): {', '.join(missing)}")
    non_strings = [field for field in REQUIRED_FIELDS if not isinstance(payload[field], str)]
    if non_strings:
        raise JobError(
            "INVALID_INPUT", f"Field(s) must be strings: {', '.join(non_strings)}"
        )
    if payload["model_choice"] not in MODEL_CHOICES:
        raise JobError(
            "INVALID_MODEL_CHOICE",
            "model_choice must be one of: " + ", ".join(sorted(MODEL_CHOICES)),
        )


def _nullable_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _collect_model(
    payload: dict[str, Any], model_name: str
) -> tuple[pd.DataFrame, list[str], str, str]:
    """Run one model artifact directory and report whether it produced rows."""
    artifact_dir = MODEL_AVAILABILITY[model_name]["artifact_dir"]
    if artifact_dir is None:
        raise JobError(
            "MODEL_UNAVAILABLE",
            f"The {model_name} model artifact could not be loaded in the current environment.",
        )

    args = SimpleNamespace(
        smiles=payload["molecule_smiles"],
        solvent=None,
        solvent_smiles=payload["solvent_smiles"],
        solvent_descriptors=PROJECT_ROOT / predict_all_models.DEFAULT_SOLVENT_DESCRIPTORS,
        standardized_combined=PROJECT_ROOT / predict_all_models.DEFAULT_STANDARDIZED_COMBINED,
        tree_model_dir=(
            PROJECT_ROOT / predict_all_models.DEFAULT_TREE_MODEL_DIR / str(artifact_dir)
        ),
        neural_model_dir=PROJECT_ROOT / "models" / "__json_runner_disabled_neural__",
        graph_model_dirs=[],
        known_emission_nm=None,
        known_quantum_yield=None,
        applicability_threshold=predict_all_models.DEFAULT_APPLICABILITY_THRESHOLD,
    )
    table, warnings, canonical_molecule, canonical_solvent, _ = (
        predict_all_models.collect_predictions(args)
    )
    return table, warnings, canonical_molecule, str(canonical_solvent)


def _unavailable_message(model_name: str) -> str:
    return (
        f"Skipped model {model_name}: its artifact could not be loaded in the "
        "current environment. This model is currently unavailable."
    )


def fluorcast_prediction_backend(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], str, str]:
    """Adapt available FluorCast model artifacts to the JSON job contract."""
    choice = str(payload["model_choice"])
    requested_models = (
        ["rf", "extratrees", "gbdt", "histgb", "graph_model_later"]
        if choice == "all"
        else [choice]
    )
    tables = []
    warnings: list[str] = []
    canonical_molecule: str | None = None
    canonical_solvent: str | None = None

    for model_name in requested_models:
        try:
            table, model_warnings, molecule, solvent = _collect_model(
                payload, model_name
            )
        except JobError as exc:
            if choice != "all":
                raise
            warnings.append(_unavailable_message(model_name))
            continue
        canonical_molecule = molecule
        canonical_solvent = solvent
        if table.empty:
            unavailable_warning = _unavailable_message(model_name)
            if choice != "all":
                raise JobError(
                    "MODEL_UNAVAILABLE",
                    f"The {model_name} model artifact could not be loaded in the current environment.",
                    warnings=model_warnings,
                )
            warnings.extend(model_warnings)
            warnings.append(unavailable_warning)
            continue
        tables.append(table)
        warnings.extend(model_warnings)

    if not tables:
        if choice != "all":
            raise JobError(
                "MODEL_UNAVAILABLE",
                f"The {choice} model artifact could not be loaded in the current environment.",
                warnings=warnings,
            )
        raise JobError(
            "PREDICTION_BACKEND_NOT_CONNECTED",
            "No available model artifacts produced predictions.",
            warnings=warnings,
        )

    selected = pd.concat(tables, ignore_index=True)
    predictions = []
    for row in selected.to_dict(orient="records"):
        predictions.append(
            {
                "model_name": str(row["model"]),
                "predicted_emission_nm": _nullable_number(
                    row.get("predicted_emission_nm")
                ),
                "predicted_quantum_yield": _nullable_number(
                    row.get("predicted_quantum_yield")
                ),
                "nearest_training_similarity": _nullable_number(
                    row.get("nearest_training_similarity")
                ),
                "nearest_training_smiles": row.get("nearest_training_smiles") or None,
                "warnings": [],
            }
        )
    assert canonical_molecule is not None and canonical_solvent is not None
    return predictions, warnings, canonical_molecule, canonical_solvent


def write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run_job(
    input_path: Path,
    output_path: Path,
    backend: PredictionBackend = fluorcast_prediction_backend,
) -> int:
    """Run a prediction job and always attempt to write its JSON result."""
    payload: dict[str, Any] = {}
    try:
        payload = read_input(input_path)
        validate_input(payload)
        predictions, warnings, canonical_molecule, canonical_solvent = backend(payload)
        result = {
            "status": "success",
            "job_id": payload["job_id"],
            "canonical_molecule_smiles": canonical_molecule,
            "canonical_solvent_smiles": canonical_solvent,
            "predictions": predictions,
            "warnings": warnings,
        }
        exit_code = 0
    except Exception as exc:  # The job contract requires an output for every failure.
        result = {
            "status": "failed",
            "job_id": payload.get("job_id"),
            "error_code": getattr(exc, "code", "PREDICTION_JOB_FAILED"),
            "error_message": str(exc),
            "traceback": traceback_module.format_exc(),
            "warnings": getattr(exc, "warnings", []),
        }
        exit_code = 1
    write_output(output_path, result)
    return exit_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_job(args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
