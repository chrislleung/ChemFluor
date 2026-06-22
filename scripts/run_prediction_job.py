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
MODEL_CHOICES = {"all", "rf", "extratrees", "histgb", "graph_model_later"}
PredictionBackend = Callable[
    [dict[str, Any]], tuple[list[dict[str, Any]], list[str], str, str]
]


class JobError(Exception):
    """An expected job failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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


def _model_selected(model_name: str, choice: str) -> bool:
    if choice == "all":
        return True
    normalized = model_name.casefold().replace("_", "").replace("-", "")
    aliases = {
        "rf": ("rf", "randomforest"),
        "extratrees": ("extratrees", "extra trees"),
        "histgb": ("histgb", "histgradientboosting"),
    }
    return any(alias in normalized for alias in aliases.get(choice, ()))


def fluorcast_prediction_backend(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], str, str]:
    """Adapt the existing all-model predictor to the JSON job contract."""
    if payload["model_choice"] == "graph_model_later":
        raise JobError(
            "PREDICTION_BACKEND_NOT_CONNECTED",
            "graph_model_later is reserved for a future graph prediction backend.",
        )

    args = SimpleNamespace(
        smiles=payload["molecule_smiles"],
        solvent=None,
        solvent_smiles=payload["solvent_smiles"],
        solvent_descriptors=PROJECT_ROOT / predict_all_models.DEFAULT_SOLVENT_DESCRIPTORS,
        standardized_combined=PROJECT_ROOT / predict_all_models.DEFAULT_STANDARDIZED_COMBINED,
        tree_model_dir=PROJECT_ROOT / predict_all_models.DEFAULT_TREE_MODEL_DIR,
        neural_model_dir=PROJECT_ROOT / predict_all_models.DEFAULT_NEURAL_MODEL_DIR,
        graph_model_dirs=[PROJECT_ROOT / predict_all_models.DEFAULT_GRAPH_MODEL_DIR],
        known_emission_nm=None,
        known_quantum_yield=None,
        applicability_threshold=predict_all_models.DEFAULT_APPLICABILITY_THRESHOLD,
    )
    table, warnings, canonical_molecule, canonical_solvent, _ = (
        predict_all_models.collect_predictions(args)
    )
    selected = table.loc[
        table["model"].astype(str).map(
            lambda name: _model_selected(name, str(payload["model_choice"]))
        )
    ]
    if selected.empty:
        raise JobError(
            "PREDICTION_BACKEND_NOT_CONNECTED",
            f"No available model artifacts matched model_choice={payload['model_choice']!r}.",
        )

    predictions = []
    for row in selected.to_dict(orient="records"):
        predictions.append(
            {
                "model_name": str(row["model"]),
                "predicted_absorption_nm": None,
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
    return predictions, warnings, canonical_molecule, str(canonical_solvent)


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
            "warnings": [],
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
