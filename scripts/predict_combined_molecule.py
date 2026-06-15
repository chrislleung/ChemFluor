"""Predict photophysical properties for one molecule with combined models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.combined_prediction import (  # noqa: E402
    applicability_domain_payload,
    build_single_feature_matrix,
    canonicalize_required,
    exact_reference_matches_payload,
    get_solvent_descriptor_row,
    load_or_infer_feature_metadata,
    load_solvent_descriptors,
    require_rdkit,
)
from chemfluor.data_standardization import TARGET_COLUMNS  # noqa: E402


DEFAULT_MODEL_DIR = Path("models/chemfluor_combined")
DEFAULT_SOLVENT_DESCRIPTORS = Path("data/solvent_descriptors_expanded_deep4chem.csv")
DEFAULT_OUT = Path("outputs/predictions/new_molecule_prediction.json")
DEFAULT_APPLICABILITY_THRESHOLD = 0.50
KNOWN_VALUE_ARGUMENTS = {
    "absorption_nm": "known_absorption_nm",
    "emission_nm": "known_emission_nm",
    "lifetime_ns": "known_lifetime_ns",
    "quantum_yield": "known_quantum_yield",
    "log_extinction": "known_log_extinction",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Predict one molecule using trained ChemFluor + Deep4Chem combined models."
        )
    )
    parser.add_argument("--smiles", required=True, help="Chromophore SMILES.")
    parser.add_argument(
        "--solvent-smiles",
        required=True,
        help="Solvent SMILES, for example O for water or CCO for ethanol.",
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, type=Path)
    parser.add_argument(
        "--solvent-descriptors", default=DEFAULT_SOLVENT_DESCRIPTORS, type=Path
    )
    parser.add_argument("--out", default=DEFAULT_OUT, type=Path)
    parser.add_argument("--name", default=None, help="Optional candidate name.")
    parser.add_argument("--out-csv", default=None, type=Path)
    parser.add_argument(
        "--model-type",
        default="rf",
        help="Model filename suffix to load, for example rf, histgb, mlp, or lightgbm.",
    )
    parser.add_argument(
        "--applicability-threshold",
        default=DEFAULT_APPLICABILITY_THRESHOLD,
        type=float,
        help="Tanimoto threshold below which a candidate is flagged.",
    )
    parser.add_argument(
        "--no-applicability-domain",
        action="store_true",
        help="Disable Morgan fingerprint Tanimoto applicability-domain scoring.",
    )
    parser.add_argument("--known-absorption-nm", type=float, default=None)
    parser.add_argument("--known-emission-nm", type=float, default=None)
    parser.add_argument("--known-lifetime-ns", type=float, default=None)
    parser.add_argument("--known-quantum-yield", type=float, default=None)
    parser.add_argument("--known-log-extinction", type=float, default=None)
    parser.add_argument(
        "--check-exact-reference-match",
        action="store_true",
        help=(
            "Report exact canonical chromophore matches from "
            "combined_modeling_rows_after_feature_merge.csv."
        ),
    )
    return parser.parse_args()


def validate_model_dir(model_dir: Path) -> None:
    """Validate that the model directory exists."""
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not model_dir.is_dir():
        raise ValueError(f"Model path is not a directory: {model_dir}")


def load_available_models(
    model_dir: Path, model_type: str = "rf"
) -> tuple[dict[str, Any], list[str]]:
    """Load every target model present in the combined model directory."""
    models: dict[str, Any] = {}
    warnings: list[str] = []
    for target in TARGET_COLUMNS:
        path = model_dir / f"{target}_{model_type}.joblib"
        if path.exists():
            models[target] = joblib.load(path)
        else:
            warnings.append(f"Model not found for {target}; skipping: {path}")
    if not models:
        raise FileNotFoundError(f"No target model joblib files found in {model_dir}")
    return models, warnings


def first_model_feature_count(models: dict[str, Any]) -> int | None:
    """Return n_features_in_ from the first model that exposes it."""
    for model in models.values():
        feature_count = getattr(model, "n_features_in_", None)
        if feature_count is not None:
            return int(feature_count)
    return None


def predict_available_targets(
    canonical_smiles: str,
    models: dict[str, Any],
    metadata: dict[str, Any],
    solvent_descriptor_row: pd.Series | None,
) -> dict[str, float]:
    """Predict all loaded targets with target-specific imputation medians."""
    radius = int(metadata.get("fingerprint_radius", 2))
    n_bits = int(metadata.get("fingerprint_n_bits", 2048))
    descriptor_columns = list(metadata.get("solvent_descriptor_columns_used", []))
    medians_by_target = metadata.get("median_values_used_for_imputation", {})

    predictions: dict[str, float] = {}
    for target, model in models.items():
        medians = medians_by_target.get(target, {})
        features = build_single_feature_matrix(
            canonical_smiles=canonical_smiles,
            solvent_descriptor_row=solvent_descriptor_row,
            descriptor_columns=descriptor_columns,
            medians=medians,
            radius=radius,
            n_bits=n_bits,
        )
        predicted = model.predict(features)[0]
        predictions[target] = float(predicted)
    return predictions


def collect_known_values(args: argparse.Namespace) -> dict[str, float]:
    """Collect optional known target values from parsed CLI arguments."""
    known_values: dict[str, float] = {}
    for target, argument_name in KNOWN_VALUE_ARGUMENTS.items():
        value = getattr(args, argument_name)
        if value is not None:
            known_values[target] = float(value)
    return known_values


def compute_prediction_errors(
    predictions: dict[str, float],
    known_values: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Compute residual and absolute error for targets with prediction and known value."""
    errors: dict[str, dict[str, float]] = {}
    for target, known in known_values.items():
        if target not in predictions:
            continue
        predicted = float(predictions[target])
        residual = predicted - float(known)
        errors[target] = {
            "predicted": predicted,
            "known": float(known),
            "residual": residual,
            "absolute_error": abs(residual),
        }
    return errors


def flatten_for_csv(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten the JSON payload into a one-row CSV-friendly table."""
    row: dict[str, Any] = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "predictions",
            "applicability_domain",
            "known_values",
            "errors",
            "exact_reference_matches",
        }
    }
    for target, value in payload.get("predictions", {}).items():
        row[target] = value
    for key, value in payload.get("applicability_domain", {}).items():
        row[key] = value
    for target, value in payload.get("known_values", {}).items():
        row[f"known_{target}"] = value
    for target, values in payload.get("errors", {}).items():
        row[f"error_{target}"] = values["absolute_error"]
        row[f"residual_{target}"] = values["residual"]
    if "exact_reference_matches" in payload:
        row["exact_reference_match_count"] = payload["exact_reference_matches"]["count"]
    return pd.DataFrame([row])


def print_summary(payload: dict[str, Any], warnings: list[str]) -> None:
    """Print a readable terminal summary."""
    print("\nCombined-model molecule prediction")
    if payload.get("name"):
        print(f"Name: {payload['name']}")
    print(f"Canonical SMILES: {payload['canonical_smiles']}")
    print(f"Canonical solvent SMILES: {payload['canonical_solvent_smiles']}")
    print(f"Model directory: {payload['model_dir']}")
    print("\nPredictions:")
    for target, value in payload["predictions"].items():
        print(f"  {target}: {value:.6g}")
    if not payload["predictions"]:
        print("  (none)")
    if payload.get("applicability_domain"):
        domain = payload["applicability_domain"]
        print("\nApplicability domain:")
        similarity = domain.get("nearest_training_similarity")
        if similarity is not None:
            print(f"  nearest_training_similarity: {similarity:.4f}")
        print(f"  confidence_label: {domain.get('confidence_label', 'unknown')}")
        print(
            "  interpretation: "
            f"{domain.get('confidence_interpretation', 'Applicability-domain similarity could not be computed.')}"
        )
        if "nearest_training_smiles" in domain:
            print(f"  nearest_training_smiles: {domain['nearest_training_smiles']}")
        if "outside_applicability_domain" in domain:
            print(
                "  outside_applicability_domain: "
                f"{domain['outside_applicability_domain']}"
            )
        if "threshold" in domain:
            print(f"  threshold: {domain['threshold']:.2f}")
    if payload.get("known_values"):
        print("\nKnown-value evaluation:")
        for target, value in payload["known_values"].items():
            print(f"  known_{target}: {value:.6g}")
        for target, values in payload.get("errors", {}).items():
            print(
                f"  {target} residual: {values['residual']:.6g}; "
                f"absolute_error: {values['absolute_error']:.6g}"
            )
    if payload.get("exact_reference_matches") is not None:
        print(
            "\nExact reference matches found: "
            f"{payload['exact_reference_matches']['count']}"
        )
    domain = payload.get("applicability_domain", {})
    similarity = domain.get("nearest_training_similarity")
    threshold = domain.get("threshold", DEFAULT_APPLICABILITY_THRESHOLD)
    if similarity is not None and similarity < threshold:
        print(
            "WARNING: This prediction is low-confidence for single-molecule use "
            f"because the nearest training similarity is below {threshold:.2f}."
        )
    for warning in warnings:
        print(f"WARNING: {warning}")


def main() -> int:
    """Run one-molecule combined-model prediction."""
    args = parse_args()
    warnings: list[str] = []
    try:
        require_rdkit()
        validate_model_dir(args.model_dir)
        canonical_smiles = canonicalize_required(args.smiles, "molecule")
        canonical_solvent_smiles = canonicalize_required(
            args.solvent_smiles, "solvent"
        )

        models, model_warnings = load_available_models(args.model_dir, args.model_type)
        warnings.extend(model_warnings)

        solvent_descriptors = load_solvent_descriptors(args.solvent_descriptors)
        metadata, metadata_warnings = load_or_infer_feature_metadata(
            model_dir=args.model_dir,
            solvent_descriptors=solvent_descriptors,
            model_feature_count=first_model_feature_count(models),
        )
        warnings.extend(metadata_warnings)

        solvent_descriptor_row = get_solvent_descriptor_row(
            descriptors=solvent_descriptors,
            solvent_smiles=args.solvent_smiles,
            canonical_solvent_smiles=canonical_solvent_smiles,
        )
        if solvent_descriptor_row is None:
            warnings.append(
                "Solvent descriptors not found; using training medians for "
                "solvent descriptor features."
            )

        predictions = predict_available_targets(
            canonical_smiles=canonical_smiles,
            models=models,
            metadata=metadata,
            solvent_descriptor_row=solvent_descriptor_row,
        )
        known_values = collect_known_values(args)
        errors = compute_prediction_errors(predictions, known_values)
        applicability_domain, domain_warnings = applicability_domain_payload(
            canonical_smiles=canonical_smiles,
            model_dir=args.model_dir,
            threshold=args.applicability_threshold,
            radius=int(metadata.get("fingerprint_radius", 2)),
            n_bits=int(metadata.get("fingerprint_n_bits", 2048)),
            disabled=args.no_applicability_domain,
        )
        warnings.extend(domain_warnings)
        exact_reference_matches = None
        if args.check_exact_reference_match:
            exact_reference_matches = exact_reference_matches_payload(
                canonical_smiles=canonical_smiles,
                model_dir=args.model_dir,
            )

        payload = {
            "name": args.name,
            "input_smiles": args.smiles,
            "canonical_smiles": canonical_smiles,
            "solvent_smiles": args.solvent_smiles,
            "canonical_solvent_smiles": canonical_solvent_smiles,
            "model_dir": str(args.model_dir),
            "model_type": args.model_type,
            "predictions": predictions,
            "applicability_domain": applicability_domain,
        }
        if known_values:
            payload["known_values"] = known_values
            payload["errors"] = errors
        if exact_reference_matches is not None:
            payload["exact_reference_matches"] = exact_reference_matches

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if args.out_csv is not None:
            args.out_csv.parent.mkdir(parents=True, exist_ok=True)
            flatten_for_csv(payload).to_csv(args.out_csv, index=False)

        print_summary(payload, warnings)
        print(f"\nSaved JSON prediction to: {args.out}")
        if args.out_csv is not None:
            print(f"Saved CSV prediction to: {args.out_csv}")
        return 0
    except (
        FileNotFoundError,
        ImportError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
