from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config
from src.applicability import (
    canonicalize_input_molecule,
    confidence_assessment,
    nearest_training_molecules,
    scaffold_novelty,
    solvent_novelty,
)
from src.features import build_feature_matrix_inference
from src.utils import ensure_output_dirs


PREDICTIONS_DIR = config.OUTPUT_DIR / "predictions"


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def load_prediction_assets() -> dict | None:
    required = {
        "wavelength_model": config.MODEL_DIR / "best_wavelength_lightgbm.pkl",
        "plqy_model": config.MODEL_DIR / "best_plqy_lightgbm.pkl",
        "plqy_classifier": config.MODEL_DIR / "best_plqy_classifier.pkl",
        "feature_artifacts": config.MODEL_DIR / "feature_artifacts.pkl",
        "metadata": config.MODEL_DIR / "inference_metadata.pkl",
    }
    missing = [path for path in required.values() if not path.exists()]
    if missing:
        print("Missing trained model or metadata. Please run python -m src.train first.")
        for path in missing:
            print(f"- Missing: {path}")
        return None

    assets = {name: load_pickle(path) for name, path in required.items()}
    optional_seed_models = {
        "wavelength_seed_models": config.MODEL_DIR / "wavelength_seed_models.pkl",
        "plqy_seed_models": config.MODEL_DIR / "plqy_seed_models.pkl",
    }
    for name, path in optional_seed_models.items():
        assets[name] = load_pickle(path) if path.exists() else None
    return assets


def bright_probability(classifier, X: pd.DataFrame) -> float | None:
    if not hasattr(classifier, "predict_proba"):
        return None
    proba = classifier.predict_proba(X)[0]
    classes = list(getattr(classifier, "classes_", []))
    if 1 in classes:
        return float(proba[classes.index(1)])
    if len(proba) == 2:
        return float(proba[1])
    return None


def seed_uncertainty(models: list | None, X: pd.DataFrame) -> dict | None:
    if not models:
        return None
    preds = np.asarray([model.predict(X)[0] for model in models], dtype=float)
    return {"mean_prediction": float(preds.mean()), "prediction_std": float(preds.std())}


def uncertainty_caution(wave_uncertainty: dict | None, plqy_uncertainty: dict | None) -> str | None:
    cautions = []
    if wave_uncertainty and wave_uncertainty["prediction_std"] >= 50.0:
        cautions.append(f"wavelength ensemble uncertainty is high ({wave_uncertainty['prediction_std']:.1f} nm)")
    if plqy_uncertainty and plqy_uncertainty["prediction_std"] >= 0.20:
        cautions.append(f"PLQY ensemble uncertainty is high ({plqy_uncertainty['prediction_std']:.2f})")
    if not cautions:
        return None
    return "Extra caution: " + "; ".join(cautions) + "."


def predict_one(smiles: str, solvent: str, name: str | None, assets: dict) -> dict:
    solvent = str(solvent).strip()
    identity = canonicalize_input_molecule(smiles)
    input_df = pd.DataFrame(
        [
            {
                config.SMILES_COL: str(smiles).strip(),
                "canonical_smiles": identity.canonical_smiles,
                config.SOLVENT_COL: solvent,
            }
        ]
    )
    X, feature_status = build_feature_matrix_inference(input_df, assets["feature_artifacts"])

    wavelength_pred = float(assets["wavelength_model"].predict(X)[0])
    plqy_pred = float(np.clip(assets["plqy_model"].predict(X)[0], 0.0, 1.0))
    bright_pred = int(assets["plqy_classifier"].predict(X)[0])
    bright_prob = bright_probability(assets["plqy_classifier"], X)
    wavelength_uncertainty = seed_uncertainty(assets.get("wavelength_seed_models"), X)
    plqy_uncertainty = seed_uncertainty(assets.get("plqy_seed_models"), X)

    metadata = assets["metadata"]
    nearest = nearest_training_molecules(identity.mol, metadata)
    scaffold_info = scaffold_novelty(identity.scaffold, metadata)
    solvent_info = solvent_novelty(solvent, metadata)
    descriptors_missing = bool(feature_status.get("solvent_descriptors_imputed"))
    confidence = confidence_assessment(
        nearest["max_similarity"],
        scaffold_info["scaffold_seen"],
        solvent_info["solvent_seen"],
        descriptors_missing=descriptors_missing,
        uncertainty_caution=uncertainty_caution(wavelength_uncertainty, plqy_uncertainty),
    )

    return {
        "input": {
            "name": name,
            "smiles": str(smiles).strip(),
            "canonical_smiles": identity.canonical_smiles,
            "solvent": solvent,
        },
        "applicability_domain": {
            **confidence,
            **nearest,
            **scaffold_info,
            **solvent_info,
            "solvent_descriptors_imputed": descriptors_missing,
            "missing_descriptor_solvents": feature_status.get("missing_descriptor_solvents", []),
        },
        "predictions": {
            "emission_wavelength_nm": wavelength_pred,
            "plqy_regression": plqy_pred,
            "bright_dim_classification": "Bright" if bright_pred == 1 else "Dim",
            "bright_probability": bright_prob,
            "bright_threshold": metadata.get("plqy_bright_threshold", config.BRIGHT_THRESHOLD),
            "wavelength_seed_ensemble": wavelength_uncertainty,
            "plqy_seed_ensemble": plqy_uncertainty,
        },
        "notes": [
            "Prediction should be treated as low-confidence if confidence level is Low.",
            "For unfamiliar scaffolds, scaffold-split error is a better guide than random-split error.",
        ],
    }


def print_report(result: dict) -> None:
    inp = result["input"]
    ad = result["applicability_domain"]
    pred = result["predictions"]
    print("\n================ ChemFluor Prediction Report ================\n")
    print("Input:")
    print(f"- Name: {inp.get('name') or 'N/A'}")
    print(f"- SMILES: {inp['smiles']}")
    print(f"- Canonical SMILES: {inp['canonical_smiles']}")
    print(f"- Solvent: {inp['solvent']}")

    print("\nApplicability domain:")
    print(f"- Confidence: {ad['confidence_level']}")
    print(f"- Domain score: {ad['domain_score']:.2f}")
    print(f"- Max Tanimoto similarity: {ad['max_similarity']:.2f}")
    print(f"- Top-5 average similarity: {ad['top5_average_similarity']:.2f}")
    print(f"- Scaffold seen in training: {'Yes' if ad['scaffold_seen'] else 'No'}")
    print(f"- Solvent seen in training: {'Yes' if ad['solvent_seen'] else 'No'}")
    if ad.get("solvent_descriptors_imputed"):
        print("- Solvent descriptors: missing values median-imputed")
    print(f"- Warning: {ad['warning']}")

    print("\nNearest training molecules:")
    for i, neighbor in enumerate(ad["nearest_neighbors"], start=1):
        print(
            f"{i}. similarity={neighbor['similarity']:.2f} | "
            f"emission={neighbor['emission_nm']:.1f} nm | "
            f"PLQY={neighbor['plqy']:.3f} | "
            f"solvent={neighbor['solvent']} | "
            f"smiles={neighbor['canonical_smiles']}"
        )

    print("\nPredictions:")
    print(f"- Emission wavelength: {pred['emission_wavelength_nm']:.1f} nm")
    if pred.get("wavelength_seed_ensemble"):
        unc = pred["wavelength_seed_ensemble"]
        print(f"- Wavelength seed ensemble: mean={unc['mean_prediction']:.1f} nm, std={unc['prediction_std']:.1f} nm")
    print(f"- PLQY regression: {pred['plqy_regression']:.3f}")
    if pred.get("plqy_seed_ensemble"):
        unc = pred["plqy_seed_ensemble"]
        print(f"- PLQY seed ensemble: mean={unc['mean_prediction']:.3f}, std={unc['prediction_std']:.3f}")
    print(f"- Bright/dim classification: {pred['bright_dim_classification']}")
    if pred["bright_probability"] is not None:
        print(f"- Bright probability: {pred['bright_probability']:.2f}")
    print(f"- Bright threshold: PLQY > {pred['bright_threshold']}")

    print("\nNotes:")
    for note in result["notes"]:
        print(f"- {note}")


def save_single_json(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    print(f"\nSaved prediction JSON: {path}")


def flatten_for_csv(result: dict) -> dict:
    if "error" in result:
        return result
    inp = result["input"]
    ad = result["applicability_domain"]
    pred = result["predictions"]
    return {
        "name": inp.get("name"),
        "SMILES": inp["smiles"],
        "canonical_smiles": inp["canonical_smiles"],
        "solvent": inp["solvent"],
        "confidence_level": ad["confidence_level"],
        "domain_score": ad["domain_score"],
        "max_tanimoto_similarity": ad["max_similarity"],
        "top5_average_similarity": ad["top5_average_similarity"],
        "scaffold_seen": ad["scaffold_seen"],
        "solvent_seen": ad["solvent_seen"],
        "warning": ad["warning"],
        "emission_wavelength_nm": pred["emission_wavelength_nm"],
        "plqy_regression": pred["plqy_regression"],
        "bright_dim_classification": pred["bright_dim_classification"],
        "bright_probability": pred["bright_probability"],
        "bright_threshold": pred["bright_threshold"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ChemFluor properties for new molecules.")
    parser.add_argument("--smiles", help="Input molecule SMILES for single prediction.")
    parser.add_argument("--solvent", help="Solvent name for single prediction.")
    parser.add_argument("--name", help="Optional candidate name.")
    parser.add_argument("--output", help="Optional JSON output path for single prediction.")
    parser.add_argument("--csv", help="Batch input CSV with SMILES and solvent columns, and optional name column.")
    parser.add_argument("--save-csv", help="Optional output CSV path for batch predictions.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    assets = load_prediction_assets()
    if assets is None:
        return 1

    if args.csv:
        input_path = Path(args.csv)
        if not input_path.exists():
            print(f"Batch input CSV not found: {input_path}")
            return 1
        batch = pd.read_csv(input_path)
        missing_cols = [col for col in [config.SMILES_COL, config.SOLVENT_COL] if col not in batch.columns]
        if missing_cols:
            print(f"Batch CSV is missing required columns: {missing_cols}")
            return 1
        results = []
        for _, row in batch.iterrows():
            try:
                result = predict_one(row[config.SMILES_COL], row[config.SOLVENT_COL], row.get("name"), assets)
                print_report(result)
                results.append(flatten_for_csv(result))
            except Exception as exc:
                results.append(
                    {
                        "name": row.get("name"),
                        "SMILES": row.get(config.SMILES_COL),
                        "solvent": row.get(config.SOLVENT_COL),
                        "error": str(exc),
                    }
                )
                print(f"\nPrediction failed for {row.get(config.SMILES_COL)!r}: {exc}")
        save_path = Path(args.save_csv) if args.save_csv else PREDICTIONS_DIR / "batch_predictions.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(save_path, index=False)
        print(f"\nSaved batch predictions CSV: {save_path}")
        return 0

    if not args.smiles or not args.solvent:
        print("For single prediction, provide both --smiles and --solvent, or use --csv for batch prediction.")
        return 1

    try:
        result = predict_one(args.smiles, args.solvent, args.name, assets)
    except Exception as exc:
        print(f"Prediction failed: {exc}")
        return 1
    print_report(result)
    if args.output:
        save_single_json(result, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
