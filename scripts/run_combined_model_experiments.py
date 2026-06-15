"""Run systematic model comparisons for combined ChemFluor predictors."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from chemfluor.combined_prediction import (  # noqa: E402
    applicability_domain_payload,
    canonicalize_required,
    get_solvent_descriptor_row,
    load_or_infer_feature_metadata,
    load_solvent_descriptors,
    require_rdkit,
)
from chemfluor.data_standardization import TARGET_COLUMNS  # noqa: E402
import predict_combined_molecule as predictor  # noqa: E402
import train_combined_predictors as trainer  # noqa: E402


DEFAULT_OUT_ROOT = Path("models/experiments_fluodb")
DEFAULT_COMPARE_OUT = Path("outputs/model_experiments_fluodb")
DEFAULT_MODELS = "rf,extratrees,histgb,gbdt,mlp"
DEFAULT_TARGETS = "emission_nm,quantum_yield"
REGION_ORDER = ["UV", "blue", "green", "yellow/orange", "red/NIR"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and compare alternative combined ChemFluor model families."
    )
    parser.add_argument("--standardized-combined", required=True, type=Path)
    parser.add_argument(
        "--solvent-descriptors",
        default=trainer.DEFAULT_SOLVENT_DESCRIPTORS,
        type=Path,
    )
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT, type=Path)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--compare-out", default=DEFAULT_COMPARE_OUT, type=Path)
    parser.add_argument("--max-train-rows", default=None, type=int)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--n-jobs", default=-1, type=int)
    parser.add_argument("--benchmark-smiles", default=None)
    parser.add_argument("--benchmark-solvent-smiles", default=None)
    parser.add_argument("--known-emission-nm", default=None, type=float)
    parser.add_argument("--known-quantum-yield", default=None, type=float)
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    """Parse a comma-separated CLI list."""
    return [item.strip() for item in text.split(",") if item.strip()]


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON from a path."""
    return json.loads(path.read_text(encoding="utf-8"))


def run_training_for_model(
    model_type: str,
    args: argparse.Namespace,
    targets: list[str],
) -> Path:
    """Run the existing combined trainer for one model type."""
    model_dir = args.out_root / model_type
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_combined_predictors.py"),
        "--standardized-combined",
        str(args.standardized_combined),
        "--solvent-descriptors",
        str(args.solvent_descriptors),
        "--out-dir",
        str(model_dir),
        "--model",
        model_type,
        "--targets",
        ",".join(targets),
        "--random-state",
        str(args.random_state),
        "--n-jobs",
        str(args.n_jobs),
    ]
    if args.max_train_rows is not None:
        command.extend(["--max-train-rows", str(args.max_train_rows)])

    print(f"\nTraining {model_type} into {model_dir}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        print(f"WARNING: training failed for {model_type}; continuing.")
    return model_dir


def collect_model_metrics(model_dirs: dict[str, Path]) -> pd.DataFrame:
    """Collect metrics.json files into one comparison table."""
    rows: list[dict[str, Any]] = []
    for model_name, model_dir in model_dirs.items():
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"WARNING: metrics not found for {model_name}: {metrics_path}")
            continue
        metrics = load_json(metrics_path)
        metric_rows = metrics.values() if isinstance(metrics, dict) else metrics
        for metric in metric_rows:
            row = {
                "model": model_name,
                "target": metric.get("target"),
                "mae": metric.get("mae"),
                "rmse": metric.get("rmse"),
                "r2": metric.get("r2"),
                "train_rows": metric.get("train_rows"),
                "test_rows": metric.get("test_rows"),
            }
            rows.append(row)
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return pd.DataFrame(
            columns=["model", "target", "mae", "rmse", "r2", "train_rows", "test_rows"]
        )
    sort_columns = ["target", "mae"]
    if "emission_nm" in set(comparison["target"]):
        comparison["_target_rank"] = np.where(comparison["target"] == "emission_nm", 0, 1)
        sort_columns = ["_target_rank", "mae"]
    comparison = comparison.sort_values(sort_columns, kind="mergesort")
    return comparison.drop(columns=["_target_rank"], errors="ignore").reset_index(
        drop=True
    )


def wavelength_region(value: float) -> str:
    """Assign an emission wavelength to a named region."""
    if value < 400:
        return "UV"
    if value < 500:
        return "blue"
    if value < 550:
        return "green"
    if value < 600:
        return "yellow/orange"
    return "red/NIR"


def compute_error_by_region(predictions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Compute emission error summaries by wavelength region."""
    required = {"y_true", "y_pred"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table missing column(s): {sorted(missing)}")
    working = predictions.copy()
    working["residual"] = working["y_true"] - working["y_pred"]
    working["absolute_error"] = working["residual"].abs()
    working["wavelength_region"] = working["y_true"].map(wavelength_region)
    rows: list[dict[str, Any]] = []
    for region in REGION_ORDER:
        subset = working[working["wavelength_region"] == region]
        if subset.empty:
            continue
        rows.append(
            {
                "model": model_name,
                "wavelength_region": region,
                "rows": int(len(subset)),
                "mean_absolute_error": float(subset["absolute_error"].mean()),
                "median_absolute_error": float(subset["absolute_error"].median()),
                "rmse": float(np.sqrt(np.mean(np.square(subset["residual"])))),
                "mean_residual": float(subset["residual"].mean()),
            }
        )
    return pd.DataFrame(rows)


def collect_error_by_region(model_dirs: dict[str, Path]) -> pd.DataFrame:
    """Collect wavelength-region emission errors for all trained models."""
    tables: list[pd.DataFrame] = []
    for model_name, model_dir in model_dirs.items():
        prediction_path = model_dir / "predictions_emission_nm.csv"
        if not prediction_path.exists():
            print(
                f"WARNING: emission prediction CSV not found for {model_name}: "
                f"{prediction_path}"
            )
            continue
        tables.append(compute_error_by_region(pd.read_csv(prediction_path), model_name))
    if not tables:
        return pd.DataFrame(
            columns=[
                "model",
                "wavelength_region",
                "rows",
                "mean_absolute_error",
                "median_absolute_error",
                "rmse",
                "mean_residual",
            ]
        )
    return pd.concat(tables, ignore_index=True)


def write_markdown_comparison(comparison: pd.DataFrame, path: Path) -> None:
    """Write a compact markdown model-comparison table."""
    if comparison.empty:
        path.write_text("# Combined Model Comparison\n\nNo metrics were collected.\n", encoding="utf-8")
        return
    rounded = comparison.copy()
    for column in ["mae", "rmse", "r2"]:
        rounded[column] = rounded[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
        )
    headers = list(rounded.columns)
    rows = [[str(value) for value in row] for row in rounded.to_numpy()]
    table = "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )
    path.write_text(
        "# Combined Model Comparison\n\n"
        "Models are ranked by emission MAE when emission metrics are available.\n\n"
        + table
        + "\n",
        encoding="utf-8",
    )


def benchmark_prediction_for_model(
    model_name: str,
    model_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Predict the optional benchmark molecule with one trained model directory."""
    require_rdkit()
    canonical_smiles = canonicalize_required(args.benchmark_smiles, "molecule")
    canonical_solvent = canonicalize_required(args.benchmark_solvent_smiles, "solvent")
    models, warnings = predictor.load_available_models(model_dir, model_name)
    for warning in warnings:
        print(f"WARNING: {warning}")

    solvent_descriptors = load_solvent_descriptors(args.solvent_descriptors)
    metadata, metadata_warnings = load_or_infer_feature_metadata(
        model_dir=model_dir,
        solvent_descriptors=solvent_descriptors,
        model_feature_count=predictor.first_model_feature_count(models),
    )
    for warning in metadata_warnings:
        print(f"WARNING: {warning}")
    solvent_row = get_solvent_descriptor_row(
        solvent_descriptors,
        solvent_smiles=args.benchmark_solvent_smiles,
        canonical_solvent_smiles=canonical_solvent,
    )
    predictions = predictor.predict_available_targets(
        canonical_smiles=canonical_smiles,
        models=models,
        metadata=metadata,
        solvent_descriptor_row=solvent_row,
    )
    domain, domain_warnings = applicability_domain_payload(
        canonical_smiles=canonical_smiles,
        model_dir=model_dir,
        threshold=predictor.DEFAULT_APPLICABILITY_THRESHOLD,
        radius=int(metadata.get("fingerprint_radius", 2)),
        n_bits=int(metadata.get("fingerprint_n_bits", 2048)),
        disabled=False,
    )
    for warning in domain_warnings:
        print(f"WARNING: {warning}")

    predicted_emission = predictions.get("emission_nm")
    predicted_qy = predictions.get("quantum_yield")
    known_emission = args.known_emission_nm
    known_qy = args.known_quantum_yield
    return {
        "model": model_name,
        "predicted_emission_nm": predicted_emission,
        "known_emission_nm": known_emission,
        "emission_absolute_error": (
            None
            if predicted_emission is None or known_emission is None
            else abs(predicted_emission - known_emission)
        ),
        "predicted_quantum_yield": predicted_qy,
        "known_quantum_yield": known_qy,
        "quantum_yield_absolute_error": (
            None if predicted_qy is None or known_qy is None else abs(predicted_qy - known_qy)
        ),
        "nearest_training_similarity": domain.get("nearest_training_similarity"),
        "confidence_label": domain.get("confidence_label"),
        "outside_applicability_domain": domain.get("outside_applicability_domain"),
    }


def collect_benchmark_predictions(
    model_dirs: dict[str, Path], args: argparse.Namespace
) -> pd.DataFrame:
    """Run benchmark predictions when benchmark inputs are provided."""
    if not args.benchmark_smiles or not args.benchmark_solvent_smiles:
        return pd.DataFrame()
    rows = []
    for model_name, model_dir in model_dirs.items():
        try:
            rows.append(benchmark_prediction_for_model(model_name, model_dir, args))
        except (FileNotFoundError, ValueError, ImportError, KeyError) as exc:
            print(f"WARNING: benchmark failed for {model_name}: {exc}")
    return pd.DataFrame(rows)


def write_outputs(
    compare_out: Path,
    comparison: pd.DataFrame,
    region_comparison: pd.DataFrame,
    benchmark_comparison: pd.DataFrame,
) -> None:
    """Write comparison outputs."""
    compare_out.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(compare_out / "model_comparison.csv", index=False)
    write_markdown_comparison(comparison, compare_out / "model_comparison.md")
    region_comparison.to_csv(
        compare_out / "error_by_region_comparison.csv", index=False
    )
    if not benchmark_comparison.empty:
        benchmark_comparison.to_csv(
            compare_out / "benchmark_prediction_comparison.csv", index=False
        )


def main() -> int:
    """Run all requested model experiments."""
    args = parse_args()
    try:
        models = parse_csv_list(args.models)
        targets = trainer.parse_targets(args.targets)
        args.out_root.mkdir(parents=True, exist_ok=True)
        model_dirs = {
            model_name: run_training_for_model(model_name, args, targets)
            for model_name in models
        }
        comparison = collect_model_metrics(model_dirs)
        region_comparison = collect_error_by_region(model_dirs)
        benchmark_comparison = collect_benchmark_predictions(model_dirs, args)
        write_outputs(
            compare_out=args.compare_out,
            comparison=comparison,
            region_comparison=region_comparison,
            benchmark_comparison=benchmark_comparison,
        )
        print(f"Saved model comparison to: {args.compare_out / 'model_comparison.csv'}")
        print(
            "Saved region comparison to: "
            f"{args.compare_out / 'error_by_region_comparison.csv'}"
        )
        if not benchmark_comparison.empty:
            print(
                "Saved benchmark comparison to: "
                f"{args.compare_out / 'benchmark_prediction_comparison.csv'}"
            )
        return 0
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
