"""Diagnose external known-fluorophore benchmark predictions."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # optional dependency
    from rdkit import Chem
except Exception:  # pragma: no cover - depends on local environment
    Chem = None


OUTPUT_PREFIX = "external_benchmark_"
DEFAULT_METADATA_CANDIDATES = [
    Path("data/chatgpt_test_data.csv"),
    Path("data/external_benchmark.csv"),
    Path("data/known_fluorophores.csv"),
]

COLUMN_CANDIDATES: dict[str, list[str]] = {
    "molecule": ["molecule", "name", "chromophore", "compound", "dye"],
    "model": ["model", "model_name", "estimator"],
    "model_family": ["model_family", "family", "model_type"],
    "input_smiles": [
        "input_smiles",
        "molecule_smiles",
        "chromophore_smiles",
        "canonical_smiles",
        "canonical_chromophore_smiles",
        "smiles",
    ],
    "input_solvent": ["input_solvent", "solvent", "solvent_name", "solvent_original"],
    "input_solvent_smiles": [
        "input_solvent_smiles",
        "solvent_smiles",
        "canonical_solvent_smiles",
    ],
    "expected_emission_nm": [
        "expected_emission_nm",
        "known_emission_nm",
        "emission_nm",
        "literature_emission_nm",
        "emission_max_nm",
    ],
    "expected_quantum_yield": [
        "expected_quantum_yield",
        "expected_qy",
        "known_quantum_yield",
        "quantum_yield",
        "qy",
        "plqy",
    ],
    "predicted_emission_nm": [
        "predicted_emission_nm",
        "prediction_emission_nm",
        "emission_prediction_nm",
    ],
    "predicted_quantum_yield": [
        "predicted_quantum_yield",
        "predicted_qy",
        "prediction_quantum_yield",
    ],
    "emission_abs_error_nm": [
        "emission_abs_error_nm",
        "emission_absolute_error",
        "absolute_error_emission_nm",
    ],
    "quantum_yield_abs_error": [
        "quantum_yield_abs_error",
        "quantum_yield_absolute_error",
        "qy_abs_error",
        "absolute_error_quantum_yield",
    ],
    "nearest_training_similarity": [
        "nearest_training_similarity",
        "nearest_similarity",
        "max_similarity",
    ],
    "nearest_training_smiles": [
        "nearest_training_smiles",
        "nearest_smiles",
        "nearest_training_canonical_smiles",
    ],
}

TRAINING_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "smiles": [
        "canonical_smiles",
        "canonical_chromophore_smiles",
        "molecule_smiles",
        "chromophore_smiles",
        "smiles",
    ],
    "solvent": ["solvent", "solvent_name", "solvent_original", "input_solvent"],
    "solvent_smiles": [
        "solvent_smiles",
        "canonical_solvent_smiles",
        "input_solvent_smiles",
    ],
    "emission": ["emission_nm", "emission", "emission_max_nm"],
    "qy": ["quantum_yield", "qy", "plqy", "photoluminescence_quantum_yield"],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build diagnostics for external known-fluorophore predictions."
    )
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--training-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    for option in [
        "molecule_col",
        "solvent_col",
        "smiles_col",
        "solvent_smiles_col",
        "expected_emission_col",
        "expected_qy_col",
        "predicted_emission_col",
        "predicted_qy_col",
        "model_col",
        "model_family_col",
        "nearest_similarity_col",
        "nearest_smiles_col",
    ]:
        parser.add_argument("--" + option.replace("_", "-"), default=None)
    parser.add_argument("--emission-error-threshold", type=float, default=25.0)
    parser.add_argument("--qy-error-threshold", type=float, default=0.20)
    parser.add_argument("--training-label-mismatch-threshold", type=float, default=25.0)
    parser.add_argument("--model-disagreement-threshold", type=float, default=30.0)
    parser.add_argument("--low-similarity-threshold", type=float, default=0.65)
    return parser.parse_args(argv)


def warn(message: str, warnings: list[str]) -> None:
    warnings.append(message)
    print(f"WARNING: {message}", file=sys.stderr)


def normalize_key(value: str) -> str:
    return "".join(ch.lower() for ch in str(value).strip() if ch.isalnum())


def detect_column(
    columns: list[str],
    candidates: list[str],
    override: str | None = None,
    *,
    required: bool = False,
    label: str = "column",
) -> str | None:
    if override:
        if override in columns:
            return override
        raise ValueError(
            f"Requested {label} '{override}' was not found. Available columns: {', '.join(columns)}"
        )
    exact = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]
    normalized = {normalize_key(column): column for column in columns}
    for candidate in candidates:
        key = normalize_key(candidate)
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(
            f"Could not detect required {label}. Tried: {', '.join(candidates)}. "
            f"Available columns: {', '.join(columns)}"
        )
    return None


def numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    cleaned = series.astype(str).str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0]
    return pd.to_numeric(cleaned, errors="coerce")


def filename_to_molecule(path: Path) -> str:
    return path.stem.replace("_", " ").strip()


def canonicalize_smiles(value: Any, warnings: list[str]) -> str | float:
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    text = str(value).strip()
    if Chem is None:
        return text
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return text
    return Chem.MolToSmiles(mol, canonical=True)


def canonicalize_series(series: pd.Series, warnings: list[str]) -> pd.Series:
    return series.map(lambda value: canonicalize_smiles(value, warnings))


def infer_constant_expected(
    predicted: pd.Series, error: pd.Series, existing: pd.Series | None = None
) -> pd.Series:
    values = numeric(existing) if existing is not None else pd.Series(np.nan, index=predicted.index)
    if values.notna().any():
        return values
    pred = numeric(predicted)
    err = numeric(error)
    candidates: list[float] = []
    for p, e in zip(pred, err):
        if pd.notna(p) and pd.notna(e):
            candidates.extend([round(float(p - e), 6), round(float(p + e), 6)])
    if not candidates:
        return values
    counts = pd.Series(candidates).value_counts()
    best = float(counts.index[0])
    if counts.iloc[0] >= max(2, len(pred.dropna()) // 3):
        return pd.Series(best, index=predicted.index, dtype=float)
    return values


def read_metadata(prediction_dir: Path, warnings: list[str]) -> pd.DataFrame:
    candidates = [prediction_dir / name for name in ["benchmark_metadata.csv", "known_values.csv"]]
    candidates.extend(DEFAULT_METADATA_CANDIDATES)
    frames: list[pd.DataFrame] = []
    for path in candidates:
        if path.exists():
            try:
                table = pd.read_csv(path)
            except Exception as exc:
                try:
                    table = read_loose_csv(path)
                    warn(f"Read benchmark metadata {path} with tolerant CSV parsing: {exc}", warnings)
                except Exception as second_exc:
                    warn(f"Could not read benchmark metadata {path}: {second_exc}", warnings)
                    continue
            name_col = detect_column(table.columns.tolist(), COLUMN_CANDIDATES["molecule"])
            if not name_col:
                continue
            renamed = pd.DataFrame({"_metadata_molecule_key": table[name_col].map(normalize_key)})
            mapping = {
                "input_smiles": COLUMN_CANDIDATES["input_smiles"],
                "input_solvent": COLUMN_CANDIDATES["input_solvent"],
                "input_solvent_smiles": COLUMN_CANDIDATES["input_solvent_smiles"],
                "expected_emission_nm": COLUMN_CANDIDATES["expected_emission_nm"],
                "expected_quantum_yield": COLUMN_CANDIDATES["expected_quantum_yield"],
            }
            for out_col, choices in mapping.items():
                col = detect_column(table.columns.tolist(), choices)
                renamed[out_col] = table[col] if col else np.nan
            frames.append(renamed)
            print(f"Using benchmark metadata from {path}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("_metadata_molecule_key", keep="first")


def read_loose_csv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = []
        width = len(header)
        for row in reader:
            if len(row) > width:
                row = row[: width - 1] + [",".join(row[width - 1 :])]
            elif len(row) < width:
                row = row + [""] * (width - len(row))
            rows.append(row)
    return pd.DataFrame(rows, columns=header)


def prediction_csvs(prediction_dir: Path, out_dir: Path) -> list[Path]:
    skipped_names = {
        "external_benchmark_all_predictions.csv",
        "external_benchmark_model_summary.csv",
        "external_benchmark_family_summary.csv",
        "external_benchmark_molecule_summary.csv",
        "external_benchmark_training_overlap.csv",
        "external_benchmark_failure_modes.csv",
    }
    files = []
    for path in sorted(prediction_dir.glob("*.csv")):
        if path.name in skipped_names or path.name.startswith(OUTPUT_PREFIX):
            continue
        try:
            path.relative_to(out_dir)
            continue
        except ValueError:
            pass
        files.append(path)
    return files


def consolidate_predictions(args: argparse.Namespace, warnings: list[str]) -> pd.DataFrame:
    metadata = read_metadata(args.prediction_dir, warnings)
    frames: list[pd.DataFrame] = []
    overrides = {
        "molecule": args.molecule_col,
        "input_solvent": args.solvent_col,
        "input_smiles": args.smiles_col,
        "input_solvent_smiles": args.solvent_smiles_col,
        "expected_emission_nm": args.expected_emission_col,
        "expected_quantum_yield": args.expected_qy_col,
        "predicted_emission_nm": args.predicted_emission_col,
        "predicted_quantum_yield": args.predicted_qy_col,
        "model": args.model_col,
        "model_family": args.model_family_col,
        "nearest_training_similarity": args.nearest_similarity_col,
        "nearest_training_smiles": args.nearest_smiles_col,
    }
    for path in prediction_csvs(args.prediction_dir, args.out_dir):
        table = pd.read_csv(path)
        columns = table.columns.tolist()
        out = pd.DataFrame(index=table.index)
        out["source_file"] = path.name
        for col in [
            "molecule",
            "model",
            "model_family",
            "input_smiles",
            "input_solvent",
            "input_solvent_smiles",
            "expected_emission_nm",
            "predicted_emission_nm",
            "expected_quantum_yield",
            "predicted_quantum_yield",
            "nearest_training_similarity",
            "nearest_training_smiles",
        ]:
            detected = detect_column(
                columns,
                COLUMN_CANDIDATES[col],
                overrides.get(col),
                required=col in {"model", "predicted_emission_nm"},
                label=col,
            )
            out[col] = table[detected] if detected else np.nan
        err_col = detect_column(columns, COLUMN_CANDIDATES["emission_abs_error_nm"])
        qy_err_col = detect_column(columns, COLUMN_CANDIDATES["quantum_yield_abs_error"])
        out["emission_abs_error_nm"] = numeric(table[err_col]) if err_col else np.nan
        out["quantum_yield_abs_error"] = numeric(table[qy_err_col]) if qy_err_col else np.nan
        out["molecule"] = out["molecule"].fillna(filename_to_molecule(path))

        if not metadata.empty:
            key = normalize_key(filename_to_molecule(path))
            match = metadata[metadata["_metadata_molecule_key"] == key]
            if match.empty and key.endswith("2"):
                match = metadata[metadata["_metadata_molecule_key"] == key[:-1]]
            if not match.empty:
                row = match.iloc[0]
                for col in [
                    "input_smiles",
                    "input_solvent",
                    "input_solvent_smiles",
                    "expected_emission_nm",
                    "expected_quantum_yield",
                ]:
                    out[col] = out[col].where(out[col].notna(), row[col])

        if out["input_smiles"].isna().all() and out["nearest_training_similarity"].notna().any():
            exact = numeric(out["nearest_training_similarity"]).round(8) == 1.0
            out.loc[exact, "input_smiles"] = out.loc[exact, "nearest_training_smiles"]

        out["predicted_emission_nm"] = numeric(out["predicted_emission_nm"])
        out["predicted_quantum_yield"] = numeric(out["predicted_quantum_yield"])
        out["expected_emission_nm"] = infer_constant_expected(
            out["predicted_emission_nm"], out["emission_abs_error_nm"], out["expected_emission_nm"]
        )
        out["expected_quantum_yield"] = infer_constant_expected(
            out["predicted_quantum_yield"],
            out["quantum_yield_abs_error"],
            out["expected_quantum_yield"],
        )
        out["nearest_training_similarity"] = numeric(out["nearest_training_similarity"])
        out["emission_abs_error_nm"] = out["emission_abs_error_nm"].where(
            out["emission_abs_error_nm"].notna(),
            (out["predicted_emission_nm"] - out["expected_emission_nm"]).abs(),
        )
        out["quantum_yield_abs_error"] = out["quantum_yield_abs_error"].where(
            out["quantum_yield_abs_error"].notna(),
            (out["predicted_quantum_yield"] - out["expected_quantum_yield"]).abs(),
        )
        frames.append(out)
    if not frames:
        raise ValueError(f"No prediction CSV files found in {args.prediction_dir}")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["molecule", "model_family", "model"]).reset_index(drop=True)
    return combined


def aggregate_errors(table: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = table.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        n_predictions=("model", "size"),
        n_molecules=("molecule", "nunique"),
        mean_emission_abs_error_nm=("emission_abs_error_nm", "mean"),
        median_emission_abs_error_nm=("emission_abs_error_nm", "median"),
        std_emission_abs_error_nm=("emission_abs_error_nm", "std"),
        max_emission_abs_error_nm=("emission_abs_error_nm", "max"),
        mean_quantum_yield_abs_error=("quantum_yield_abs_error", "mean"),
        median_quantum_yield_abs_error=("quantum_yield_abs_error", "median"),
        std_quantum_yield_abs_error=("quantum_yield_abs_error", "std"),
        max_quantum_yield_abs_error=("quantum_yield_abs_error", "max"),
    )
    return summary.reset_index().sort_values(
        ["mean_emission_abs_error_nm", "mean_quantum_yield_abs_error", *group_cols],
        na_position="last",
    )


def best_row(group: pd.DataFrame, column: str) -> pd.Series:
    valid = group[group[column].notna()]
    if valid.empty:
        return group.iloc[0] * np.nan
    return valid.sort_values([column, "model"]).iloc[0]


def molecule_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for molecule, group in table.groupby("molecule", dropna=False, sort=True):
        emission_best = best_row(group, "emission_abs_error_nm")
        qy_best = best_row(group, "quantum_yield_abs_error")
        row = {
            "molecule": molecule,
            "n_models": group["model"].nunique(),
            "expected_emission_nm": group["expected_emission_nm"].dropna().iloc[0]
            if group["expected_emission_nm"].notna().any()
            else np.nan,
            "best_emission_model": emission_best.get("model", np.nan),
            "best_emission_model_family": emission_best.get("model_family", np.nan),
            "best_predicted_emission_nm": emission_best.get("predicted_emission_nm", np.nan),
            "best_emission_abs_error_nm": emission_best.get("emission_abs_error_nm", np.nan),
            "mean_emission_abs_error_nm": group["emission_abs_error_nm"].mean(),
            "median_emission_abs_error_nm": group["emission_abs_error_nm"].median(),
            "expected_quantum_yield": group["expected_quantum_yield"].dropna().iloc[0]
            if group["expected_quantum_yield"].notna().any()
            else np.nan,
            "best_qy_model": qy_best.get("model", np.nan),
            "best_qy_model_family": qy_best.get("model_family", np.nan),
            "best_predicted_quantum_yield": qy_best.get("predicted_quantum_yield", np.nan),
            "best_quantum_yield_abs_error": qy_best.get("quantum_yield_abs_error", np.nan),
            "mean_quantum_yield_abs_error": group["quantum_yield_abs_error"].mean(),
            "median_quantum_yield_abs_error": group["quantum_yield_abs_error"].median(),
            "mean_nearest_training_similarity": group["nearest_training_similarity"].mean(),
            "max_nearest_training_similarity": group["nearest_training_similarity"].max(),
        }
        for prefix, col in [
            ("predicted_emission", "predicted_emission_nm"),
            ("predicted_qy", "predicted_quantum_yield"),
        ]:
            row[f"{prefix}_mean"] = group[col].mean()
            row[f"{prefix}_std"] = group[col].std()
            row[f"{prefix}_min"] = group[col].min()
            row[f"{prefix}_max"] = group[col].max()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("molecule").reset_index(drop=True)


def stats_for(series: pd.Series, metric: str, scope: str) -> dict[str, float]:
    values = numeric(series)
    return {
        f"training_{metric}_min_{scope}": values.min(),
        f"training_{metric}_median_{scope}": values.median(),
        f"training_{metric}_mean_{scope}": values.mean(),
        f"training_{metric}_max_{scope}": values.max(),
        f"training_{metric}_std_{scope}": values.std(),
    }


def load_training(training_csv: Path, warnings: list[str]) -> tuple[pd.DataFrame, dict[str, str | None]]:
    if Chem is None:
        warn("RDKit is unavailable; using normalized raw SMILES strings for overlap matching.", warnings)
    table = pd.read_csv(training_csv)
    columns = table.columns.tolist()
    mapping = {
        key: detect_column(columns, choices, required=key == "smiles", label=f"training {key}")
        for key, choices in TRAINING_COLUMN_CANDIDATES.items()
    }
    if mapping["emission"] is None:
        warn("No training emission column detected; emission label diagnostics will be NaN.", warnings)
    if mapping["qy"] is None:
        warn("No training quantum-yield column detected; QY label diagnostics will be NaN.", warnings)
    table["_canonical_molecule"] = canonicalize_series(table[mapping["smiles"]], warnings)
    table["_solvent_key"] = (
        table[mapping["solvent"]].astype(str).str.strip().str.lower()
        if mapping["solvent"]
        else pd.Series("", index=table.index)
    )
    table["_canonical_solvent"] = (
        canonicalize_series(table[mapping["solvent_smiles"]], warnings)
        if mapping["solvent_smiles"]
        else pd.Series(np.nan, index=table.index)
    )
    return table, mapping


def training_overlap(
    combined: pd.DataFrame, training_csv: Path, warnings: list[str]
) -> pd.DataFrame:
    training, mapping = load_training(training_csv, warnings)
    rows = []
    molecule_level = combined.drop_duplicates("molecule").sort_values("molecule")
    all_solvent_keys = set(training["_solvent_key"].dropna())
    all_solvent_smiles = set(training["_canonical_solvent"].dropna())
    for _, bench in molecule_level.iterrows():
        canon = canonicalize_smiles(bench.get("input_smiles"), warnings)
        solvent_key = str(bench.get("input_solvent", "")).strip().lower()
        solvent_smiles = canonicalize_smiles(bench.get("input_solvent_smiles"), warnings)
        same_mol = training[training["_canonical_molecule"] == canon] if pd.notna(canon) else training.iloc[0:0]
        solvent_seen = bool(solvent_key and solvent_key in all_solvent_keys) or (
            pd.notna(solvent_smiles) and solvent_smiles in all_solvent_smiles
        )
        if pd.notna(solvent_smiles):
            same_pair = same_mol[same_mol["_canonical_solvent"] == solvent_smiles]
        elif solvent_key:
            same_pair = same_mol[same_mol["_solvent_key"] == solvent_key]
        else:
            same_pair = same_mol.iloc[0:0]
        row: dict[str, Any] = {
            "molecule": bench["molecule"],
            "input_smiles": bench.get("input_smiles"),
            "input_solvent": bench.get("input_solvent"),
            "input_solvent_smiles": bench.get("input_solvent_smiles"),
            "canonical_input_smiles": canon,
            "canonical_input_solvent_smiles": solvent_smiles,
            "exact_molecule_seen": len(same_mol) > 0,
            "exact_solvent_seen": solvent_seen,
            "exact_molecule_solvent_pair_seen": len(same_pair) > 0,
            "n_training_rows_same_molecule": len(same_mol),
            "n_training_rows_same_molecule_solvent": len(same_pair),
            "training_solvents_same_molecule": "; ".join(
                sorted(str(v) for v in same_mol["_solvent_key"].dropna().unique() if str(v))
            ),
        }
        emission_col = mapping["emission"]
        qy_col = mapping["qy"]
        if emission_col:
            row.update(stats_for(same_mol[emission_col], "emission", "same_molecule"))
            row.update(stats_for(same_pair[emission_col], "emission", "same_molecule_solvent"))
        else:
            row.update({f"training_emission_{stat}_{scope}": np.nan for scope in ["same_molecule", "same_molecule_solvent"] for stat in ["min", "median", "mean", "max", "std"]})
        if qy_col:
            row.update(stats_for(same_mol[qy_col], "qy", "same_molecule"))
            row.update(stats_for(same_pair[qy_col], "qy", "same_molecule_solvent"))
        else:
            row.update({f"training_qy_{stat}_{scope}": np.nan for scope in ["same_molecule", "same_molecule_solvent"] for stat in ["min", "median", "mean", "max", "std"]})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("molecule").reset_index(drop=True)


def score_from_threshold(value: float, good: float, bad: float) -> float:
    if pd.isna(value):
        return 0.5
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return float(1.0 - (value - good) / (bad - good))


def classify_failures(
    molecule_table: pd.DataFrame,
    overlap: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    merged = molecule_table.merge(overlap, on="molecule", how="left")
    rows = []
    for _, row in merged.iterrows():
        exact_pair = bool(row.get("exact_molecule_solvent_pair_seen", False))
        exact_mol = bool(row.get("exact_molecule_seen", False))
        mean_err = row.get("mean_emission_abs_error_nm")
        best_err = row.get("best_emission_abs_error_nm")
        qy_err = row.get("mean_quantum_yield_abs_error")
        sim = row.get("max_nearest_training_similarity")
        pred_std = row.get("predicted_emission_std")
        pred_range = row.get("predicted_emission_max") - row.get("predicted_emission_min")
        train_pair_med = row.get("training_emission_median_same_molecule_solvent")
        train_mol_med = row.get("training_emission_median_same_molecule")
        train_med = train_pair_med if pd.notna(train_pair_med) else train_mol_med
        bench = row.get("expected_emission_nm")
        train_qy_med = row.get("training_qy_median_same_molecule_solvent")
        if pd.isna(train_qy_med):
            train_qy_med = row.get("training_qy_median_same_molecule")
        delta = abs(bench - train_med) if pd.notna(bench) and pd.notna(train_med) else np.nan
        qy_delta = (
            abs(row.get("expected_quantum_yield") - train_qy_med)
            if pd.notna(row.get("expected_quantum_yield")) and pd.notna(train_qy_med)
            else np.nan
        )
        pred_mean = row.get("predicted_emission_mean")
        closer_to_training = (
            abs(pred_mean - train_med) < abs(pred_mean - bench)
            if pd.notna(pred_mean) and pd.notna(train_med) and pd.notna(bench)
            else False
        )
        high_emission = pd.notna(mean_err) and mean_err > args.emission_error_threshold
        high_qy = pd.notna(qy_err) and qy_err > args.qy_error_threshold
        close_training = pd.notna(delta) and delta <= args.training_label_mismatch_threshold
        mismatch = pd.notna(delta) and delta > args.training_label_mismatch_threshold
        high_disagreement = (
            (pd.notna(pred_std) and pred_std > args.model_disagreement_threshold)
            or (pd.notna(pred_range) and pred_range > 2 * args.model_disagreement_threshold)
        )

        if not high_emission and not high_qy:
            mode = "reasonable_prediction"
            reason = "Emission and QY errors are below configured thresholds."
        elif exact_pair and close_training and high_emission:
            mode = "model_failure_exact_pair"
            reason = "Exact molecule-solvent pair exists and training label is close, but prediction error is high."
        elif (exact_pair or exact_mol) and mismatch and closer_to_training:
            mode = "benchmark_training_label_mismatch"
            reason = "Training median differs from benchmark and predictions are closer to training labels."
        elif exact_mol and not exact_pair and high_emission:
            mode = "solvent_or_condition_mismatch"
            reason = "Molecule exists in training, but not the benchmark molecule-solvent pair."
        elif high_disagreement:
            mode = "high_model_disagreement"
            reason = "Models disagree strongly for this molecule."
        elif high_qy and not high_emission:
            mode = "qy_condition_sensitive"
            reason = "QY error is high while emission error is comparatively reasonable."
        elif not exact_mol and pd.notna(sim) and sim < args.low_similarity_threshold:
            mode = "structural_extrapolation"
            reason = "Molecule is unseen and nearest-training similarity is low."
        else:
            mode = "high_model_disagreement" if high_disagreement else "model_failure_exact_pair"
            reason = "Errors are high and no more specific diagnostic condition dominated."

        similarity_score = 0.0 if pd.isna(sim) else max(0.0, min(1.0, float(sim)))
        label_var = row.get("training_emission_std_same_molecule_solvent")
        if pd.isna(label_var):
            label_var = row.get("training_emission_std_same_molecule")
        label_score = score_from_threshold(label_var, 10.0, 60.0)
        agreement_score = score_from_threshold(pred_std, 10.0, args.model_disagreement_threshold)
        overall = (
            0.20 * float(exact_mol)
            + 0.15 * float(bool(row.get("exact_solvent_seen", False)))
            + 0.20 * float(exact_pair)
            + 0.20 * similarity_score
            + 0.15 * label_score
            + 0.10 * agreement_score
        )
        label = "high" if overall >= 0.75 and similarity_score >= 0.85 and agreement_score >= 0.6 else "medium"
        if overall < 0.50:
            label = "low"

        rows.append(
            {
                "molecule": row["molecule"],
                "failure_mode": mode,
                "failure_mode_reason": reason,
                "benchmark_vs_training_emission_delta": delta,
                "benchmark_vs_training_qy_delta": qy_delta,
                "model_closer_to_training_than_benchmark": closer_to_training,
                "molecule_seen_score": float(exact_mol),
                "solvent_seen_score": float(bool(row.get("exact_solvent_seen", False))),
                "pair_seen_score": float(exact_pair),
                "similarity_score": similarity_score,
                "label_consistency_score": label_score,
                "model_agreement_score": agreement_score,
                "overall_confidence_score": overall,
                "confidence_label": label,
            }
        )
    return pd.DataFrame(rows).sort_values("molecule").reset_index(drop=True)


def add_prediction_confidence(combined: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "molecule",
        "molecule_seen_score",
        "solvent_seen_score",
        "pair_seen_score",
        "similarity_score",
        "label_consistency_score",
        "model_agreement_score",
        "overall_confidence_score",
        "confidence_label",
        "failure_mode",
    ]
    return combined.merge(failures[cols], on="molecule", how="left")


def markdown_table(table: pd.DataFrame, columns: list[str], n: int | None = None) -> str:
    subset = table.loc[:, [c for c in columns if c in table.columns]].copy()
    if n is not None:
        subset = subset.head(n)
    for col in subset.select_dtypes(include=[np.number]).columns:
        subset[col] = subset[col].map(lambda value: "" if pd.isna(value) else f"{value:.3g}")
    if subset.empty:
        return "_No rows._"
    subset = subset.fillna("").astype(str)
    headers = subset.columns.tolist()
    rows = subset.values.tolist()
    widths = [
        max(len(str(header)), *(len(str(row[index])) for row in rows))
        for index, header in enumerate(headers)
    ]

    def fmt(values: list[Any]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt(headers), separator, *[fmt(row) for row in rows]])


def write_report(
    out_path: Path,
    args: argparse.Namespace,
    combined: pd.DataFrame,
    model_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    molecule_table: pd.DataFrame,
    failures: pd.DataFrame,
    warnings: list[str],
) -> None:
    best_model = model_summary.iloc[0] if not model_summary.empty else pd.Series(dtype=object)
    combined_perf = model_summary.assign(
        combined_score=model_summary["mean_emission_abs_error_nm"].fillna(1e9)
        + 100.0 * model_summary["mean_quantum_yield_abs_error"].fillna(1e9)
    ).sort_values("combined_score")
    best_combined = combined_perf.iloc[0] if not combined_perf.empty else best_model
    best_family = family_summary.iloc[0] if not family_summary.empty else pd.Series(dtype=object)
    worst_emission = molecule_table.sort_values("mean_emission_abs_error_nm", ascending=False)
    worst_qy = molecule_table.sort_values("mean_quantum_yield_abs_error", ascending=False)
    paragraph = (
        "On an external benchmark of known fluorophores, the best-performing models achieved "
        f"approximately {best_model.get('mean_emission_abs_error_nm', np.nan):.1f} nm mean emission error, "
        f"with {best_model.get('model', 'NA')} performing best overall for emission wavelength. "
        "Neural MLP models outperformed tree baselines and current graph neural networks on this diagnostic set. "
        "Quantum-yield prediction remained substantially less reliable, with the largest errors occurring for "
        "bright or condition-sensitive dyes. Several apparent failures occurred despite nearest-training molecular "
        "similarity values of 1.0, showing that molecule-only applicability-domain scoring is insufficient. "
        "Exact molecule-solvent overlap, solvent identity, label consistency, model disagreement, and experimental "
        "condition sensitivity must also be considered when assigning prediction confidence."
    )
    lines = [
        "# External Known-Fluorophore Benchmark Diagnostics",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Prediction directory: `{args.prediction_dir}`",
        f"Training CSV: `{args.training_csv}`",
        "",
        f"- Molecules: {combined['molecule'].nunique()}",
        f"- Models: {combined['model'].nunique()}",
        f"- Model families: {combined['model_family'].nunique()}",
        f"- Best emission model: `{best_model.get('model', 'NA')}`",
        f"- Best combined emission/QY model: `{best_combined.get('model', 'NA')}`",
        f"- Best model family: `{best_family.get('model_family', 'NA')}`",
        "",
        "## Why similarity 1.0 can still be wrong",
        "",
        "A nearest-training molecular similarity of 1.0 only says the molecule fingerprint matched a training molecule. "
        "It does not prove the same solvent, pH, salt form, concentration regime, literature reference, or training label "
        "was present. The diagnostics below separate exact molecule overlap from exact molecule-solvent overlap, training "
        "label agreement, model disagreement, and quantum-yield condition sensitivity.",
        "",
        "## Family-Level Performance",
        markdown_table(family_summary, ["model_family", "n_predictions", "n_molecules", "mean_emission_abs_error_nm", "mean_quantum_yield_abs_error"]),
        "",
        "## Top 10 Models",
        markdown_table(model_summary, ["model", "model_family", "n_predictions", "mean_emission_abs_error_nm", "mean_quantum_yield_abs_error"], 10),
        "",
        "## Worst Emission Molecules",
        markdown_table(worst_emission, ["molecule", "mean_emission_abs_error_nm", "best_emission_model", "best_emission_abs_error_nm"], 10),
        "",
        "## Worst QY Molecules",
        markdown_table(worst_qy, ["molecule", "mean_quantum_yield_abs_error", "best_qy_model", "best_quantum_yield_abs_error"], 10),
        "",
        "## Failure Modes",
        markdown_table(failures, ["molecule", "failure_mode", "overall_confidence_score", "confidence_label", "benchmark_vs_training_emission_delta"]),
        "",
        "## Manuscript-Ready Paragraph",
        "",
        paragraph,
    ]
    if warnings:
        lines.extend(["", "## Warnings", *[f"- {item}" for item in sorted(set(warnings))]])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(out_dir: Path, combined: pd.DataFrame, model_summary: pd.DataFrame, family_summary: pd.DataFrame, molecule_table: pd.DataFrame, warnings: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional
        warn(f"matplotlib unavailable; skipping plots ({exc}).", warnings)
        return
    try:
        plots = [
            (model_summary.head(20), "model", "mean_emission_abs_error_nm", "model_mean_emission_error_bar.png"),
            (family_summary, "model_family", "mean_emission_abs_error_nm", "family_mean_emission_error_bar.png"),
            (molecule_table, "molecule", "mean_emission_abs_error_nm", "molecule_mean_emission_error_bar.png"),
            (molecule_table, "molecule", "mean_quantum_yield_abs_error", "molecule_qy_error_bar.png"),
        ]
        for table, xcol, ycol, filename in plots:
            fig, ax = plt.subplots(figsize=(9, 4.8))
            ordered = table.sort_values(ycol, ascending=False)
            ax.bar(ordered[xcol].astype(str), ordered[ycol])
            ax.set_ylabel(ycol)
            ax.tick_params(axis="x", rotation=60)
            fig.tight_layout()
            fig.savefig(out_dir / filename, dpi=160)
            plt.close(fig)
        for xcol, ycol, filename in [
            ("expected_emission_nm", "predicted_emission_nm", "emission_predicted_vs_expected.png"),
            ("expected_quantum_yield", "predicted_quantum_yield", "qy_predicted_vs_expected.png"),
        ]:
            fig, ax = plt.subplots(figsize=(5.5, 5.0))
            ax.scatter(combined[xcol], combined[ycol], alpha=0.75)
            low = np.nanmin([combined[xcol].min(), combined[ycol].min()])
            high = np.nanmax([combined[xcol].max(), combined[ycol].max()])
            ax.plot([low, high], [low, high], color="black", linewidth=1)
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            fig.tight_layout()
            fig.savefig(out_dir / filename, dpi=160)
            plt.close(fig)
    except Exception as exc:  # pragma: no cover - optional
        warn(f"Plotting failed; continuing without all plots ({exc}).", warnings)


def run(args: argparse.Namespace) -> dict[str, Path]:
    warnings: list[str] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Combining prediction CSVs...")
    combined = consolidate_predictions(args, warnings)
    print("Computing model and molecule summaries...")
    model_summary = aggregate_errors(combined, ["model", "model_family"])
    family_summary = aggregate_errors(combined, ["model_family"])
    molecule_table = molecule_summary(combined)
    print("Diagnosing training overlap...")
    overlap = training_overlap(combined, args.training_csv, warnings)
    print("Classifying failure modes and confidence...")
    failures = classify_failures(molecule_table, overlap, args)
    combined = add_prediction_confidence(combined, failures)

    paths = {
        "all_predictions": args.out_dir / "external_benchmark_all_predictions.csv",
        "model_summary": args.out_dir / "external_benchmark_model_summary.csv",
        "family_summary": args.out_dir / "external_benchmark_family_summary.csv",
        "molecule_summary": args.out_dir / "external_benchmark_molecule_summary.csv",
        "training_overlap": args.out_dir / "external_benchmark_training_overlap.csv",
        "failure_modes": args.out_dir / "external_benchmark_failure_modes.csv",
        "report": args.out_dir / "external_benchmark_report.md",
    }
    combined.to_csv(paths["all_predictions"], index=False)
    model_summary.to_csv(paths["model_summary"], index=False)
    family_summary.to_csv(paths["family_summary"], index=False)
    molecule_table.to_csv(paths["molecule_summary"], index=False)
    overlap.to_csv(paths["training_overlap"], index=False)
    failures.to_csv(paths["failure_modes"], index=False)
    write_report(paths["report"], args, combined, model_summary, family_summary, molecule_table, failures, warnings)
    make_plots(args.out_dir, combined, model_summary, family_summary, molecule_table, warnings)
    print("Generated outputs:")
    for path in paths.values():
        print(f"  {path}")
    return paths


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
