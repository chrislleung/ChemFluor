"""Train combined ChemFluor + Deep4Chem solvent-aware predictors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.data_standardization import (  # noqa: E402
    TARGET_COLUMNS,
    canonicalize_smiles,
    combine_training_data,
    load_chemfluor,
    load_deep4chem,
)

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
except ImportError as exc:  # pragma: no cover - only exercised without RDKit.
    Chem = None
    DataStructs = None
    AllChem = None
    _RDKIT_IMPORT_ERROR = exc
else:
    _RDKIT_IMPORT_ERROR = None
    RDLogger.DisableLog("rdApp.*")


DEFAULT_DEEP4CHEM = Path("data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv")
DEFAULT_CHEMFLUOR = Path("data/chemfluor_data.csv")
DEFAULT_SOLVENT_DESCRIPTORS = Path("data/solvent_descriptors_expanded_deep4chem.csv")
DEFAULT_OUT_DIR = Path("models/chemfluor_combined")

IDENTITY_DESCRIPTOR_COLUMNS = {
    "solvent",
    "solvent_original",
    "canonical_smiles",
    "canonical_solvent_smiles",
    "is_valid_rdkit",
    "is_environment_label",
    "deep4chem_row_count",
    "existing_solvent_match",
    "existing_canonical_solvent_smiles",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Train combined ChemFluor + Deep4Chem optical-property predictors."
    )
    parser.add_argument("--deep4chem", default=DEFAULT_DEEP4CHEM, type=Path)
    parser.add_argument("--chemfluor", default=DEFAULT_CHEMFLUOR, type=Path)
    parser.add_argument(
        "--solvent-descriptors", default=DEFAULT_SOLVENT_DESCRIPTORS, type=Path
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument("--model", choices=["rf", "histgb"], default="rf")
    parser.add_argument("--n-bits", default=2048, type=int)
    parser.add_argument("--radius", default=2, type=int)
    return parser.parse_args()


def require_rdkit() -> None:
    """Raise a helpful error if RDKit is unavailable."""
    if Chem is None or DataStructs is None or AllChem is None:
        raise ImportError("RDKit is required to generate Morgan fingerprints.") from _RDKIT_IMPORT_ERROR


def load_combined_rows(deep4chem_path: Path, chemfluor_path: Path) -> pd.DataFrame:
    """Load standardized combined training rows using the reusable data layer."""
    # Keep direct references visible for users of this script and static analyzers.
    _ = (load_deep4chem, load_chemfluor, canonicalize_smiles)
    return combine_training_data(deep4chem_path, chemfluor_path)


def load_solvent_descriptors(path: Path) -> pd.DataFrame:
    """Load and normalize solvent descriptor key columns."""
    if not path.exists():
        raise FileNotFoundError(f"Solvent descriptor file not found: {path}")

    descriptors = pd.read_csv(path, low_memory=False)
    if descriptors.empty:
        raise ValueError(f"Solvent descriptor file is empty: {path}")

    if "canonical_solvent_smiles" not in descriptors.columns:
        if "canonical_smiles" in descriptors.columns:
            descriptors["canonical_solvent_smiles"] = descriptors["canonical_smiles"]
        else:
            descriptors["canonical_solvent_smiles"] = pd.NA
    elif "canonical_smiles" in descriptors.columns:
        descriptors["canonical_solvent_smiles"] = descriptors[
            "canonical_solvent_smiles"
        ].fillna(descriptors["canonical_smiles"])

    if "solvent_original" not in descriptors.columns:
        if "solvent" in descriptors.columns:
            descriptors["solvent_original"] = descriptors["solvent"]
        else:
            descriptors["solvent_original"] = pd.NA

    descriptors["canonical_solvent_smiles"] = descriptors[
        "canonical_solvent_smiles"
    ].apply(lambda value: value if pd.notna(value) and str(value).strip() else pd.NA)
    descriptors["solvent_original"] = descriptors["solvent_original"].apply(
        lambda value: str(value).strip() if pd.notna(value) else pd.NA
    )
    descriptors["descriptor_canonical_key"] = descriptors[
        "canonical_solvent_smiles"
    ].astype("string")
    descriptors["descriptor_solvent_key"] = descriptors["solvent_original"].astype(
        "string"
    ).str.lower()
    return descriptors


def choose_solvent_descriptor_columns(descriptors: pd.DataFrame) -> list[str]:
    """Select numeric solvent descriptor columns and exclude identity/target fields."""
    descriptor_columns: list[str] = []
    excluded = IDENTITY_DESCRIPTOR_COLUMNS | set(TARGET_COLUMNS)
    excluded.update({"descriptor_canonical_key", "descriptor_solvent_key"})

    for column in descriptors.columns:
        if column in excluded:
            continue
        numeric = pd.to_numeric(descriptors[column], errors="coerce")
        if numeric.notna().any():
            descriptors[column] = numeric
            descriptor_columns.append(column)
    return descriptor_columns


def merge_solvent_descriptors(
    combined_rows: pd.DataFrame, descriptors: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Merge solvent descriptors by canonical SMILES first, then solvent label."""
    descriptor_columns = choose_solvent_descriptor_columns(descriptors)
    descriptor_payload = [
        "descriptor_canonical_key",
        "descriptor_solvent_key",
        *descriptor_columns,
    ]

    canonical_descriptors = descriptors.dropna(
        subset=["descriptor_canonical_key"]
    ).drop_duplicates(subset=["descriptor_canonical_key"], keep="first")[
        ["descriptor_canonical_key", *descriptor_columns]
    ]
    label_descriptors = descriptors.dropna(
        subset=["descriptor_solvent_key"]
    ).drop_duplicates(subset=["descriptor_solvent_key"], keep="first")[
        ["descriptor_solvent_key", *descriptor_columns]
    ]

    modeling_rows = combined_rows.copy()
    modeling_rows["row_id"] = np.arange(len(modeling_rows))
    modeling_rows["descriptor_canonical_key"] = modeling_rows[
        "canonical_solvent_smiles"
    ].astype("string")
    modeling_rows["descriptor_solvent_key"] = modeling_rows["solvent_original"].astype(
        "string"
    ).str.lower()

    modeling_rows = modeling_rows.merge(
        canonical_descriptors,
        how="left",
        on="descriptor_canonical_key",
    )

    unmatched = modeling_rows[descriptor_columns].isna().all(axis=1)
    if unmatched.any():
        fallback = modeling_rows.loc[
            unmatched, ["row_id", "descriptor_solvent_key"]
        ].merge(label_descriptors, how="left", on="descriptor_solvent_key")
        fallback = fallback.set_index("row_id")
        for column in descriptor_columns:
            modeling_rows.loc[unmatched, column] = modeling_rows.loc[
                unmatched, "row_id"
            ].map(fallback[column])

    return (
        modeling_rows.drop(columns=["row_id", *descriptor_payload[:2]]),
        descriptor_columns,
    )


def morgan_fingerprint(smiles: str, radius: int, n_bits: int) -> np.ndarray | None:
    """Generate a Morgan fingerprint bit vector as a NumPy array."""
    require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    bit_vector = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits
    )
    array = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    return array


def add_fingerprints(
    modeling_rows: pd.DataFrame, radius: int, n_bits: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Create fingerprints and drop rows whose canonical chromophore cannot parse."""
    fingerprints: list[np.ndarray] = []
    valid_indices: list[int] = []

    for index, smiles in modeling_rows["canonical_chromophore_smiles"].items():
        fingerprint = morgan_fingerprint(str(smiles), radius=radius, n_bits=n_bits)
        if fingerprint is not None:
            fingerprints.append(fingerprint)
            valid_indices.append(index)

    if not fingerprints:
        raise ValueError("No valid chromophore fingerprints could be generated.")

    filtered_rows = modeling_rows.loc[valid_indices].reset_index(drop=True)
    return filtered_rows, np.vstack(fingerprints)


def make_model(model_type: str) -> Any:
    """Construct the requested regressor."""
    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        random_state=42,
    )


def build_feature_matrix(
    fingerprints: np.ndarray,
    descriptor_values: pd.DataFrame,
    medians: pd.Series,
) -> np.ndarray:
    """Combine fingerprint bits with median-imputed solvent descriptor values."""
    imputed_descriptors = descriptor_values.fillna(medians).fillna(0.0)
    return np.hstack(
        [fingerprints, imputed_descriptors.to_numpy(dtype=np.float32, copy=True)]
    )


def train_one_target(
    target: str,
    model_type: str,
    rows: pd.DataFrame,
    fingerprints: np.ndarray,
    descriptor_columns: list[str],
    n_bits: int,
    out_dir: Path,
) -> tuple[dict[str, Any] | None, pd.Series | None]:
    """Train, evaluate, and save one target-specific regressor."""
    target_rows = rows[rows[target].notna()].copy()
    if len(target_rows) < 100:
        print(f"WARNING: skipping {target}; only {len(target_rows)} usable rows.")
        return None, None

    target_fingerprints = fingerprints[target_rows.index.to_numpy()]
    groups = target_rows["canonical_chromophore_smiles"].to_numpy()
    if pd.Series(groups).nunique() < 2:
        print(f"WARNING: skipping {target}; fewer than 2 chromophore groups.")
        return None, None

    splitter = GroupShuffleSplit(test_size=0.2, random_state=42, n_splits=1)
    train_index, test_index = next(splitter.split(target_rows, groups=groups))

    descriptor_values = target_rows[descriptor_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    train_descriptors = descriptor_values.iloc[train_index]
    medians = train_descriptors.median(numeric_only=True)

    x_train = build_feature_matrix(
        target_fingerprints[train_index],
        descriptor_values.iloc[train_index],
        medians,
    )
    x_test = build_feature_matrix(
        target_fingerprints[test_index],
        descriptor_values.iloc[test_index],
        medians,
    )
    y_train = target_rows[target].iloc[train_index].to_numpy(dtype=float)
    y_test = target_rows[target].iloc[test_index].to_numpy(dtype=float)

    model = make_model(model_type)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    predictions = target_rows.iloc[test_index][
        [
            "canonical_chromophore_smiles",
            "solvent_original",
            "canonical_solvent_smiles",
            "source_dataset",
        ]
    ].copy()
    predictions["y_true"] = y_test
    predictions["y_pred"] = y_pred
    predictions["residual"] = predictions["y_true"] - predictions["y_pred"]

    model_path = out_dir / f"{target}_{model_type}.joblib"
    prediction_path = out_dir / f"predictions_{target}.csv"
    joblib.dump(model, model_path)
    predictions.to_csv(prediction_path, index=False)

    metrics = {
        "target": target,
        "model_type": model_type,
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)) if len(y_test) > 1 else float("nan"),
        "train_rows": int(len(train_index)),
        "test_rows": int(len(test_index)),
        "unique_train_chromophores": int(
            target_rows.iloc[train_index]["canonical_chromophore_smiles"].nunique()
        ),
        "unique_test_chromophores": int(
            target_rows.iloc[test_index]["canonical_chromophore_smiles"].nunique()
        ),
        "n_solvent_descriptor_columns": int(len(descriptor_columns)),
        "n_fingerprint_bits": int(n_bits),
        "model_path": str(model_path),
        "prediction_path": str(prediction_path),
    }
    return metrics, medians


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def print_metrics_summary(metrics_by_target: dict[str, dict[str, Any]]) -> None:
    """Print a compact metrics summary table."""
    if not metrics_by_target:
        print("No targets were trained.")
        return

    summary = pd.DataFrame(metrics_by_target.values())[
        [
            "target",
            "mae",
            "rmse",
            "r2",
            "train_rows",
            "test_rows",
            "unique_train_chromophores",
            "unique_test_chromophores",
        ]
    ]
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print("\nMetrics summary:")
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> int:
    """Run combined model training."""
    args = parse_args()

    try:
        require_rdkit()
        args.out_dir.mkdir(parents=True, exist_ok=True)

        combined_rows = load_combined_rows(args.deep4chem, args.chemfluor)
        combined_path = args.out_dir / "combined_standardized_training_rows.csv"
        combined_rows.to_csv(combined_path, index=False)

        solvent_descriptors = load_solvent_descriptors(args.solvent_descriptors)
        modeling_rows, descriptor_columns = merge_solvent_descriptors(
            combined_rows, solvent_descriptors
        )
        modeling_rows, fingerprints = add_fingerprints(
            modeling_rows, radius=args.radius, n_bits=args.n_bits
        )
        modeling_path = args.out_dir / "combined_modeling_rows_after_feature_merge.csv"
        modeling_rows.to_csv(modeling_path, index=False)

        metrics_by_target: dict[str, dict[str, Any]] = {}
        medians_by_target: dict[str, dict[str, float | None]] = {}

        for target in TARGET_COLUMNS:
            metrics, medians = train_one_target(
                target=target,
                model_type=args.model,
                rows=modeling_rows,
                fingerprints=fingerprints,
                descriptor_columns=descriptor_columns,
                n_bits=args.n_bits,
                out_dir=args.out_dir,
            )
            if metrics is None or medians is None:
                continue
            metrics_by_target[target] = metrics
            medians_by_target[target] = {
                key: (None if pd.isna(value) else float(value))
                for key, value in medians.items()
            }

        feature_metadata = {
            "fingerprint_radius": args.radius,
            "fingerprint_n_bits": args.n_bits,
            "solvent_descriptor_columns_used": descriptor_columns,
            "target_columns": TARGET_COLUMNS,
            "model_type": args.model,
            "median_values_used_for_imputation": medians_by_target,
        }
        save_json(args.out_dir / "feature_metadata.json", feature_metadata)
        save_json(args.out_dir / "metrics.json", metrics_by_target)

        print(f"Saved standardized rows to: {combined_path}")
        print(f"Saved modeling rows to: {modeling_path}")
        print(f"Saved feature metadata to: {args.out_dir / 'feature_metadata.json'}")
        print(f"Saved metrics to: {args.out_dir / 'metrics.json'}")
        print_metrics_summary(metrics_by_target)
        return 0
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
