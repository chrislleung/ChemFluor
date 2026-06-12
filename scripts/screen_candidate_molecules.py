"""Screen candidate molecules with trained combined ChemFluor RF models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.data_standardization import canonicalize_smiles  # noqa: E402

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


DEFAULT_MODEL_DIR = Path("models/chemfluor_combined")
DEFAULT_SOLVENT_DESCRIPTORS = Path("data/solvent_descriptors_expanded_deep4chem_chatgpt.csv")
DEFAULT_OUT = Path("outputs/candidate_screening/ranked_candidates.csv")
DEFAULT_APPLICABILITY_REFERENCE = "combined_modeling_rows_after_feature_merge.csv"
DEFAULT_APPLICABILITY_THRESHOLD = 0.30

REQUIRED_TARGETS = ["emission_nm", "absorption_nm", "quantum_yield"]
OPTIONAL_TARGETS = ["log_extinction"]
PREDICTION_OUTPUT_COLUMNS = {
    "absorption_nm": "predicted_absorption_nm",
    "emission_nm": "predicted_emission_nm",
    "quantum_yield": "predicted_quantum_yield",
    "log_extinction": "predicted_log_extinction",
}
PREFERRED_METADATA_COLUMNS = ["name", "scaffold", "substituent"]
CORE_OUTPUT_COLUMNS = [
    "smiles",
    "canonical_smiles",
    "solvent_smiles",
    "predicted_absorption_nm",
    "predicted_emission_nm",
    "predicted_quantum_yield",
    "predicted_log_extinction",
    "nearest_training_similarity",
    "nearest_training_smiles",
    "outside_applicability_domain",
    "emission_error_from_target",
    "score",
    "estimated_brightness_score",
]
INTERNAL_TEMPORARY_COLUMNS = set()
REFERENCE_SMILES_COLUMNS = [
    "canonical_chromophore_smiles",
    "chromophore_smiles",
    "canonical_smiles",
    "smiles",
]
CANDIDATE_APPLICABILITY_SMILES_COLUMNS = ["canonical_smiles", "smiles"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rank candidate molecules with trained ChemFluor combined RF models."
    )
    parser.add_argument(
        "--candidates",
        required=True,
        type=Path,
        help="CSV containing at least a smiles column.",
    )
    parser.add_argument(
        "--solvent-smiles",
        required=True,
        help="Solvent SMILES used for screening, for example CCO for ethanol.",
    )
    parser.add_argument(
        "--target-emission",
        required=True,
        type=float,
        help="Desired emission wavelength in nm.",
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, type=Path)
    parser.add_argument(
        "--solvent-descriptors", default=DEFAULT_SOLVENT_DESCRIPTORS, type=Path
    )
    parser.add_argument(
        "--applicability-reference-csv",
        type=Path,
        default=None,
        help=(
            "CSV containing reference training/modeling chromophore SMILES. "
            "Defaults to <model-dir>/combined_modeling_rows_after_feature_merge.csv "
            "when available."
        ),
    )
    parser.add_argument(
        "--applicability-threshold",
        default=DEFAULT_APPLICABILITY_THRESHOLD,
        type=float,
        help="Tanimoto similarity threshold below which candidates are flagged.",
    )
    parser.add_argument(
        "--no-applicability-domain",
        action="store_true",
        help="Disable Morgan fingerprint Tanimoto applicability-domain scoring.",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, type=Path)
    return parser.parse_args()


def require_rdkit() -> None:
    """Raise a helpful error if RDKit is unavailable."""
    if Chem is None or DataStructs is None or AllChem is None:
        raise ImportError("RDKit is required to screen candidate molecules.") from _RDKIT_IMPORT_ERROR


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file with a helpful missing-file error."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def find_smiles_column(df: pd.DataFrame, candidates: Sequence[str] | None = None) -> str:
    """Find a SMILES column using a priority-ordered list of candidate names."""
    candidate_names = list(candidates) if candidates is not None else ["smiles"]
    normalized_columns = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidate_names:
        column = normalized_columns.get(candidate.lower())
        if column is not None:
            return column
    raise ValueError(
        "CSV must contain one of these SMILES columns: " + ", ".join(candidate_names)
    )


def load_candidates(path: Path) -> pd.DataFrame:
    """Load candidates and canonicalize molecule SMILES."""
    if not path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {path}")

    candidates = pd.read_csv(path)
    smiles_column = find_smiles_column(candidates)
    candidates = candidates.copy()
    candidates["smiles"] = candidates[smiles_column].astype(str).str.strip()
    candidates["canonical_smiles"] = candidates["smiles"].map(canonicalize_smiles)
    before = len(candidates)
    candidates = candidates.dropna(subset=["canonical_smiles"]).copy()
    dropped = before - len(candidates)
    if dropped:
        print(f"WARNING: dropped {dropped} candidate(s) with invalid SMILES.")
    if candidates.empty:
        raise ValueError("No valid candidate SMILES remain after canonicalization.")
    return candidates.reset_index(drop=True)


def load_models(model_dir: Path) -> dict[str, Any]:
    """Load required and optional RF models from the model directory."""
    models: dict[str, Any] = {}
    for target in REQUIRED_TARGETS:
        path = model_dir / f"{target}_rf.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Required model not found: {path}")
        models[target] = joblib.load(path)

    for target in OPTIONAL_TARGETS:
        path = model_dir / f"{target}_rf.joblib"
        if path.exists():
            models[target] = joblib.load(path)
        else:
            print(f"WARNING: optional model not found, skipping {target}: {path}")

    return models


def morgan_fingerprint(smiles: str, radius: int, n_bits: int) -> np.ndarray | None:
    """Generate a Morgan fingerprint bit vector for one canonical SMILES."""
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


def build_fingerprints(canonical_smiles: pd.Series, radius: int, n_bits: int) -> np.ndarray:
    """Build a stacked fingerprint matrix for valid canonical SMILES."""
    fingerprints = [
        morgan_fingerprint(smiles, radius=radius, n_bits=n_bits)
        for smiles in canonical_smiles
    ]
    if any(fingerprint is None for fingerprint in fingerprints):
        raise ValueError("A canonical candidate SMILES failed RDKit fingerprinting.")
    return np.vstack(fingerprints)


def morgan_bitvect(smiles: str, radius: int, n_bits: int) -> Any | None:
    """Generate an RDKit Morgan fingerprint bit vector for Tanimoto similarity."""
    require_rdkit()
    canonical = canonicalize_smiles(str(smiles))
    if canonical is None:
        return None
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


def infer_applicability_reference(model_dir: Path) -> Path | None:
    """Infer the default applicability-reference CSV from the model directory."""
    reference_path = model_dir / DEFAULT_APPLICABILITY_REFERENCE
    if reference_path.exists():
        return reference_path
    return None


def load_reference_fingerprints(
    reference_csv: Path,
    radius: int,
    n_bits: int,
) -> tuple[list[Any], list[str]]:
    """Load, canonicalize, deduplicate, and fingerprint reference chromophores."""
    if not reference_csv.exists():
        raise FileNotFoundError(f"Applicability reference CSV not found: {reference_csv}")

    reference_rows = pd.read_csv(reference_csv, low_memory=False)
    smiles_column = find_smiles_column(reference_rows, REFERENCE_SMILES_COLUMNS)
    canonical_smiles = (
        reference_rows[smiles_column]
        .dropna()
        .astype(str)
        .map(canonicalize_smiles)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    reference_fps: list[Any] = []
    reference_smiles: list[str] = []
    for smiles in canonical_smiles:
        fingerprint = morgan_bitvect(smiles, radius=radius, n_bits=n_bits)
        if fingerprint is None:
            continue
        reference_fps.append(fingerprint)
        reference_smiles.append(smiles)

    if not reference_fps:
        raise ValueError(
            f"No valid reference chromophore fingerprints found in {reference_csv}"
        )
    return reference_fps, reference_smiles


def compute_nearest_training_similarity(
    candidate_smiles: object,
    reference_fps: Sequence[Any],
    reference_smiles: Sequence[str],
    radius: int,
    n_bits: int,
) -> tuple[float, str]:
    """Find the nearest reference chromophore by Morgan Tanimoto similarity."""
    if pd.isna(candidate_smiles):
        return float("nan"), ""

    candidate_fp = morgan_bitvect(str(candidate_smiles), radius=radius, n_bits=n_bits)
    if candidate_fp is None:
        return float("nan"), ""

    similarities = DataStructs.BulkTanimotoSimilarity(candidate_fp, list(reference_fps))
    if not similarities:
        return float("nan"), ""

    best_index = max(range(len(similarities)), key=lambda index: similarities[index])
    return float(similarities[best_index]), reference_smiles[best_index]


def add_applicability_domain_columns(
    df: pd.DataFrame,
    reference_csv: Path,
    threshold: float,
    radius: int = 2,
    n_bits: int = 2048,
) -> pd.DataFrame:
    """Add nearest-reference Morgan Tanimoto applicability-domain columns."""
    smiles_column = find_smiles_column(df, CANDIDATE_APPLICABILITY_SMILES_COLUMNS)
    reference_fps, reference_smiles = load_reference_fingerprints(
        reference_csv=reference_csv,
        radius=radius,
        n_bits=n_bits,
    )

    nearest = [
        compute_nearest_training_similarity(
            candidate_smiles=smiles,
            reference_fps=reference_fps,
            reference_smiles=reference_smiles,
            radius=radius,
            n_bits=n_bits,
        )
        for smiles in df[smiles_column]
    ]

    output = df.copy()
    output["nearest_training_similarity"] = [value[0] for value in nearest]
    output["nearest_training_smiles"] = [value[1] for value in nearest]
    output["outside_applicability_domain"] = (
        output["nearest_training_similarity"].isna()
        | (output["nearest_training_similarity"] < threshold)
    )
    return output


def maybe_add_applicability_domain_columns(
    df: pd.DataFrame,
    model_dir: Path,
    reference_csv: Path | None,
    threshold: float,
    metadata: dict[str, Any],
    disabled: bool,
) -> pd.DataFrame:
    """Add applicability columns when a reference CSV is available."""
    if disabled:
        print("Applicability-domain scoring disabled.")
        return df
    if threshold < 0 or threshold > 1:
        raise ValueError(
            f"Applicability threshold must be between 0 and 1, got {threshold}."
        )

    selected_reference = reference_csv or infer_applicability_reference(model_dir)
    if selected_reference is None:
        print(
            "WARNING: no applicability reference CSV found; continuing without "
            "applicability-domain columns."
        )
        return df
    if not selected_reference.exists():
        print(
            "WARNING: applicability reference CSV not found: "
            f"{selected_reference}; continuing without applicability-domain columns."
        )
        return df

    # The current training script saves all post-feature-merge modeling rows but does
    # not persist per-row train/test membership. Until that split membership is saved,
    # this reference set may include both train and held-out chromophores.
    radius = int(metadata.get("fingerprint_radius", 2))
    n_bits = int(metadata.get("fingerprint_n_bits", 2048))
    output = add_applicability_domain_columns(
        df=df,
        reference_csv=selected_reference,
        threshold=threshold,
        radius=radius,
        n_bits=n_bits,
    )
    outside_count = int(output["outside_applicability_domain"].sum())
    print(
        "Added applicability-domain scores using "
        f"{selected_reference} at threshold {threshold:.2f}; "
        f"{outside_count} candidate(s) flagged outside domain."
    )
    return output


def canonicalize_solvent(solvent_smiles: str) -> str:
    """Canonicalize the screening solvent SMILES."""
    canonical = canonicalize_smiles(solvent_smiles)
    if canonical is None:
        raise ValueError(f"Invalid solvent SMILES: {solvent_smiles}")
    return canonical


def load_solvent_descriptors(path: Path) -> pd.DataFrame:
    """Load solvent descriptors and normalize key columns."""
    if not path.exists():
        raise FileNotFoundError(f"Solvent descriptor file not found: {path}")

    descriptors = pd.read_csv(path, low_memory=False)
    if "canonical_solvent_smiles" not in descriptors.columns:
        if "canonical_smiles" in descriptors.columns:
            descriptors["canonical_solvent_smiles"] = descriptors["canonical_smiles"]
        else:
            descriptors["canonical_solvent_smiles"] = pd.NA

    if "solvent_original" not in descriptors.columns:
        if "solvent" in descriptors.columns:
            descriptors["solvent_original"] = descriptors["solvent"]
        else:
            descriptors["solvent_original"] = pd.NA

    descriptors["canonical_solvent_smiles"] = descriptors[
        "canonical_solvent_smiles"
    ].apply(lambda value: str(value).strip() if pd.notna(value) else pd.NA)
    descriptors["solvent_original"] = descriptors["solvent_original"].apply(
        lambda value: str(value).strip() if pd.notna(value) else pd.NA
    )
    return descriptors


def get_solvent_descriptor_row(
    descriptors: pd.DataFrame,
    solvent_smiles: str,
    canonical_solvent_smiles: str,
) -> pd.Series | None:
    """Find a descriptor row by canonical solvent SMILES, then original label."""
    canonical_matches = descriptors[
        descriptors["canonical_solvent_smiles"] == canonical_solvent_smiles
    ]
    if not canonical_matches.empty:
        return canonical_matches.iloc[0]

    label_matches = descriptors[
        descriptors["solvent_original"].astype("string").str.lower()
        == solvent_smiles.strip().lower()
    ]
    if not label_matches.empty:
        return label_matches.iloc[0]
    return None


def build_solvent_descriptor_vector(
    descriptor_row: pd.Series | None,
    descriptor_columns: list[str],
    medians: dict[str, Any],
) -> np.ndarray:
    """Build one solvent descriptor vector using training medians for missing values."""
    values: list[float] = []
    for column in descriptor_columns:
        value = pd.NA
        if descriptor_row is not None and column in descriptor_row.index:
            value = descriptor_row[column]
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            numeric = medians.get(column, 0.0)
        if pd.isna(numeric):
            numeric = 0.0
        values.append(float(numeric))
    return np.asarray(values, dtype=np.float32)


def build_feature_matrix(
    fingerprints: np.ndarray,
    solvent_vector: np.ndarray,
) -> np.ndarray:
    """Append the same solvent descriptor vector to every fingerprint row."""
    solvent_matrix = np.tile(solvent_vector, (fingerprints.shape[0], 1))
    return np.hstack([fingerprints, solvent_matrix])


def predict_targets(
    candidates: pd.DataFrame,
    models: dict[str, Any],
    metadata: dict[str, Any],
    solvent_descriptor_row: pd.Series | None,
) -> pd.DataFrame:
    """Predict all available targets for candidates."""
    radius = int(metadata["fingerprint_radius"])
    n_bits = int(metadata["fingerprint_n_bits"])
    descriptor_columns = list(metadata["solvent_descriptor_columns_used"])
    medians_by_target = metadata.get("median_values_used_for_imputation", {})

    fingerprints = build_fingerprints(candidates["canonical_smiles"], radius, n_bits)
    output = candidates.copy()

    for target, model in models.items():
        medians = medians_by_target.get(target, {})
        solvent_vector = build_solvent_descriptor_vector(
            descriptor_row=solvent_descriptor_row,
            descriptor_columns=descriptor_columns,
            medians=medians,
        )
        features = build_feature_matrix(fingerprints, solvent_vector)
        output[PREDICTION_OUTPUT_COLUMNS[target]] = model.predict(features)

    return output


def rank_candidates(
    predictions: pd.DataFrame,
    solvent_smiles: str,
    target_emission: float,
) -> pd.DataFrame:
    """Compute ranking scores and output columns."""
    ranked = predictions.copy()
    ranked["solvent_smiles"] = solvent_smiles
    ranked["emission_error_from_target"] = (
        ranked["predicted_emission_nm"] - target_emission
    ).abs()
    ranked["score"] = (
        -ranked["emission_error_from_target"]
        + 200 * ranked["predicted_quantum_yield"]
    )

    if "predicted_log_extinction" in ranked.columns:
        ranked["estimated_brightness_score"] = (
            ranked["predicted_quantum_yield"]
            * np.power(10.0, ranked["predicted_log_extinction"])
        )
    else:
        ranked["predicted_log_extinction"] = pd.NA
        ranked["estimated_brightness_score"] = pd.NA

    preferred_metadata = [
        column for column in PREFERRED_METADATA_COLUMNS if column in ranked.columns
    ]
    remaining_metadata = [
        column
        for column in ranked.columns
        if column not in preferred_metadata
        and column not in CORE_OUTPUT_COLUMNS
        and column not in INTERNAL_TEMPORARY_COLUMNS
        and not column.startswith("_")
    ]
    columns = preferred_metadata + remaining_metadata + [
        column for column in CORE_OUTPUT_COLUMNS if column in ranked.columns
    ]
    columns = list(dict.fromkeys(columns))
    return ranked[columns].sort_values("score", ascending=False).reset_index(drop=True)


def main() -> int:
    """Run first-pass molecule screening."""
    args = parse_args()
    try:
        require_rdkit()
        metadata = load_json(args.model_dir / "feature_metadata.json")
        models = load_models(args.model_dir)
        candidates = load_candidates(args.candidates)

        canonical_solvent = canonicalize_solvent(args.solvent_smiles)
        solvent_descriptors = load_solvent_descriptors(args.solvent_descriptors)
        solvent_row = get_solvent_descriptor_row(
            descriptors=solvent_descriptors,
            solvent_smiles=args.solvent_smiles,
            canonical_solvent_smiles=canonical_solvent,
        )
        if solvent_row is None:
            print(
                "WARNING: solvent descriptors not found; using training medians "
                "for solvent descriptor features."
            )

        predictions = predict_targets(
            candidates=candidates,
            models=models,
            metadata=metadata,
            solvent_descriptor_row=solvent_row,
        )
        predictions = maybe_add_applicability_domain_columns(
            df=predictions,
            model_dir=args.model_dir,
            reference_csv=args.applicability_reference_csv,
            threshold=args.applicability_threshold,
            metadata=metadata,
            disabled=args.no_applicability_domain,
        )
        ranked = rank_candidates(
            predictions=predictions,
            solvent_smiles=canonical_solvent,
            target_emission=args.target_emission,
        )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(args.out, index=False)
        print(f"Screened {len(ranked)} candidate(s).")
        print(f"Saved ranked candidates to: {args.out}")
        print("Output columns: " + ", ".join(ranked.columns))
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
