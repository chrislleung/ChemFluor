"""Utilities for combined ChemFluor model prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from chemfluor.data_standardization import TARGET_COLUMNS, canonicalize_smiles

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


DEFAULT_APPLICABILITY_REFERENCE = "combined_modeling_rows_after_feature_merge.csv"
DEFAULT_RADIUS = 2
DEFAULT_N_BITS = 2048
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
REFERENCE_SMILES_COLUMNS = [
    "canonical_chromophore_smiles",
    "chromophore_smiles",
    "canonical_smiles",
    "smiles",
]
CONFIDENCE_INTERPRETATIONS = {
    "high": "Candidate is close to known training/reference molecules.",
    "medium": (
        "Candidate is moderately similar; prediction may be useful but should be "
        "treated cautiously."
    ),
    "low-medium": (
        "Candidate is only weakly similar; prediction may be extrapolative."
    ),
    "low": "Candidate is outside or near the edge of the model domain.",
    "unknown": "Applicability-domain similarity could not be computed.",
}
REFERENCE_MATCH_COLUMNS = [
    "canonical_chromophore_smiles",
    "canonical_solvent_smiles",
    "absorption_nm",
    "emission_nm",
    "lifetime_ns",
    "quantum_yield",
    "log_extinction",
    "source_dataset",
    "fluodb_source",
    "tag_name",
    "fluodb_tag_name",
]


def require_rdkit() -> None:
    """Raise a helpful error if RDKit is unavailable."""
    if Chem is None or DataStructs is None or AllChem is None:
        raise ImportError("RDKit is required for combined-model prediction.") from _RDKIT_IMPORT_ERROR


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def canonicalize_required(smiles: str, label: str) -> str:
    """Canonicalize a required SMILES string or raise a helpful error."""
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        raise ValueError(f"Invalid {label} SMILES: {smiles}")
    return canonical


def find_smiles_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    """Find a SMILES column using a priority-ordered list of candidate names."""
    normalized_columns = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        column = normalized_columns.get(candidate.lower())
        if column is not None:
            return column
    raise ValueError(
        "CSV must contain one of these SMILES columns: " + ", ".join(candidates)
    )


def load_solvent_descriptors(path: Path) -> pd.DataFrame:
    """Load solvent descriptors and normalize key columns."""
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
    ].apply(lambda value: str(value).strip() if pd.notna(value) else pd.NA)
    descriptors["solvent_original"] = descriptors["solvent_original"].apply(
        lambda value: str(value).strip() if pd.notna(value) else pd.NA
    )
    return descriptors


def choose_solvent_descriptor_columns(descriptors: pd.DataFrame) -> list[str]:
    """Select numeric solvent descriptor columns using the trainer's exclusions."""
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


def build_single_feature_matrix(
    canonical_smiles: str,
    solvent_descriptor_row: pd.Series | None,
    descriptor_columns: list[str],
    medians: dict[str, Any],
    radius: int,
    n_bits: int,
) -> np.ndarray:
    """Build the one-row combined feature matrix used by trained models."""
    fingerprint = morgan_fingerprint(canonical_smiles, radius=radius, n_bits=n_bits)
    if fingerprint is None:
        raise ValueError(f"Could not fingerprint canonical SMILES: {canonical_smiles}")
    solvent_vector = build_solvent_descriptor_vector(
        descriptor_row=solvent_descriptor_row,
        descriptor_columns=descriptor_columns,
        medians=medians,
    )
    return np.hstack([fingerprint, solvent_vector]).reshape(1, -1)


def load_or_infer_feature_metadata(
    model_dir: Path,
    solvent_descriptors: pd.DataFrame,
    model_feature_count: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Load feature metadata, or infer enough of it from saved modeling rows."""
    metadata_path = model_dir / "feature_metadata.json"
    warnings: list[str] = []
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        return metadata, warnings

    descriptor_columns = choose_solvent_descriptor_columns(solvent_descriptors)
    n_bits = DEFAULT_N_BITS
    if model_feature_count is not None:
        inferred_n_bits = model_feature_count - len(descriptor_columns)
        if inferred_n_bits > 0:
            n_bits = int(inferred_n_bits)

    metadata = {
        "fingerprint_radius": DEFAULT_RADIUS,
        "fingerprint_n_bits": n_bits,
        "solvent_descriptor_columns_used": descriptor_columns,
        "target_columns": TARGET_COLUMNS,
        "model_type": "rf",
        "median_values_used_for_imputation": {},
    }
    warnings.append(
        f"feature_metadata.json not found in {model_dir}; inferred descriptor "
        "columns, fingerprint size, and medians where possible."
    )

    modeling_path = model_dir / DEFAULT_APPLICABILITY_REFERENCE
    if not modeling_path.exists():
        warnings.append(
            "combined_modeling_rows_after_feature_merge.csv not found; missing "
            "solvent descriptors will be imputed with 0.0."
        )
        return metadata, warnings

    modeling_rows = pd.read_csv(modeling_path, low_memory=False)
    medians_by_target: dict[str, dict[str, float | None]] = {}
    for target in TARGET_COLUMNS:
        if target in modeling_rows.columns:
            target_rows = modeling_rows[pd.to_numeric(modeling_rows[target], errors="coerce").notna()]
        else:
            target_rows = modeling_rows
        available_columns = [
            column for column in descriptor_columns if column in target_rows.columns
        ]
        medians = target_rows[available_columns].apply(
            pd.to_numeric, errors="coerce"
        ).median(numeric_only=True)
        medians_by_target[target] = {
            key: (None if pd.isna(value) else float(value))
            for key, value in medians.items()
        }
    metadata["median_values_used_for_imputation"] = medians_by_target
    return metadata, warnings


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
    candidate_smiles: str,
    reference_fps: Sequence[Any],
    reference_smiles: Sequence[str],
    radius: int,
    n_bits: int,
) -> tuple[float, str]:
    """Find the nearest reference chromophore by Morgan Tanimoto similarity."""
    candidate_fp = morgan_bitvect(candidate_smiles, radius=radius, n_bits=n_bits)
    if candidate_fp is None:
        return float("nan"), ""

    similarities = DataStructs.BulkTanimotoSimilarity(candidate_fp, list(reference_fps))
    if not similarities:
        return float("nan"), ""

    best_index = max(range(len(similarities)), key=lambda index: similarities[index])
    return float(similarities[best_index]), reference_smiles[best_index]


def similarity_confidence_label(similarity: float | None) -> str:
    """Map nearest-reference similarity to a qualitative confidence label."""
    if similarity is None or pd.isna(similarity):
        return "unknown"
    if similarity >= 0.70:
        return "high"
    if similarity >= 0.50:
        return "medium"
    if similarity >= 0.35:
        return "low-medium"
    return "low"


def similarity_confidence_interpretation(label: str) -> str:
    """Return a short human-readable interpretation for a confidence label."""
    return CONFIDENCE_INTERPRETATIONS.get(label, CONFIDENCE_INTERPRETATIONS["unknown"])


def applicability_domain_payload(
    canonical_smiles: str,
    model_dir: Path,
    threshold: float,
    radius: int,
    n_bits: int,
    disabled: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Build JSON-ready applicability-domain output for one candidate."""
    if disabled:
        label = "unknown"
        return (
            {
                "confidence_label": label,
                "confidence_interpretation": similarity_confidence_interpretation(label),
            },
            ["Applicability-domain scoring disabled."],
        )
    if threshold < 0 or threshold > 1:
        raise ValueError(
            f"Applicability threshold must be between 0 and 1, got {threshold}."
        )

    reference_csv = model_dir / DEFAULT_APPLICABILITY_REFERENCE
    if not reference_csv.exists():
        label = "unknown"
        return (
            {
                "confidence_label": label,
                "confidence_interpretation": similarity_confidence_interpretation(label),
            },
            [
                "No applicability reference CSV found; continuing without "
                "applicability-domain scoring."
            ],
        )

    reference_fps, reference_smiles = load_reference_fingerprints(
        reference_csv=reference_csv,
        radius=radius,
        n_bits=n_bits,
    )
    similarity, nearest_smiles = compute_nearest_training_similarity(
        candidate_smiles=canonical_smiles,
        reference_fps=reference_fps,
        reference_smiles=reference_smiles,
        radius=radius,
        n_bits=n_bits,
    )
    outside_domain = bool(pd.isna(similarity) or similarity < threshold)
    confidence_label = similarity_confidence_label(similarity)
    return (
        {
            "nearest_training_similarity": similarity,
            "nearest_training_smiles": nearest_smiles,
            "outside_applicability_domain": outside_domain,
            "threshold": float(threshold),
            "confidence_label": confidence_label,
            "confidence_interpretation": similarity_confidence_interpretation(
                confidence_label
            ),
        },
        [],
    )


def exact_reference_matches_payload(
    canonical_smiles: str,
    model_dir: Path,
    max_rows: int = 20,
) -> dict[str, Any]:
    """Return exact canonical chromophore matches from saved reference rows."""
    reference_csv = model_dir / DEFAULT_APPLICABILITY_REFERENCE
    if not reference_csv.exists():
        return {"count": 0, "rows": []}

    reference_rows = pd.read_csv(reference_csv, low_memory=False)
    smiles_column = find_smiles_column(reference_rows, REFERENCE_SMILES_COLUMNS)
    working = reference_rows.copy()
    working["_canonical_match_smiles"] = working[smiles_column].map(canonicalize_smiles)
    matches = working[working["_canonical_match_smiles"] == canonical_smiles].copy()
    if matches.empty:
        return {"count": 0, "rows": []}

    output_columns = [
        column for column in REFERENCE_MATCH_COLUMNS if column in matches.columns
    ]
    if smiles_column not in output_columns:
        output_columns.insert(0, smiles_column)
    rows = matches[output_columns].head(max_rows).replace({np.nan: None}).to_dict(
        orient="records"
    )
    return {"count": int(len(matches)), "rows": rows}
