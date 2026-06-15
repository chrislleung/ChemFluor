"""Standardize ChemFluor and Deep4Chem data for combined training."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

import pandas as pd

try:
    from rdkit import Chem, RDLogger
except ImportError as exc:  # pragma: no cover - exercised only without RDKit installed.
    Chem = None
    RDLogger = None
    _RDKIT_IMPORT_ERROR = exc
else:
    _RDKIT_IMPORT_ERROR = None
    RDLogger.DisableLog("rdApp.*")


DEEP4CHEM_COLUMN_MAP = {
    "Chromophore": "chromophore_smiles",
    "Solvent": "solvent_original",
    "Absorption max (nm)": "absorption_nm",
    "Emission max (nm)": "emission_nm",
    "Lifetime (ns)": "lifetime_ns",
    "Quantum yield": "quantum_yield",
    "log(e/mol-1 dm3 cm-1)": "log_extinction",
}

STANDARD_COLUMNS = [
    "chromophore_smiles",
    "solvent_original",
    "canonical_chromophore_smiles",
    "canonical_solvent_smiles",
    "absorption_nm",
    "emission_nm",
    "lifetime_ns",
    "quantum_yield",
    "log_extinction",
    "source_dataset",
]

FLUODB_METADATA_COLUMNS = [
    "fluodb_source",
    "fluodb_tag",
    "fluodb_tag_name",
    "fluodb_solvent_num",
    "fluodb_split",
    "reference_doi",
]

TARGET_COLUMNS = [
    "absorption_nm",
    "emission_nm",
    "lifetime_ns",
    "quantum_yield",
    "log_extinction",
]

CHEMFLUOR_COLUMN_CANDIDATES = {
    "chromophore_smiles": ["SMILES", "smiles", "chromophore_smiles", "Chromophore"],
    "solvent_original": ["solvent", "Solvent", "solvent_original"],
    "absorption_nm": ["Absorption/nm", "absorption_nm", "Absorption max (nm)"],
    "emission_nm": ["Emission/nm", "emission_nm", "Emission max (nm)"],
    "quantum_yield": ["PLQY", "Quantum yield", "quantum_yield"],
}

FLUODB_COLUMN_MAP = {
    "smiles": "chromophore_smiles",
    "solvent": "solvent_original",
    "absorption/nm": "absorption_nm",
    "emission/nm": "emission_nm",
    "plqy": "quantum_yield",
    "reference(doi)": "reference_doi",
    "source": "fluodb_source",
    "tag": "fluodb_tag",
    "tag_name": "fluodb_tag_name",
    "solvent_num": "fluodb_solvent_num",
    "split": "fluodb_split",
}

SOURCE_PREFERENCE = ["chemfluor", "deep4chem", "FluoDB-Lite"]

CHEMFLUOR_KEYWORD_CANDIDATES = {
    "chromophore_smiles": ["smiles", "chromophore"],
    "solvent_original": ["solvent"],
    "absorption_nm": ["absorption"],
    "emission_nm": ["emission"],
    "quantum_yield": ["plqy", "quantum"],
}


def _require_rdkit() -> None:
    """Raise a helpful error if RDKit is not available."""
    if Chem is None:
        raise ImportError("RDKit is required for SMILES canonicalization.") from _RDKIT_IMPORT_ERROR


def canonicalize_smiles(smiles: object) -> str | None:
    """Return RDKit canonical SMILES, or None when parsing fails."""
    _require_rdkit()
    if pd.isna(smiles):
        return None

    text = str(smiles).strip()
    if not text:
        return None

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV path with common encoding fallbacks and clear errors."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    attempts = [
        {"encoding": "utf-8-sig", "low_memory": False},
        {"encoding": "utf-8", "low_memory": False},
        {"encoding": "cp1252", "low_memory": False},
        {"encoding": "latin1", "low_memory": False},
    ]
    errors: list[str] = []
    for kwargs in attempts:
        try:
            return pd.read_csv(csv_path, **kwargs)
        except UnicodeDecodeError as exc:
            errors.append(f"{kwargs}: {exc}")

    raise ValueError(
        f"Could not load CSV with supported encodings: {csv_path}\n"
        + "\n".join(errors)
    )


def _validate_columns(df: pd.DataFrame, required_columns: Iterable[str], label: str) -> None:
    """Validate that all required columns exist."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if not missing_columns:
        return

    available = "\n".join(f"  - {column}" for column in df.columns)
    missing = ", ".join(missing_columns)
    raise ValueError(f"{label} is missing column(s): {missing}\n\nAvailable columns:\n{available}")


def _coerce_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Convert standard target columns to numeric values."""
    standardized = df.copy()
    for column in TARGET_COLUMNS:
        if column not in standardized.columns:
            standardized[column] = pd.NA
        standardized[column] = pd.to_numeric(standardized[column], errors="coerce")
    return standardized


def _finalize_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the output contains the full standard column set in order."""
    finalized = df.copy()
    for column in STANDARD_COLUMNS:
        if column not in finalized.columns:
            finalized[column] = pd.NA
    return finalized[STANDARD_COLUMNS]


def _finalize_standard_and_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure standardized columns plus FluoDB metadata are present."""
    finalized = df.copy()
    for column in [*STANDARD_COLUMNS, *FLUODB_METADATA_COLUMNS]:
        if column not in finalized.columns:
            finalized[column] = pd.NA
    return finalized[[*STANDARD_COLUMNS, *FLUODB_METADATA_COLUMNS]]


def _infer_column(
    columns: pd.Index, output_name: str, required: bool = True
) -> str | None:
    """Infer a ChemFluor source column for a standardized output column."""
    for candidate in CHEMFLUOR_COLUMN_CANDIDATES.get(output_name, []):
        if candidate in columns:
            return candidate

    lowered = {str(column).lower(): column for column in columns}
    for keyword in CHEMFLUOR_KEYWORD_CANDIDATES.get(output_name, []):
        matches = [original for lower, original in lowered.items() if keyword in lower]
        if matches:
            return str(matches[0])

    if required:
        available = "\n".join(f"  - {column}" for column in columns)
        raise ValueError(
            f"Could not infer ChemFluor column for '{output_name}'.\n\n"
            f"Available columns:\n{available}"
        )
    return None


def load_deep4chem(path: str | Path) -> pd.DataFrame:
    """Load and standardize the Deep4Chem chromophore dataset."""
    df = _load_csv(path)
    _validate_columns(df, DEEP4CHEM_COLUMN_MAP.keys(), "Deep4Chem CSV")

    standardized = df.rename(columns=DEEP4CHEM_COLUMN_MAP)[
        list(DEEP4CHEM_COLUMN_MAP.values())
    ].copy()
    standardized = _coerce_targets(standardized)
    standardized["chromophore_smiles"] = standardized["chromophore_smiles"].astype(str).str.strip()
    standardized["solvent_original"] = standardized["solvent_original"].astype(str).str.strip()
    standardized["canonical_chromophore_smiles"] = standardized[
        "chromophore_smiles"
    ].map(canonicalize_smiles)
    standardized["canonical_solvent_smiles"] = standardized["solvent_original"].map(
        canonicalize_smiles
    )
    standardized["source_dataset"] = "deep4chem"
    return _finalize_standard_columns(standardized)


def load_chemfluor(path: str | Path) -> pd.DataFrame:
    """Load ChemFluor data and standardize likely molecule, solvent, and target columns."""
    df = _load_csv(path)
    column_map = {
        "chromophore_smiles": _infer_column(df.columns, "chromophore_smiles"),
        "solvent_original": _infer_column(df.columns, "solvent_original"),
        "absorption_nm": _infer_column(df.columns, "absorption_nm"),
        "emission_nm": _infer_column(df.columns, "emission_nm"),
        "quantum_yield": _infer_column(df.columns, "quantum_yield"),
    }

    standardized = pd.DataFrame()
    for output_name, input_name in column_map.items():
        assert input_name is not None
        standardized[output_name] = df[input_name]

    standardized["lifetime_ns"] = pd.NA
    standardized["log_extinction"] = pd.NA
    standardized = _coerce_targets(standardized)
    standardized["chromophore_smiles"] = standardized["chromophore_smiles"].astype(str).str.strip()
    standardized["solvent_original"] = standardized["solvent_original"].astype(str).str.strip()
    standardized["canonical_chromophore_smiles"] = standardized[
        "chromophore_smiles"
    ].map(canonicalize_smiles)
    standardized["canonical_solvent_smiles"] = standardized["solvent_original"].map(
        canonicalize_smiles
    )
    standardized["source_dataset"] = "chemfluor"
    return _finalize_standard_columns(standardized)


def load_fluodb_lite(path: str | Path) -> pd.DataFrame:
    """Load and standardize FluoDB-Lite into the shared ChemFluor schema."""
    df = _load_csv(path)
    required = ["smiles", "solvent"]
    _validate_columns(df, required, "FluoDB-Lite CSV")

    standardized = pd.DataFrame()
    for input_name, output_name in FLUODB_COLUMN_MAP.items():
        standardized[output_name] = df[input_name] if input_name in df.columns else pd.NA

    standardized["lifetime_ns"] = pd.NA
    standardized = _coerce_targets(standardized)
    extinction = pd.to_numeric(df.get("e/m-1cm-1", pd.NA), errors="coerce")
    standardized["log_extinction"] = np.where(
        extinction > 0, np.log10(extinction), np.nan
    )
    standardized["chromophore_smiles"] = standardized["chromophore_smiles"].astype(str).str.strip()
    standardized["solvent_original"] = standardized["solvent_original"].apply(
        lambda value: str(value).strip() if pd.notna(value) else pd.NA
    )
    standardized["canonical_chromophore_smiles"] = standardized[
        "chromophore_smiles"
    ].map(canonicalize_smiles)
    standardized["canonical_solvent_smiles"] = standardized["solvent_original"].map(
        canonicalize_smiles
    )
    standardized["source_dataset"] = "FluoDB-Lite"
    standardized = standardized.dropna(subset=["canonical_chromophore_smiles"]).copy()
    standardized = standardized.dropna(subset=TARGET_COLUMNS, how="all").copy()
    return _finalize_standard_and_metadata_columns(standardized).reset_index(drop=True)


def _target_key_value(value: object) -> str:
    """Represent numeric target values stably for duplicate keys."""
    if pd.isna(value):
        return "<NA>"
    return f"{float(value):.6g}"


def make_measurement_dedup_key(df: pd.DataFrame) -> pd.Series:
    """Build an exact-measurement duplicate key for standardized rows."""
    parts = pd.DataFrame(index=df.index)
    for column in ["canonical_chromophore_smiles", "canonical_solvent_smiles"]:
        parts[column] = df.get(column, pd.Series(pd.NA, index=df.index)).fillna("<NA>").astype(str)
    for column in ["absorption_nm", "emission_nm", "quantum_yield", "log_extinction"]:
        parts[column] = df.get(column, pd.Series(pd.NA, index=df.index)).map(_target_key_value)
    return parts.astype(str).agg("|".join, axis=1)


def _source_rank(source: object, prefer_sources: list[str]) -> int:
    source_text = str(source).lower()
    lowered = [item.lower() for item in prefer_sources]
    return lowered.index(source_text) if source_text in lowered else len(lowered)


def deduplicate_standardized_rows(
    df: pd.DataFrame, prefer_sources: list[str] | None = None
) -> pd.DataFrame:
    """Remove exact measurement duplicates while keeping source-preferred rows."""
    prefer_sources = prefer_sources or SOURCE_PREFERENCE
    working = df.copy()
    working["_dedup_key"] = make_measurement_dedup_key(working)
    working["_source_rank"] = working["source_dataset"].map(
        lambda source: _source_rank(source, prefer_sources)
    )
    working["_original_order"] = range(len(working))
    working = working.sort_values(
        ["_dedup_key", "_source_rank", "_original_order"], kind="mergesort"
    )
    deduplicated = working.drop_duplicates(subset=["_dedup_key"], keep="first")
    return deduplicated.sort_values("_original_order").drop(
        columns=["_dedup_key", "_source_rank", "_original_order"]
    ).reset_index(drop=True)


def molecule_solvent_replicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return molecule-solvent pairs with multiple non-identical measurements."""
    working = df.copy()
    working["_measurement_key"] = make_measurement_dedup_key(working)
    group_columns = ["canonical_chromophore_smiles", "canonical_solvent_smiles"]
    grouped = (
        working.groupby(group_columns, dropna=False)
        .agg(
            row_count=("source_dataset", "size"),
            unique_measurements=("_measurement_key", "nunique"),
            source_combination=("source_dataset", lambda values: ",".join(sorted(set(map(str, values))))),
            min_emission_nm=("emission_nm", "min"),
            max_emission_nm=("emission_nm", "max"),
        )
        .reset_index()
    )
    return grouped[
        (grouped["row_count"] > 1) & (grouped["unique_measurements"] > 1)
    ].reset_index(drop=True)


def red_region_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count emission coverage at red/orange/NIR thresholds."""
    emission = pd.to_numeric(df.get("emission_nm"), errors="coerce")
    return {
        f"emission_ge_{threshold}": int((emission >= threshold).sum())
        for threshold in [550, 580, 600, 650, 700, 750]
    }


def analyze_dataset_overlap(df: pd.DataFrame) -> dict:
    """Summarize exact duplicates, source overlap, replicates, and red coverage."""
    working = df.copy()
    working["_dedup_key"] = make_measurement_dedup_key(working)
    duplicate_rows_removed = int(working.duplicated("_dedup_key").sum())
    duplicate_groups = working[working.duplicated("_dedup_key", keep=False)]
    source_combinations = (
        duplicate_groups.groupby("_dedup_key")["source_dataset"]
        .apply(lambda values: ",".join(sorted(set(map(str, values)))))
        .value_counts()
        .to_dict()
    )
    fluodb_keys = set(working.loc[working["source_dataset"].astype(str).str.lower() == "fluodb-lite", "_dedup_key"])
    chemfluor_keys = set(working.loc[working["source_dataset"].astype(str).str.lower() == "chemfluor", "_dedup_key"])
    deep4chem_keys = set(working.loc[working["source_dataset"].astype(str).str.lower() == "deep4chem", "_dedup_key"])
    replicates = molecule_solvent_replicates(working.drop(columns=["_dedup_key"]))
    return {
        "rows_by_source": working["source_dataset"].astype(str).value_counts().to_dict(),
        "total_rows": int(len(working)),
        "unique_exact_measurements": int(working["_dedup_key"].nunique()),
        "exact_duplicate_rows_removed": duplicate_rows_removed,
        "molecule_solvent_pairs_with_multiple_measurements": int(len(replicates)),
        "duplicate_counts_by_source_combination": source_combinations,
        "fluodb_exact_overlaps_with_chemfluor": int(len(fluodb_keys & chemfluor_keys)),
        "fluodb_exact_overlaps_with_deep4chem": int(len(fluodb_keys & deep4chem_keys)),
        "red_region_counts": red_region_counts(working),
    }


def combine_training_data(
    deep4chem_path: str | Path, chemfluor_path: str | Path
) -> pd.DataFrame:
    """Load, combine, and lightly filter ChemFluor plus Deep4Chem training rows."""
    combined = pd.concat(
        [load_deep4chem(deep4chem_path), load_chemfluor(chemfluor_path)],
        ignore_index=True,
    )
    combined = combined.dropna(subset=["canonical_chromophore_smiles"]).copy()
    combined = combined.dropna(subset=TARGET_COLUMNS, how="all").copy()
    combined = combined.drop_duplicates().reset_index(drop=True)
    return combined[STANDARD_COLUMNS]
