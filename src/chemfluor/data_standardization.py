"""Standardize ChemFluor and Deep4Chem data for combined training."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

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
