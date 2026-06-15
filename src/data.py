from __future__ import annotations

import pandas as pd
from rdkit import Chem

from . import config


def canonicalize_smiles(smiles: object) -> str | None:
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_raw_data(path=None) -> pd.DataFrame:
    resolved_path = config.resolve_chemfluor_data_path(path)
    print(f"Loading dataset: {resolved_path}")
    return pd.read_csv(resolved_path, encoding="latin1")


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    required = [config.SMILES_COL, config.SOLVENT_COL, config.WAVELENGTH_COL, config.PLQY_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    stats: dict[str, int] = {"raw_rows": len(df)}
    print(f"Raw rows: {len(df)}")

    before = len(df)
    df = df.dropna(subset=required).copy()
    stats["rows_after_required_dropna"] = len(df)
    print(f"Dropped {before - len(df)} rows missing SMILES, solvent, wavelength, or PLQY.")

    df[config.WAVELENGTH_COL] = pd.to_numeric(df[config.WAVELENGTH_COL], errors="coerce")
    df[config.PLQY_COL] = pd.to_numeric(df[config.PLQY_COL], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[config.WAVELENGTH_COL, config.PLQY_COL])
    print(f"Dropped {before - len(df)} rows with non-numeric targets.")

    exact_duplicates = int(df.duplicated().sum())
    print(f"Exact duplicate rows found: {exact_duplicates}")

    df["canonical_smiles"] = df[config.SMILES_COL].map(canonicalize_smiles)
    invalid = int(df["canonical_smiles"].isna().sum())
    if invalid:
        print(f"Dropping invalid SMILES rows: {invalid}")
    df = df.dropna(subset=["canonical_smiles"]).copy()
    df[config.SOLVENT_COL] = df[config.SOLVENT_COL].astype(str).str.strip()

    pair_duplicates = int(df.duplicated(subset=["canonical_smiles", config.SOLVENT_COL]).sum())
    print(f"Duplicate canonical molecule-solvent rows found: {pair_duplicates}")
    before = len(df)
    grouped = (
        df.groupby(["canonical_smiles", config.SOLVENT_COL], as_index=False)
        .agg(
            {
                config.SMILES_COL: "first",
                config.WAVELENGTH_COL: "mean",
                config.PLQY_COL: "mean",
            }
        )
    )
    merged = before - len(grouped)
    print(f"Merged duplicate molecule-solvent rows by averaging targets: {merged}")

    grouped["SMILES"] = grouped["canonical_smiles"]
    stats.update(
        {
            "cleaned_rows": len(grouped),
            "invalid_smiles": invalid,
            "exact_duplicates": exact_duplicates,
            "merged_duplicate_pairs": merged,
            "unique_molecules": grouped["canonical_smiles"].nunique(),
            "unique_solvents": grouped[config.SOLVENT_COL].nunique(),
        }
    )
    print(f"Cleaned rows: {len(grouped)}")
    return grouped, stats
