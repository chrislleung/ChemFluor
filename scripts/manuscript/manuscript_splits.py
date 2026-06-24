"""Leakage-safe data splits for manuscript experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass(frozen=True)
class SplitResult:
    """Indices and leakage diagnostics for one train/test split."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    group_column: str | None
    train_groups: int
    test_groups: int
    overlapping_groups: int


def bemis_murcko_scaffold(smiles: object) -> str:
    """Return a canonical Bemis-Murcko scaffold, including an acyclic marker."""
    if pd.isna(smiles):
        raise ValueError("Cannot generate a scaffold from a missing SMILES value.")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Cannot generate a scaffold from invalid SMILES: {smiles}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=False
    )
    return scaffold or "<ACYCLIC>"


def add_scaffold_column(
    rows: pd.DataFrame,
    smiles_column: str = "canonical_chromophore_smiles",
) -> pd.DataFrame:
    """Return a copy with a ``bemis_murcko_scaffold`` column."""
    if smiles_column not in rows.columns:
        raise ValueError(f"Missing SMILES column: {smiles_column}")
    result = rows.copy()
    result["bemis_murcko_scaffold"] = result[smiles_column].map(
        bemis_murcko_scaffold
    )
    return result


def make_split(
    rows: pd.DataFrame,
    split_name: str,
    test_size: float,
    seed: int,
) -> SplitResult:
    """Create a random, molecule-grouped, or scaffold-grouped split."""
    if len(rows) < 2:
        raise ValueError("At least two rows are required for a train/test split.")
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")

    positions = np.arange(len(rows))
    if split_name == "random":
        train_idx, test_idx = train_test_split(
            positions, test_size=test_size, random_state=seed
        )
        return SplitResult(
            np.asarray(train_idx),
            np.asarray(test_idx),
            None,
            0,
            0,
            0,
        )

    if split_name == "molecule":
        group_column = "canonical_chromophore_smiles"
    elif split_name == "scaffold":
        group_column = "bemis_murcko_scaffold"
    else:
        raise ValueError(f"Unknown split: {split_name}")

    if group_column not in rows.columns:
        raise ValueError(f"Missing split group column: {group_column}")
    groups = rows[group_column].astype(str).to_numpy()
    if pd.Series(groups).nunique() < 2:
        raise ValueError(f"{split_name} split requires at least two unique groups.")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train_idx, test_idx = next(splitter.split(positions, groups=groups))
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(
            f"{split_name} leakage detected: {len(overlap)} overlapping groups."
        )
    return SplitResult(
        np.asarray(train_idx),
        np.asarray(test_idx),
        group_column,
        len(train_groups),
        len(test_groups),
        len(overlap),
    )
