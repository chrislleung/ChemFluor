"""Tests for manuscript-specific split behavior."""

from __future__ import annotations

import pandas as pd

from scripts.manuscript.manuscript_splits import (
    add_scaffold_column,
    bemis_murcko_scaffold,
    make_split,
)


def sample_rows() -> pd.DataFrame:
    """Create repeated molecules spanning several scaffolds."""
    smiles = [
        "c1ccccc1",
        "c1ccccc1",
        "Cc1ccccc1",
        "Cc1ccccc1",
        "c1ccncc1",
        "c1ccncc1",
        "C1CCCCC1",
        "C1CCCCC1",
        "CCO",
        "CCO",
        "CCN",
        "CCN",
    ]
    return add_scaffold_column(
        pd.DataFrame({"canonical_chromophore_smiles": smiles})
    )


def test_scaffold_generation() -> None:
    assert bemis_murcko_scaffold("Cc1ccccc1") == "c1ccccc1"
    assert bemis_murcko_scaffold("CCO") == "<ACYCLIC>"


def test_molecule_split_has_no_molecule_leakage() -> None:
    rows = sample_rows()
    result = make_split(rows, "molecule", test_size=0.3, seed=0)
    train = set(rows.iloc[result.train_indices]["canonical_chromophore_smiles"])
    test = set(rows.iloc[result.test_indices]["canonical_chromophore_smiles"])
    assert train.isdisjoint(test)
    assert result.overlapping_groups == 0


def test_scaffold_split_has_no_scaffold_leakage() -> None:
    rows = sample_rows()
    result = make_split(rows, "scaffold", test_size=0.3, seed=0)
    train = set(rows.iloc[result.train_indices]["bemis_murcko_scaffold"])
    test = set(rows.iloc[result.test_indices]["bemis_murcko_scaffold"])
    assert train.isdisjoint(test)
    assert result.overlapping_groups == 0
