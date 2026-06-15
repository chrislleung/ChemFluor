from __future__ import annotations

import pandas as pd

from chemfluor.data_standardization import (
    analyze_dataset_overlap,
    deduplicate_standardized_rows,
    molecule_solvent_replicates,
    red_region_counts,
)


def standardized_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chromophore_smiles": ["CCO", "CCO", "CCO", "CCN"],
            "solvent_original": ["O", "O", "O", "O"],
            "canonical_chromophore_smiles": ["CCO", "CCO", "CCO", "CCN"],
            "canonical_solvent_smiles": ["O", "O", "O", "O"],
            "absorption_nm": [350, 350, 355, 420],
            "emission_nm": [610, 610, 630, 540],
            "lifetime_ns": [pd.NA, pd.NA, pd.NA, pd.NA],
            "quantum_yield": [0.5, 0.5, 0.6, 0.3],
            "log_extinction": [4.0, 4.0, 4.1, 3.5],
            "source_dataset": ["FluoDB-Lite", "ChemFluor", "Deep4Chem", "FluoDB-Lite"],
        }
    )


def test_exact_duplicates_removed_with_source_preference() -> None:
    deduplicated = deduplicate_standardized_rows(
        standardized_rows(), prefer_sources=["ChemFluor", "Deep4Chem", "FluoDB-Lite"]
    )

    assert len(deduplicated) == 3
    exact_duplicate = deduplicated[
        (deduplicated["canonical_chromophore_smiles"] == "CCO")
        & (deduplicated["emission_nm"] == 610)
    ]
    assert len(exact_duplicate) == 1
    assert exact_duplicate.iloc[0]["source_dataset"] == "ChemFluor"


def test_molecule_solvent_replicates_are_reported_not_collapsed() -> None:
    rows = standardized_rows()
    deduplicated = deduplicate_standardized_rows(
        rows, prefer_sources=["ChemFluor", "Deep4Chem", "FluoDB-Lite"]
    )
    replicates = molecule_solvent_replicates(deduplicated)

    assert len(replicates) == 1
    assert replicates.loc[0, "canonical_chromophore_smiles"] == "CCO"
    assert replicates.loc[0, "unique_measurements"] == 2


def test_overlap_and_red_region_counts_are_correct() -> None:
    rows = standardized_rows()
    overlap = analyze_dataset_overlap(rows)
    red_counts = red_region_counts(rows)

    assert overlap["exact_duplicate_rows_removed"] == 1
    assert overlap["fluodb_exact_overlaps_with_chemfluor"] == 1
    assert red_counts["emission_ge_580"] == 3
    assert red_counts["emission_ge_600"] == 3
    assert red_counts["emission_ge_650"] == 0
