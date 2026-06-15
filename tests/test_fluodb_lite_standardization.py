from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from chemfluor.data_standardization import load_fluodb_lite


def test_fluodb_lite_columns_map_and_values_canonicalize(tmp_path: Path) -> None:
    path = tmp_path / "FluoDB-Lite.csv"
    pd.DataFrame(
        {
            "absorption/nm": [400],
            "emission/nm": [610],
            "plqy": [0.5],
            "e/m-1cm-1": [10000],
            "smiles": ["C(C)O"],
            "solvent": ["CC#N"],
            "reference(doi)": ["10.example/test"],
            "source": ["Deep4Chem,ChemFluo"],
            "tag": [8],
            "tag_name": ["PAHs"],
            "solvent_num": [5],
            "split": ["train"],
        }
    ).to_csv(path, index=False)

    result = load_fluodb_lite(path)

    assert len(result) == 1
    assert result.loc[0, "chromophore_smiles"] == "C(C)O"
    assert result.loc[0, "canonical_chromophore_smiles"] == "CCO"
    assert result.loc[0, "solvent_original"] == "CC#N"
    assert result.loc[0, "canonical_solvent_smiles"] == "CC#N"
    assert result.loc[0, "absorption_nm"] == 400
    assert result.loc[0, "emission_nm"] == 610
    assert result.loc[0, "quantum_yield"] == 0.5
    assert math.isclose(result.loc[0, "log_extinction"], 4.0)
    assert result.loc[0, "source_dataset"] == "FluoDB-Lite"
    assert result.loc[0, "fluodb_source"] == "Deep4Chem,ChemFluo"
    assert result.loc[0, "reference_doi"] == "10.example/test"


def test_fluodb_lite_drops_invalid_chromophore_and_no_target_rows(tmp_path: Path) -> None:
    path = tmp_path / "FluoDB-Lite.csv"
    pd.DataFrame(
        {
            "absorption/nm": [pd.NA, pd.NA, pd.NA],
            "emission/nm": [500, pd.NA, pd.NA],
            "plqy": [pd.NA, pd.NA, pd.NA],
            "e/m-1cm-1": [pd.NA, pd.NA, pd.NA],
            "smiles": ["CCO", "not_a_smiles", "CCN"],
            "solvent": ["not_a_solvent", "CCO", pd.NA],
        }
    ).to_csv(path, index=False)

    result = load_fluodb_lite(path)

    assert len(result) == 1
    assert result.loc[0, "canonical_chromophore_smiles"] == "CCO"
    assert pd.isna(result.loc[0, "canonical_solvent_smiles"])
    assert result.loc[0, "emission_nm"] == 500
