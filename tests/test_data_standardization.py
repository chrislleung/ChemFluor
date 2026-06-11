from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.data_standardization import (  # noqa: E402
    canonicalize_smiles,
    combine_training_data,
    load_deep4chem,
)


def test_canonicalize_smiles_valid() -> None:
    assert canonicalize_smiles("CCO") == "CCO"


def test_canonicalize_smiles_invalid_environment_label() -> None:
    assert canonicalize_smiles("gas") is None


def test_load_deep4chem_tiny_csv(tmp_path: Path) -> None:
    deep4chem_path = tmp_path / "deep4chem.csv"
    pd.DataFrame(
        {
            "Chromophore": ["CCO"],
            "Solvent": ["O"],
            "Absorption max (nm)": ["350"],
            "Emission max (nm)": ["420"],
            "Lifetime (ns)": ["1.5"],
            "Quantum yield": ["0.7"],
            "log(e/mol-1 dm3 cm-1)": ["4.2"],
        }
    ).to_csv(deep4chem_path, index=False)

    result = load_deep4chem(deep4chem_path)

    assert len(result) == 1
    assert result.loc[0, "chromophore_smiles"] == "CCO"
    assert result.loc[0, "solvent_original"] == "O"
    assert result.loc[0, "canonical_chromophore_smiles"] == "CCO"
    assert result.loc[0, "canonical_solvent_smiles"] == "O"
    assert result.loc[0, "absorption_nm"] == 350
    assert result.loc[0, "source_dataset"] == "deep4chem"


def test_combine_training_data_drops_invalid_chromophore_smiles(tmp_path: Path) -> None:
    deep4chem_path = tmp_path / "deep4chem.csv"
    chemfluor_path = tmp_path / "chemfluor.csv"

    pd.DataFrame(
        {
            "Chromophore": ["CCO", "not_a_smiles"],
            "Solvent": ["O", "O"],
            "Absorption max (nm)": [350, 360],
            "Emission max (nm)": [420, 430],
            "Lifetime (ns)": [1.5, 1.6],
            "Quantum yield": [0.7, 0.8],
            "log(e/mol-1 dm3 cm-1)": [4.2, 4.3],
        }
    ).to_csv(deep4chem_path, index=False)
    pd.DataFrame(
        {
            "SMILES": ["CCN"],
            "solvent": ["DMSO"],
            "Absorption/nm": [390],
            "Emission/nm": [460],
            "PLQY": [0.4],
        }
    ).to_csv(chemfluor_path, index=False)

    result = combine_training_data(deep4chem_path, chemfluor_path)

    assert len(result) == 2
    assert "not_a_smiles" not in set(result["chromophore_smiles"])
    assert result["canonical_chromophore_smiles"].isna().sum() == 0
    assert set(result["source_dataset"]) == {"deep4chem", "chemfluor"}
