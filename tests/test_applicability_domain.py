from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "screen_candidate_molecules.py"

spec = importlib.util.spec_from_file_location("screen_candidate_molecules", SCRIPT_PATH)
screening = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(screening)


def test_identical_candidate_gets_similarity_one(tmp_path: Path) -> None:
    reference_csv = tmp_path / "reference.csv"
    pd.DataFrame({"canonical_chromophore_smiles": ["c1ccccc1"]}).to_csv(
        reference_csv, index=False
    )
    candidates = pd.DataFrame({"canonical_smiles": ["c1ccccc1"]})

    scored = screening.add_applicability_domain_columns(
        candidates, reference_csv=reference_csv, threshold=0.30
    )

    assert math.isclose(scored.loc[0, "nearest_training_similarity"], 1.0)
    assert scored.loc[0, "nearest_training_smiles"] == "c1ccccc1"
    assert bool(scored.loc[0, "outside_applicability_domain"]) is False


def test_dissimilar_candidate_is_flagged_below_threshold(tmp_path: Path) -> None:
    reference_csv = tmp_path / "reference.csv"
    pd.DataFrame({"canonical_chromophore_smiles": ["c1ccccc1"]}).to_csv(
        reference_csv, index=False
    )
    candidates = pd.DataFrame({"canonical_smiles": ["CCCCCCCC"]})

    scored = screening.add_applicability_domain_columns(
        candidates, reference_csv=reference_csv, threshold=0.90
    )

    assert scored.loc[0, "nearest_training_similarity"] < 0.90
    assert bool(scored.loc[0, "outside_applicability_domain"]) is True


def test_missing_reference_csv_does_not_crash_optional_scoring(tmp_path: Path) -> None:
    candidates = pd.DataFrame({"canonical_smiles": ["CCO"]})
    missing_reference = tmp_path / "missing.csv"

    scored = screening.maybe_add_applicability_domain_columns(
        df=candidates,
        model_dir=tmp_path,
        reference_csv=missing_reference,
        threshold=0.30,
        metadata={"fingerprint_radius": 2, "fingerprint_n_bits": 2048},
        disabled=False,
    )

    assert scored.equals(candidates)


def test_smiles_column_fallback_logic(tmp_path: Path) -> None:
    reference_csv = tmp_path / "reference.csv"
    pd.DataFrame({"chromophore_smiles": ["CCO"]}).to_csv(reference_csv, index=False)
    candidates = pd.DataFrame({"smiles": ["CCO"]})

    scored = screening.add_applicability_domain_columns(
        candidates, reference_csv=reference_csv, threshold=0.30
    )

    assert math.isclose(scored.loc[0, "nearest_training_similarity"], 1.0)
    assert scored.loc[0, "nearest_training_smiles"] == "CCO"


def test_invalid_candidate_smiles_are_flagged(tmp_path: Path) -> None:
    reference_csv = tmp_path / "reference.csv"
    pd.DataFrame({"canonical_chromophore_smiles": ["CCO"]}).to_csv(
        reference_csv, index=False
    )
    candidates = pd.DataFrame({"canonical_smiles": ["not_a_smiles"]})

    scored = screening.add_applicability_domain_columns(
        candidates, reference_csv=reference_csv, threshold=0.30
    )

    assert math.isnan(scored.loc[0, "nearest_training_similarity"])
    assert scored.loc[0, "nearest_training_smiles"] == ""
    assert bool(scored.loc[0, "outside_applicability_domain"]) is True
