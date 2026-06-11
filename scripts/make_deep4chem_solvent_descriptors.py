"""Build an expanded solvent descriptor table for the Deep4Chem dataset.

Run from the project root:
    python scripts/make_deep4chem_solvent_descriptors.py `
      --deep4chem "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv" `
      --existing-solvents data/solvent_descriptors.csv `
      --output data/solvent_descriptors_expanded_deep4chem.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SOLVENT_COLUMN = "Solvent"
DEFAULT_REPORT_PATH = Path("outputs/deep4chem_analysis/solvent_descriptor_report.txt")

NON_MOLECULAR_LABELS = {
    "gas",
    "solid",
    "film",
    "neat",
    "vacuum",
    "air",
}

DESCRIPTOR_COLUMNS = [
    "molecular_weight",
    "exact_molecular_weight",
    "logp",
    "molar_refractivity",
    "tpsa",
    "num_hbond_donors",
    "num_hbond_acceptors",
    "num_rotatable_bonds",
    "ring_count",
    "aromatic_ring_count",
    "heavy_atom_count",
    "num_heteroatoms",
    "fraction_csp3",
    "formal_charge",
    "num_valence_electrons",
]

KNOWN_SOLVENT_ALIASES = {
    "1,2-propanediol": "CC(O)CO",
    "1-butanol": "CCCCO",
    "1-decanol": "CCCCCCCCCCO",
    "1-hexanol": "CCCCCCO",
    "1-methyl-2-pyrrolidinone": "CN1CCCC1=O",
    "1-octanol": "CCCCCCCCO",
    "1-propanol": "CCCO",
    "2-methyl-2-propanol": "CC(C)(C)O",
    "2-methylbutane": "CCC(C)C",
    "2-pentanone": "CCCC(C)=O",
    "2-propanol": "CC(C)O",
    "acetone": "CC(C)=O",
    "benzene": "c1ccccc1",
    "bromobenzene": "Brc1ccccc1",
    "butyl acetate": "CCCCOC(C)=O",
    "ccl4": "ClC(Cl)(Cl)Cl",
    "ch2cl2": "ClCCl",
    "chcl3": "ClC(Cl)Cl",
    "chlorobenzene": "Clc1ccccc1",
    "cyclohexane": "C1CCCCC1",
    "di-n-butyl ether": "CCCCOCCCC",
    "diethyl ether": "CCOCC",
    "diisopropyl ether": "CC(C)OC(C)C",
    "dioxane": "C1COCCO1",
    "dma": "CN(C)C(C)=O",
    "dmf": "CN(C)C=O",
    "dmso": "CS(C)=O",
    "ethyl acetate": "CCOC(C)=O",
    "ethylene glycol": "OCCO",
    "etoh": "CCO",
    "formamide": "NC=O",
    "glycerol": "OCC(O)CO",
    "h2o": "O",
    "heptane": "CCCCCCC",
    "hexane": "CCCCCC",
    "me-thf": "CC1CCCO1",
    "mecn": "CC#N",
    "meoh": "CO",
    "methyl acetate": "COC(C)=O",
    "methyl formate": "COC=O",
    "methylcyclohexane": "CC1CCCCC1",
    "n-methylformamide": "CNC=O",
    "o-dimethoxybenzene": "COc1ccccc1OC",
    "pyridine": "c1ccncc1",
    "tert-pentanol": "CCC(C)(C)O",
    "tfe": "OCC(F)(F)F",
    "thf": "C1CCOC1",
    "toluene": "Cc1ccccc1",
    "triethylamine": "CCN(CC)CC",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create expanded Deep4Chem solvent descriptors with RDKit features."
    )
    parser.add_argument(
        "--deep4chem",
        required=True,
        type=Path,
        help="Path to the raw Deep4Chem CSV.",
    )
    parser.add_argument(
        "--existing-solvents",
        required=True,
        type=Path,
        help="Path to the existing physical solvent descriptor CSV.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for the expanded solvent descriptor CSV.",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT_PATH,
        type=Path,
        help=f"Path for the text report. Defaults to {DEFAULT_REPORT_PATH}.",
    )
    return parser.parse_args()


def load_csv_robustly(csv_path: Path) -> pd.DataFrame:
    """Load a CSV with common encoding and delimiter fallbacks."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {csv_path}")

    attempts = [
        {"encoding": "utf-8-sig", "low_memory": False},
        {"encoding": "utf-8", "low_memory": False},
        {"encoding": "cp1252", "low_memory": False},
        {"encoding": "latin1", "low_memory": False},
        {"encoding": "utf-8-sig", "sep": None, "engine": "python"},
        {"encoding": "cp1252", "sep": None, "engine": "python"},
    ]

    errors: list[str] = []
    for kwargs in attempts:
        try:
            return pd.read_csv(csv_path, **kwargs)
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            errors.append(f"{kwargs}: {exc}")

    raise ValueError(
        "Could not load CSV after several attempts:\n" + "\n".join(errors)
    )


def validate_columns(df: pd.DataFrame, required_columns: list[str], label: str) -> None:
    """Raise a helpful error if required columns are missing."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if not missing_columns:
        return

    available = "\n".join(f"  - {column}" for column in df.columns)
    missing = ", ".join(missing_columns)
    raise ValueError(f"{label} is missing column(s): {missing}\n\nAvailable columns:\n{available}")


def import_rdkit() -> dict[str, Any]:
    """Import RDKit modules and silence parser logging."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to build solvent descriptors. Install rdkit and rerun."
        ) from exc

    RDLogger.DisableLog("rdApp.*")
    return {
        "Chem": Chem,
        "Crippen": Crippen,
        "Descriptors": Descriptors,
        "Lipinski": Lipinski,
        "rdMolDescriptors": rdMolDescriptors,
    }


def normalize_label(value: object) -> str:
    """Normalize a solvent label for matching."""
    return str(value).strip()


def normalize_key(value: object) -> str:
    """Normalize a solvent label for case-insensitive lookup."""
    return normalize_label(value).lower()


def canonicalize_smiles(smiles: str, chem: Any) -> str | None:
    """Return canonical SMILES if RDKit can parse the string."""
    text = normalize_label(smiles)
    if not text or normalize_key(text) in NON_MOLECULAR_LABELS:
        return None

    mol = chem.MolFromSmiles(text)
    if mol is None:
        return None
    return chem.MolToSmiles(mol, canonical=True)


def compute_rdkit_descriptors(mol: Any, rdkit: dict[str, Any]) -> dict[str, float | int]:
    """Compute the RDKit solvent descriptor set requested for valid molecules."""
    descriptors = rdkit["Descriptors"]
    crippen = rdkit["Crippen"]
    lipinski = rdkit["Lipinski"]
    rd_mol_descriptors = rdkit["rdMolDescriptors"]

    return {
        "molecular_weight": descriptors.MolWt(mol),
        "exact_molecular_weight": descriptors.ExactMolWt(mol),
        "logp": crippen.MolLogP(mol),
        "molar_refractivity": crippen.MolMR(mol),
        "tpsa": rd_mol_descriptors.CalcTPSA(mol),
        "num_hbond_donors": lipinski.NumHDonors(mol),
        "num_hbond_acceptors": lipinski.NumHAcceptors(mol),
        "num_rotatable_bonds": lipinski.NumRotatableBonds(mol),
        "ring_count": lipinski.RingCount(mol),
        "aromatic_ring_count": rd_mol_descriptors.CalcNumAromaticRings(mol),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "num_heteroatoms": lipinski.NumHeteroatoms(mol),
        "fraction_csp3": rd_mol_descriptors.CalcFractionCSP3(mol),
        "formal_charge": sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        "num_valence_electrons": descriptors.NumValenceElectrons(mol),
    }


def extract_unique_solvents(deep4chem_df: pd.DataFrame) -> pd.DataFrame:
    """Build one row per unique non-empty Deep4Chem solvent value."""
    solvents = deep4chem_df[SOLVENT_COLUMN].dropna().astype(str).str.strip()
    solvents = solvents[solvents.ne("")]
    return (
        solvents.value_counts()
        .rename_axis("solvent_original")
        .reset_index(name="deep4chem_row_count")
    )


def build_deep4chem_descriptor_rows(
    solvent_counts: pd.DataFrame, rdkit: dict[str, Any]
) -> pd.DataFrame:
    """Create RDKit descriptor rows for unique Deep4Chem solvent values."""
    chem = rdkit["Chem"]
    records: list[dict[str, Any]] = []

    for row in solvent_counts.itertuples(index=False):
        solvent_original = normalize_label(row.solvent_original)
        is_environment_label = normalize_key(solvent_original) in NON_MOLECULAR_LABELS
        canonical_smiles = None
        is_valid_rdkit = False
        descriptor_values = {column: pd.NA for column in DESCRIPTOR_COLUMNS}

        if not is_environment_label:
            mol = chem.MolFromSmiles(solvent_original)
            if mol is not None:
                is_valid_rdkit = True
                canonical_smiles = chem.MolToSmiles(mol, canonical=True)
                descriptor_values = compute_rdkit_descriptors(mol, rdkit)

        records.append(
            {
                "solvent_original": solvent_original,
                "canonical_solvent_smiles": canonical_smiles,
                "is_valid_rdkit": is_valid_rdkit,
                "is_environment_label": is_environment_label,
                "deep4chem_row_count": int(row.deep4chem_row_count),
                **descriptor_values,
            }
        )

    return pd.DataFrame.from_records(records)


def prepare_existing_solvent_descriptors(
    existing_df: pd.DataFrame, rdkit: dict[str, Any]
) -> pd.DataFrame:
    """Prepare existing physical solvent descriptors for direct and SMILES matching."""
    validate_columns(existing_df, ["solvent"], "Existing solvent descriptor CSV")

    chem = rdkit["Chem"]
    prepared = existing_df.copy()
    prepared["existing_solvent_match"] = prepared["solvent"].astype(str).str.strip()
    prepared["existing_solvent_key"] = prepared["existing_solvent_match"].str.lower()

    canonical_values: list[str | None] = []
    for key, original in zip(
        prepared["existing_solvent_key"], prepared["existing_solvent_match"]
    ):
        alias_smiles = KNOWN_SOLVENT_ALIASES.get(key)
        canonical_values.append(
            canonicalize_smiles(alias_smiles or original, chem)
        )
    prepared["existing_canonical_solvent_smiles"] = canonical_values

    return prepared.drop_duplicates(
        subset=["existing_solvent_key", "existing_canonical_solvent_smiles"],
        keep="first",
    )


def merge_existing_descriptors(
    deep4chem_descriptors: pd.DataFrame, existing_descriptors: pd.DataFrame
) -> pd.DataFrame:
    """Left-merge existing physical descriptors without dropping Deep4Chem solvents."""
    physical_columns = [
        column
        for column in existing_descriptors.columns
        if column not in {"solvent", "existing_solvent_key", "existing_canonical_solvent_smiles"}
    ]

    direct_existing = existing_descriptors.drop_duplicates(
        subset=["existing_solvent_key"], keep="first"
    )[["existing_solvent_key", *physical_columns]]
    canonical_existing = existing_descriptors.dropna(
        subset=["existing_canonical_solvent_smiles"]
    ).drop_duplicates(subset=["existing_canonical_solvent_smiles"], keep="first")[
        ["existing_canonical_solvent_smiles", *physical_columns]
    ]

    merged = deep4chem_descriptors.copy()
    merged["solvent_key"] = merged["solvent_original"].str.lower()
    merged = merged.merge(
        direct_existing,
        how="left",
        left_on="solvent_key",
        right_on="existing_solvent_key",
    ).drop(columns=["existing_solvent_key", "solvent_key"])

    missing_existing_match = merged["existing_solvent_match"].isna()
    canonical_matches = merged.loc[missing_existing_match].merge(
        canonical_existing,
        how="left",
        left_on="canonical_solvent_smiles",
        right_on="existing_canonical_solvent_smiles",
        suffixes=("", "_canonical"),
    )

    for column in physical_columns:
        canonical_column = f"{column}_canonical"
        if canonical_column in canonical_matches.columns:
            merged.loc[missing_existing_match, column] = canonical_matches[
                canonical_column
            ].to_numpy()

    return merged.sort_values(
        ["deep4chem_row_count", "solvent_original"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def build_report(expanded_df: pd.DataFrame) -> str:
    """Create the text report for the expanded solvent descriptor build."""
    invalid_df = expanded_df[
        (~expanded_df["is_valid_rdkit"]) | (expanded_df["is_environment_label"])
    ][["solvent_original", "deep4chem_row_count", "is_environment_label"]]

    top_30 = expanded_df[["solvent_original", "deep4chem_row_count"]].head(30)

    with pd.option_context("display.max_rows", None, "display.width", 120):
        top_30_text = top_30.to_string(index=False)
        invalid_text = invalid_df.to_string(index=False)

    lines = [
        "Deep4Chem Solvent Descriptor Report",
        "=" * 38,
        "",
        f"Unique solvents: {len(expanded_df)}",
        f"Valid RDKit solvent SMILES: {int(expanded_df['is_valid_rdkit'].sum())}",
        f"Invalid/environment labels: {len(invalid_df)}",
        "",
        "Top 30 solvents by Deep4Chem row count:",
        top_30_text,
        "",
        "Invalid/environment labels:",
        invalid_text,
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(expanded_df: pd.DataFrame, output_path: Path, report_path: Path) -> None:
    """Write the expanded descriptor CSV and text report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    expanded_df.to_csv(output_path, index=False)
    report_path.write_text(build_report(expanded_df), encoding="utf-8")


def main() -> int:
    """Run the Deep4Chem solvent descriptor build."""
    args = parse_args()

    try:
        rdkit = import_rdkit()

        deep4chem_df = load_csv_robustly(args.deep4chem)
        validate_columns(deep4chem_df, [SOLVENT_COLUMN], "Deep4Chem CSV")

        existing_df = load_csv_robustly(args.existing_solvents)

        solvent_counts = extract_unique_solvents(deep4chem_df)
        deep4chem_descriptors = build_deep4chem_descriptor_rows(solvent_counts, rdkit)
        existing_descriptors = prepare_existing_solvent_descriptors(existing_df, rdkit)
        expanded_df = merge_existing_descriptors(
            deep4chem_descriptors, existing_descriptors
        )

        write_outputs(expanded_df, args.output, args.report)

        invalid_or_environment = expanded_df[
            (~expanded_df["is_valid_rdkit"]) | (expanded_df["is_environment_label"])
        ]
        print(f"Unique solvents: {len(expanded_df)}")
        print(f"Valid RDKit solvent SMILES: {int(expanded_df['is_valid_rdkit'].sum())}")
        print(f"Invalid/environment labels: {len(invalid_or_environment)}")
        print(f"Saved expanded descriptors to: {args.output}")
        print(f"Saved report to: {args.report}")
        return 0
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
