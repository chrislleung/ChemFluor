"""Explore the Deep4Chem chromophore dataset.

Run from the project root:
    python scripts/analyze_deep4chem_dataset.py --input "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import pandas as pd


NON_MOLECULAR_LABELS = {
    "gas",
    "solid",
    "film",
    "neat",
    "vacuum",
    "air",
}

REQUIRED_COLUMNS = [
    "Chromophore",
    "Solvent",
    "Absorption max (nm)",
    "Emission max (nm)",
    "Lifetime (ns)",
    "Quantum yield",
    "log(e/mol-1 dm3 cm-1)",
]

NUMERIC_COLUMNS = [
    "Absorption max (nm)",
    "Emission max (nm)",
    "Lifetime (ns)",
    "Quantum yield",
    "log(e/mol-1 dm3 cm-1)",
]

DEFAULT_OUTPUT_DIR = Path("outputs/deep4chem_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a quick exploratory report for the Deep4Chem dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the Deep4Chem CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help=f"Directory for output files. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    return parser.parse_args()


def load_csv_robustly(csv_path: Path) -> pd.DataFrame:
    """Load a CSV with common encodings and delimiter detection fallbacks."""
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

    joined_errors = "\n".join(errors)
    raise ValueError(f"Could not load CSV after several attempts:\n{joined_errors}")


def validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        available = "\n".join(f"  - {column}" for column in df.columns)
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"Missing required column(s): {missing}\n\nAvailable columns:\n{available}"
        )


def get_smiles_validator() -> tuple[Callable[[str], bool] | None, str | None]:
    try:
        from rdkit import Chem
        from rdkit import RDLogger
    except ImportError:
        return None, "RDKit is not installed; SMILES validation was skipped."

    RDLogger.DisableLog("rdApp.*")

    def is_valid_smiles(text: str) -> bool:
        return Chem.MolFromSmiles(text) is not None

    return is_valid_smiles, None


def normalize_text_values(series: pd.Series) -> pd.Series:
    values = series.dropna().astype(str).str.strip()
    return values[values.ne("")]


def build_smiles_validity_by_value(
    unique_values: pd.Index, validator: Callable[[str], bool] | None
) -> dict[str, bool]:
    validity: dict[str, bool] = {}
    for value in unique_values:
        text = str(value)
        if text.lower() in NON_MOLECULAR_LABELS:
            validity[text] = False
        elif validator is None:
            validity[text] = False
        else:
            validity[text] = validator(text)
    return validity


def count_valid_smiles(series: pd.Series) -> int | None:
    validator, warning = get_smiles_validator()
    if warning:
        return None
    assert validator is not None
    value_counts = normalize_text_values(series).value_counts()
    validity = build_smiles_validity_by_value(value_counts.index, validator)
    return int(
        sum(count for value, count in value_counts.items() if validity[str(value)])
    )


def build_invalid_solvents(df: pd.DataFrame) -> pd.DataFrame:
    validator, _warning = get_smiles_validator()
    value_counts = normalize_text_values(df["Solvent"]).value_counts()
    if validator is None:
        invalid_counts = {
            value: count
            for value, count in value_counts.items()
            if str(value).lower() in NON_MOLECULAR_LABELS
        }
    else:
        validity = build_smiles_validity_by_value(value_counts.index, validator)
        invalid_counts = {
            value: count
            for value, count in value_counts.items()
            if not validity[str(value)]
        }
    return (
        pd.Series(invalid_counts, dtype="int64")
        .rename_axis("Solvent")
        .reset_index(name="count")
        .sort_values(["count", "Solvent"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )


def build_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    return pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count": missing_count.values,
            "missing_percent": (missing_count.values / len(df) * 100)
            if len(df)
            else 0.0,
        }
    )


def build_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    summary = numeric_df.describe().transpose()
    summary.insert(0, "non_null_count", numeric_df.notna().sum())
    summary.insert(1, "missing_count", numeric_df.isna().sum())
    return summary.reset_index(names="column")


def build_top_solvents(df: pd.DataFrame, limit: int = 30) -> pd.DataFrame:
    return (
        df["Solvent"]
        .fillna("<missing>")
        .astype(str)
        .str.strip()
        .replace("", "<blank>")
        .value_counts()
        .head(limit)
        .rename_axis("Solvent")
        .reset_index(name="count")
    )


def format_dataframe(df: pd.DataFrame) -> str:
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        160,
        "display.max_colwidth",
        80,
    ):
        return df.to_string(index=False)


def build_report(
    df: pd.DataFrame,
    missing_values: pd.DataFrame,
    top_solvents: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    invalid_solvents: pd.DataFrame,
    chromophore_valid_smiles: int | None,
    solvent_valid_smiles: int | None,
) -> str:
    rdkit_warning = ""
    if chromophore_valid_smiles is None or solvent_valid_smiles is None:
        rdkit_warning = "\nRDKit is not installed; SMILES validation was skipped.\n"

    lines = [
        "Deep4Chem Dataset Exploratory Summary",
        "=" * 42,
        "",
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        "",
        "Columns:",
        *[f"  - {column}" for column in df.columns],
        "",
        "First 5 rows:",
        format_dataframe(df.head(5)),
        "",
        "Missing values:",
        format_dataframe(missing_values),
        "",
        f"Unique chromophores: {df['Chromophore'].nunique(dropna=True)}",
        f"Unique solvents: {df['Solvent'].nunique(dropna=True)}",
        "",
        "Top 30 most common solvents:",
        format_dataframe(top_solvents),
        "",
        "Invalid solvent SMILES values:",
        format_dataframe(invalid_solvents),
        "",
        "Numeric summary:",
        format_dataframe(numeric_summary),
        "",
        "RDKit SMILES validation:",
        f"  - Valid Chromophore SMILES values: {chromophore_valid_smiles if chromophore_valid_smiles is not None else 'not computed'}",
        f"  - Valid Solvent SMILES values: {solvent_valid_smiles if solvent_valid_smiles is not None else 'not computed'}",
        rdkit_warning.rstrip(),
    ]
    return "\n".join(lines).rstrip() + "\n"


def print_report_sections(
    df: pd.DataFrame,
    missing_values: pd.DataFrame,
    top_solvents: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    invalid_solvents: pd.DataFrame,
    chromophore_valid_smiles: int | None,
    solvent_valid_smiles: int | None,
) -> None:
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn names:")
    for column in df.columns:
        print(f"- {column}")

    print("\nFirst 5 rows:")
    print(format_dataframe(df.head(5)))

    print("\nMissing-value counts and percentages:")
    print(format_dataframe(missing_values))

    print(f"\nUnique chromophores from Chromophore: {df['Chromophore'].nunique(dropna=True)}")
    print(f"Unique solvents from Solvent: {df['Solvent'].nunique(dropna=True)}")

    print("\nTop 30 most common solvents:")
    print(format_dataframe(top_solvents))

    print("\nInvalid solvent SMILES values:")
    print(format_dataframe(invalid_solvents))

    print("\nSummary statistics:")
    print(format_dataframe(numeric_summary))

    print("\nRDKit SMILES validation:")
    if chromophore_valid_smiles is None or solvent_valid_smiles is None:
        print("RDKit is not installed; SMILES validation was skipped.")
    else:
        print(f"Valid Chromophore SMILES values: {chromophore_valid_smiles}")
        print(f"Valid Solvent SMILES values: {solvent_valid_smiles}")


def write_outputs(
    output_dir: Path,
    report: str,
    missing_values: pd.DataFrame,
    top_solvents: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    invalid_solvents: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "deep4chem_summary.txt").write_text(report, encoding="utf-8")
    top_solvents.to_csv(output_dir / "top_solvents.csv", index=False)
    missing_values.to_csv(output_dir / "missing_values.csv", index=False)
    numeric_summary.to_csv(output_dir / "numeric_summary.csv", index=False)
    invalid_solvents.to_csv(output_dir / "invalid_solvents.csv", index=False)


def main() -> int:
    args = parse_args()

    try:
        df = load_csv_robustly(args.input)
        validate_required_columns(df)

        missing_values = build_missing_values(df)
        top_solvents = build_top_solvents(df)
        numeric_summary = build_numeric_summary(df)
        invalid_solvents = build_invalid_solvents(df)
        chromophore_valid_smiles = count_valid_smiles(df["Chromophore"])
        solvent_valid_smiles = count_valid_smiles(df["Solvent"])

        report = build_report(
            df=df,
            missing_values=missing_values,
            top_solvents=top_solvents,
            numeric_summary=numeric_summary,
            invalid_solvents=invalid_solvents,
            chromophore_valid_smiles=chromophore_valid_smiles,
            solvent_valid_smiles=solvent_valid_smiles,
        )

        print_report_sections(
            df=df,
            missing_values=missing_values,
            top_solvents=top_solvents,
            numeric_summary=numeric_summary,
            invalid_solvents=invalid_solvents,
            chromophore_valid_smiles=chromophore_valid_smiles,
            solvent_valid_smiles=solvent_valid_smiles,
        )
        write_outputs(
            output_dir=args.output_dir,
            report=report,
            missing_values=missing_values,
            top_solvents=top_solvents,
            numeric_summary=numeric_summary,
            invalid_solvents=invalid_solvents,
        )
        print(f"\nSaved report and quick-look CSVs to: {args.output_dir}")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
