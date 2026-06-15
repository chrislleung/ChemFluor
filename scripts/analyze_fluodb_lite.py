"""Analyze FluoDB-Lite before using it for model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = ["absorption/nm", "emission/nm", "plqy", "e/m-1cm-1"]
RED_THRESHOLDS = [550, 580, 600, 650, 700, 750]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a FluoDB-Lite CSV file.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def value_counts_csv(df: pd.DataFrame, column: str, path: Path) -> pd.DataFrame:
    """Save value counts for a column, or an empty count table if absent."""
    if column not in df.columns:
        counts = pd.DataFrame(columns=[column, "count"])
    else:
        counts = (
            df[column]
            .fillna("<missing>")
            .astype(str)
            .value_counts()
            .rename_axis(column)
            .reset_index(name="count")
        )
    counts.to_csv(path, index=False)
    return counts


def analyze_fluodb_lite(input_path: Path, out_dir: Path) -> dict[str, object]:
    """Analyze and save compact FluoDB-Lite diagnostics."""
    if not input_path.exists():
        raise FileNotFoundError(f"FluoDB-Lite CSV not found: {input_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path, low_memory=False)
    numeric = df.copy()
    for column in NUMERIC_COLUMNS:
        if column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    missing = (
        df.isna()
        .sum()
        .rename_axis("column")
        .reset_index(name="missing_count")
    )
    missing["missing_fraction"] = missing["missing_count"] / max(len(df), 1)
    missing.to_csv(out_dir / "missing_values.csv", index=False)

    numeric_summary = numeric[[c for c in NUMERIC_COLUMNS if c in numeric.columns]].describe().T
    numeric_summary.to_csv(out_dir / "numeric_summary.csv")
    source_counts = value_counts_csv(df, "source", out_dir / "source_counts.csv")
    value_counts_csv(df, "tag_name", out_dir / "scaffold_counts.csv")
    value_counts_csv(df, "split", out_dir / "split_counts.csv")
    value_counts_csv(df, "solvent", out_dir / "solvent_counts.csv")

    emission = pd.to_numeric(df.get("emission/nm"), errors="coerce")
    red_counts = {f"emission_ge_{threshold}": int((emission >= threshold).sum()) for threshold in RED_THRESHOLDS}
    red_ge_600 = df.loc[emission >= 600].copy()
    red_ge_600.head(100).to_csv(out_dir / "red_region_rows_preview.csv", index=False)
    value_counts_csv(red_ge_600, "tag_name", out_dir / "top_red_scaffolds_ge600.csv")
    value_counts_csv(red_ge_600, "source", out_dir / "top_red_sources_ge600.csv")

    usable_counts = {
        column: int(pd.to_numeric(df.get(column), errors="coerce").notna().sum())
        for column in NUMERIC_COLUMNS
    }
    summary = {
        "input": str(input_path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(map(str, df.columns)),
        "unique_fluorophore_smiles": int(df["smiles"].nunique()) if "smiles" in df.columns else 0,
        "unique_solvents": int(df["solvent"].nunique()) if "solvent" in df.columns else 0,
        "usable_target_counts": usable_counts,
        "red_region_counts": red_counts,
        "top_sources": source_counts.head(10).to_dict(orient="records"),
    }
    lines = [
        "FluoDB-Lite Analysis Summary",
        f"Input: {input_path}",
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        "",
        "Columns:",
        *[f"- {column}" for column in df.columns],
        "",
        f"Unique fluorophore SMILES: {summary['unique_fluorophore_smiles']}",
        f"Unique solvents: {summary['unique_solvents']}",
        "",
        "Usable target rows:",
        *[f"- {column}: {count}" for column, count in usable_counts.items()],
        "",
        "Red/orange/NIR emission coverage:",
        *[f"- >= {threshold} nm: {red_counts[f'emission_ge_{threshold}']}" for threshold in RED_THRESHOLDS],
    ]
    (out_dir / "fluodb_lite_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return summary


def main() -> int:
    args = parse_args()
    try:
        analyze_fluodb_lite(args.input, args.out_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
