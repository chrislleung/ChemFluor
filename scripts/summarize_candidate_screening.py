"""Summarize ranked ChemFluor candidate screening outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


EXPECTED_COLUMNS = [
    "target_emission_nm",
    "rank",
    "name",
    "scaffold",
    "substituent",
    "smiles",
    "canonical_smiles",
    "solvent_smiles",
    "predicted_absorption_nm",
    "predicted_emission_nm",
    "predicted_quantum_yield",
    "predicted_log_extinction",
    "nearest_training_similarity",
    "nearest_training_smiles",
    "outside_applicability_domain",
    "emission_error_from_target",
    "score",
    "estimated_brightness_score",
]

OPTIONAL_INPUT_COLUMNS = [column for column in EXPECTED_COLUMNS if column not in {"target_emission_nm", "rank"}]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize ranked ChemFluor candidate screening CSV files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        type=Path,
        help="One or more ranked screening CSV files.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        required=True,
        type=float,
        help="One target emission wavelength per input CSV.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output summary CSV path.")
    parser.add_argument(
        "--markdown", required=True, type=Path, help="Output Markdown report path."
    )
    parser.add_argument(
        "--top-n",
        default=10,
        type=int,
        help="Number of top-ranked candidates to summarize per target.",
    )
    return parser.parse_args()


def validate_inputs(input_paths: Sequence[Path], targets: Sequence[float], top_n: int) -> None:
    """Validate CLI inputs before reading data."""
    if len(input_paths) != len(targets):
        raise ValueError(
            f"Expected one target per input file, got {len(input_paths)} input file(s) "
            f"and {len(targets)} target value(s)."
        )
    if top_n < 1:
        raise ValueError(f"--top-n must be at least 1, got {top_n}.")
    missing_paths = [path for path in input_paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Input file(s) not found: {missing}")


def format_number(value: object, digits: int = 3) -> str:
    """Format numeric values compactly for Markdown tables."""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def markdown_cell(value: object) -> str:
    """Format a value for use in a Markdown table cell."""
    return format_number(value).replace("|", "\\|")


def format_target(target: float) -> str:
    """Format a target wavelength without noisy decimal places."""
    return format_number(target, digits=1)


def warn_missing_columns(path: Path, missing_columns: Sequence[str]) -> None:
    """Print a deterministic warning for missing optional columns."""
    if missing_columns:
        print(
            "WARNING: "
            f"{path} is missing optional column(s): {', '.join(missing_columns)}",
            file=sys.stderr,
        )


def load_top_candidates(path: Path, target: float, top_n: int) -> pd.DataFrame:
    """Load one ranked CSV and return its top candidates with target and rank columns."""
    ranked = pd.read_csv(path)
    missing_columns = [column for column in OPTIONAL_INPUT_COLUMNS if column not in ranked.columns]
    warn_missing_columns(path, missing_columns)

    top = ranked.head(top_n).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    top.insert(0, "target_emission_nm", target)

    available_columns = [column for column in EXPECTED_COLUMNS if column in top.columns]
    return top.loc[:, available_columns]


def summarize_screening(
    input_paths: Sequence[Path],
    targets: Sequence[float],
    out_path: Path,
    markdown_path: Path,
    top_n: int = 10,
) -> pd.DataFrame:
    """Create a combined top-N CSV summary and Markdown screening report."""
    validate_inputs(input_paths, targets, top_n)

    summary_tables = [
        load_top_candidates(path, target, top_n)
        for path, target in zip(input_paths, targets, strict=True)
    ]
    summary = pd.concat(summary_tables, ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    markdown_path.write_text(build_markdown_report(summary, top_n), encoding="utf-8")
    return summary


def target_region_note(target: float) -> str | None:
    """Return a simple color-region interpretation for common emission targets."""
    if abs(target - 450) <= 25:
        return "This target can be interpreted as blue-region screening."
    if abs(target - 520) <= 25:
        return "This target can be interpreted as green-region screening."
    if abs(target - 600) <= 25:
        return "This target can be interpreted as orange/red-region screening."
    return None


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Build a Markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def candidate_value(row: pd.Series, column: str) -> object:
    """Get a candidate value with a blank fallback for missing columns."""
    return row[column] if column in row.index else ""


def build_best_candidate_table(summary: pd.DataFrame) -> str:
    """Build the one-row-per-target Markdown table."""
    has_brightness = "estimated_brightness_score" in summary.columns
    headers = [
        "Target emission",
        "Top candidate",
        "Scaffold",
        "Substituent",
        "Predicted emission",
        "Predicted quantum yield",
        "Emission error",
        "Score",
    ]
    if has_brightness:
        headers.append("Estimated brightness score")

    rows = []
    for target in sorted(summary["target_emission_nm"].unique()):
        target_rows = summary[summary["target_emission_nm"] == target].sort_values("rank")
        best = target_rows.iloc[0]
        row = [
            f"{format_target(float(target))} nm",
            candidate_value(best, "name"),
            candidate_value(best, "scaffold"),
            candidate_value(best, "substituent"),
            candidate_value(best, "predicted_emission_nm"),
            candidate_value(best, "predicted_quantum_yield"),
            candidate_value(best, "emission_error_from_target"),
            candidate_value(best, "score"),
        ]
        if has_brightness:
            row.append(candidate_value(best, "estimated_brightness_score"))
        rows.append(row)

    return markdown_table(headers, rows)


def value_counts_lines(rows: pd.DataFrame, column: str) -> list[str]:
    """Format value counts for a candidate metadata column."""
    if column not in rows.columns:
        return []
    counts = rows[column].dropna().astype(str).value_counts(sort=True)
    return [f"- {value}: {count}" for value, count in counts.items()]


def build_target_section(target: float, rows: pd.DataFrame, top_n: int) -> str:
    """Build the Markdown section for one target wavelength."""
    best = rows.sort_values("rank").iloc[0]
    lines = [
        f"## Target emission: {format_target(target)} nm",
        "",
        f"- Top candidate: {format_number(candidate_value(best, 'name'))}",
        f"- Predicted emission: {format_number(candidate_value(best, 'predicted_emission_nm'))} nm",
        f"- Predicted quantum yield: {format_number(candidate_value(best, 'predicted_quantum_yield'))}",
        f"- Emission error: {format_number(candidate_value(best, 'emission_error_from_target'))} nm",
    ]

    region_note = target_region_note(target)
    if region_note:
        lines.append(f"- Interpretation: {region_note}")

    emission_error = candidate_value(best, "emission_error_from_target")
    numeric_emission_error = pd.to_numeric(emission_error, errors="coerce")
    if pd.notna(numeric_emission_error) and float(numeric_emission_error) > 30:
        lines.append(
            "- Note: the best candidate is more than 30 nm from the target, so the "
            "current candidate library may not contain candidates close enough to "
            "that target and should be expanded."
        )

    scaffold_lines = value_counts_lines(rows, "scaffold")
    if scaffold_lines:
        lines.extend(["", f"Top scaffold counts among the top {top_n} candidates:"])
        lines.extend(scaffold_lines)

        top_scaffold_count = int(rows["scaffold"].dropna().astype(str).value_counts().iloc[0])
        if top_scaffold_count >= len(rows) / 2:
            top_scaffold = rows["scaffold"].dropna().astype(str).value_counts().index[0]
            lines.append(
                f"- Note: the ranking is dominated by the {top_scaffold} scaffold family."
            )

    substituent_lines = value_counts_lines(rows, "substituent")
    if substituent_lines:
        lines.extend(["", f"Top substituent counts among the top {top_n} candidates:"])
        lines.extend(substituent_lines)

    return "\n".join(lines)


def build_markdown_report(summary: pd.DataFrame, top_n: int) -> str:
    """Build the full Markdown screening report."""
    lines = [
        "# Candidate Screening Summary",
        "",
        "This report summarizes model-ranked candidate fluorophores for different "
        "target emission wavelengths.",
        "",
        "## Best Candidate by Target",
        "",
        build_best_candidate_table(summary),
        "",
    ]

    for target in sorted(summary["target_emission_nm"].unique()):
        target_rows = summary[summary["target_emission_nm"] == target].sort_values("rank")
        lines.append(build_target_section(float(target), target_rows, top_n))
        lines.append("")

    lines.extend(
        [
            "## Caution",
            "",
            "These are model-ranked candidates for prioritization, not experimentally "
            "validated fluorophores.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run the command-line summarizer."""
    args = parse_args()
    try:
        summary = summarize_screening(
            input_paths=args.inputs,
            targets=args.targets,
            out_path=args.out,
            markdown_path=args.markdown,
            top_n=args.top_n,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print(f"Wrote {len(summary)} summarized candidate row(s) to {args.out}")
    print(f"Wrote Markdown report to {args.markdown}")


if __name__ == "__main__":
    main()
