"""Prepare a deduplicated ChemFluor + Deep4Chem + FluoDB-Lite dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.data_standardization import (  # noqa: E402
    SOURCE_PREFERENCE,
    TARGET_COLUMNS,
    analyze_dataset_overlap,
    deduplicate_standardized_rows,
    load_chemfluor,
    load_deep4chem,
    load_fluodb_lite,
    molecule_solvent_replicates,
    red_region_counts,
)


RED_THRESHOLDS = [550, 580, 600, 650, 700]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardize, analyze overlap, and deduplicate FluoDB-Lite."
    )
    parser.add_argument("--fluodb", required=True, type=Path)
    parser.add_argument("--chemfluor", required=True, type=Path)
    parser.add_argument("--deep4chem", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def source_red_counts(df: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    """Build red-region count rows by source."""
    rows: list[dict[str, Any]] = []
    for source, group in df.groupby("source_dataset", dropna=False):
        counts = red_region_counts(group)
        rows.append({"stage": label, "source_dataset": source, "rows": len(group), **counts})
    rows.append({"stage": label, "source_dataset": "ALL", "rows": len(df), **red_region_counts(df)})
    return rows


def json_safe(payload: Any) -> Any:
    """Convert NumPy/pandas scalars for JSON serialization."""
    if hasattr(payload, "item"):
        try:
            return payload.item()
        except ValueError:
            pass
    if isinstance(payload, dict):
        return {str(key): json_safe(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [json_safe(value) for value in payload]
    return payload


def build_overlap_report(
    before: pd.DataFrame,
    after: pd.DataFrame,
    overlap_before: dict[str, Any],
    overlap_after: dict[str, Any],
) -> str:
    """Build a concise Markdown overlap report."""
    lines = [
        "# FluoDB-Lite Overlap Report",
        "",
        "## Rows By Source Before Deduplication",
        "",
    ]
    for source, count in overlap_before["rows_by_source"].items():
        lines.append(f"- {source}: {count}")
    lines.extend(
        [
            "",
            "## Deduplication",
            "",
            f"- Total rows before deduplication: {len(before)}",
            f"- Rows after deduplication: {len(after)}",
            f"- Exact duplicate rows removed: {len(before) - len(after)}",
            f"- FluoDB-Lite exact overlaps with ChemFluor: {overlap_before['fluodb_exact_overlaps_with_chemfluor']}",
            f"- FluoDB-Lite exact overlaps with Deep4Chem: {overlap_before['fluodb_exact_overlaps_with_deep4chem']}",
            f"- Molecule-solvent pairs with multiple measurements: {overlap_before['molecule_solvent_pairs_with_multiple_measurements']}",
            "",
            "Exact duplicates are defined as rows with the same canonical chromophore SMILES, canonical solvent SMILES, absorption, emission, quantum yield, and log extinction. Source priority is ChemFluor, then Deep4Chem, then FluoDB-Lite.",
            "",
            "## Red/Orange/NIR Coverage",
            "",
        ]
    )
    for threshold in RED_THRESHOLDS:
        key = f"emission_ge_{threshold}"
        lines.append(
            f"- >= {threshold} nm: before {overlap_before['red_region_counts'][key]}, "
            f"after {overlap_after['red_region_counts'][key]}"
        )
    lines.append("")
    return "\n".join(lines)


def prepare_fluodb_lite(
    fluodb_path: Path,
    chemfluor_path: Path,
    deep4chem_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Create standardized and exact-deduplicated FluoDB-Lite integration files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chemfluor = load_chemfluor(chemfluor_path)
    deep4chem = load_deep4chem(deep4chem_path)
    fluodb = load_fluodb_lite(fluodb_path)

    chemfluor.to_csv(out_dir / "chemfluor_standardized.csv", index=False)
    deep4chem.to_csv(out_dir / "deep4chem_standardized.csv", index=False)
    fluodb.to_csv(out_dir / "fluodb_lite_standardized.csv", index=False)

    combined = pd.concat([chemfluor, deep4chem, fluodb], ignore_index=True, sort=False)
    combined = combined.dropna(subset=["canonical_chromophore_smiles"]).copy()
    combined = combined.dropna(subset=TARGET_COLUMNS, how="all").copy()
    combined.to_csv(out_dir / "combined_before_dedup.csv", index=False)
    overlap_before = analyze_dataset_overlap(combined)
    deduplicated = deduplicate_standardized_rows(combined, prefer_sources=SOURCE_PREFERENCE)
    deduplicated.to_csv(out_dir / "combined_deduplicated.csv", index=False)
    overlap_after = analyze_dataset_overlap(deduplicated)

    replicates = molecule_solvent_replicates(combined)
    replicates.to_csv(out_dir / "molecule_solvent_replicates.csv", index=False)

    red_summary = pd.DataFrame(
        [*source_red_counts(combined, "before_dedup"), *source_red_counts(deduplicated, "after_dedup")]
    )
    red_summary.to_csv(out_dir / "red_region_summary.csv", index=False)

    summary = {
        "source_preference": SOURCE_PREFERENCE,
        "before_deduplication": overlap_before,
        "after_deduplication": overlap_after,
        "exact_duplicate_rows_removed": int(len(combined) - len(deduplicated)),
    }
    (out_dir / "overlap_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    report = build_overlap_report(combined, deduplicated, overlap_before, overlap_after)
    (out_dir / "overlap_report.md").write_text(report, encoding="utf-8")

    print(report)
    return summary


def main() -> int:
    args = parse_args()
    try:
        prepare_fluodb_lite(args.fluodb, args.chemfluor, args.deep4chem, args.out_dir)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
