"""Create paper-ready tables, claims, and figure captions from experiment outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TABLE_FILES = {
    "table_1_dataset_scale": [
        "workflow",
        "dataset sources",
        "raw rows",
        "cleaned/deduplicated rows",
        "unique chromophores",
        "unique solvents",
        "emission rows",
        "QY rows",
        "absorption rows",
    ],
    "table_2_methodology": [
        "workflow",
        "molecular representation",
        "solvent representation",
        "model families",
        "split strategies",
        "targets",
        "uncertainty/applicability-domain handling",
    ],
    "table_3_emission_models": [
        "split",
        "model",
        "MAE_nm",
        "RMSE_nm",
        "R2",
        "MAE_eV",
        "train_rows",
        "test_rows",
        "comment",
    ],
    "table_4_quantum_yield_models": [
        "split",
        "model",
        "QY_MAE",
        "QY_RMSE",
        "QY_R2",
        "bright_classifier_accuracy",
        "bright_classifier_F1",
    ],
    "table_5_wavelength_regions": [
        "split",
        "model",
        "UV_MAE",
        "blue_MAE",
        "green_MAE",
        "yellow_orange_MAE",
        "red_NIR_MAE",
        "worst_region",
        "comment",
    ],
    "table_6_graph_models": [
        "model",
        "seeds",
        "emission_MAE_mean",
        "emission_MAE_std",
        "best_seed_MAE",
        "QY_MAE",
        "best_use_case",
        "limitation",
    ],
}

ORIGINAL_SCALE_ALIASES = {
    "dataset sources": ["dataset sources", "dataset_sources"],
    "raw rows": ["raw rows", "raw_rows"],
    "cleaned/deduplicated rows": [
        "cleaned/deduplicated rows",
        "cleaned_rows",
        "deduplicated_rows",
    ],
    "unique chromophores": ["unique chromophores", "unique_chromophores"],
    "unique solvents": ["unique solvents", "unique_solvents"],
    "emission rows": ["emission rows", "emission_rows"],
    "QY rows": ["QY rows", "qy_rows", "quantum_yield_rows"],
    "absorption rows": ["absorption rows", "absorption_rows"],
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-results-dir", type=Path, default=Path("outputs/paper_comparison")
    )
    parser.add_argument("--original-chemfluor-reference", type=Path, default=None)
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("outputs/paper_comparison/paper_summary.md"),
    )
    parser.add_argument(
        "--out-csv-dir",
        type=Path,
        default=Path("outputs/paper_comparison/paper_tables"),
    )
    return parser.parse_args()


def warn(message: str) -> None:
    """Print a consistent non-fatal warning."""
    print(f"WARNING: {message}", file=sys.stderr)


def read_optional_csv(path: Path, label: str) -> pd.DataFrame:
    """Read a CSV if available, warning and returning an empty table otherwise."""
    if not path.exists():
        warn(f"{label} not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        warn(f"Could not read {label} at {path}: {exc}")
        return pd.DataFrame()


def numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric values."""
    return pd.to_numeric(series, errors="coerce")


def aggregate_metrics(metrics: pd.DataFrame, target: str) -> pd.DataFrame:
    """Average metric rows over seeds for one target."""
    columns = [
        "split",
        "model",
        "mae",
        "rmse",
        "r2",
        "mae_ev",
        "train_rows",
        "test_rows",
    ]
    if metrics.empty or not {"target", "split", "model"}.issubset(metrics.columns):
        return pd.DataFrame(columns=columns)
    subset = metrics[metrics["target"].astype(str) == target].copy()
    if subset.empty:
        return pd.DataFrame(columns=columns)
    available = [
        column
        for column in ["mae", "rmse", "r2", "mae_ev", "train_rows", "test_rows"]
        if column in subset.columns
    ]
    for column in available:
        subset[column] = numeric(subset[column])
    result = subset.groupby(["split", "model"], as_index=False)[available].mean()
    return result.reindex(columns=columns)


def audit_value(
    audit: pd.DataFrame, section: str, metric: str, default: Any = "not available"
) -> Any:
    """Extract one value from the long-form dataset audit."""
    required = {"section", "metric", "value"}
    if audit.empty or not required.issubset(audit.columns):
        return default
    match = audit[
        (audit["section"].astype(str) == section)
        & (audit["metric"].astype(str) == metric)
    ]
    return default if match.empty else match.iloc[0]["value"]


def original_reference_row(reference: pd.DataFrame) -> dict[str, Any]:
    """Extract a dataset-scale row from a manually supplied reference CSV."""
    row = {"workflow": "Original ChemFluor"}
    for output_column, aliases in ORIGINAL_SCALE_ALIASES.items():
        value: Any = "not provided"
        if not reference.empty:
            if {"metric", "value"}.issubset(reference.columns):
                keys = reference["metric"].astype(str).str.strip().str.lower()
                for alias in aliases:
                    match = reference[keys == alias.lower()]
                    if not match.empty:
                        value = match.iloc[0]["value"]
                        break
            else:
                normalized = {str(column).strip().lower(): column for column in reference.columns}
                for alias in aliases:
                    if alias.lower() in normalized:
                        value = reference.iloc[0][normalized[alias.lower()]]
                        break
        row[output_column] = value
    return row


def table_1(audit: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Build the original-vs-current dataset-scale table."""
    sources = "not available"
    if {"section", "source_dataset"}.issubset(audit.columns):
        source_rows = audit[audit["section"].astype(str) == "rows_by_source"]
        values = sorted(
            source_rows["source_dataset"].dropna().astype(str).unique().tolist()
        )
        if values:
            sources = ", ".join(values)
    current = {
        "workflow": "FluorCast",
        "dataset sources": sources,
        "raw rows": audit_value(audit, "deduplication", "rows_before_deduplication"),
        "cleaned/deduplicated rows": audit_value(audit, "overall", "total_rows"),
        "unique chromophores": audit_value(
            audit, "overall", "unique_canonical_chromophores"
        ),
        "unique solvents": audit_value(
            audit, "overall", "unique_canonical_solvents"
        ),
        "emission rows": audit_value(audit, "target_coverage", "emission_nm"),
        "QY rows": audit_value(audit, "target_coverage", "quantum_yield"),
        "absorption rows": audit_value(audit, "target_coverage", "absorption_nm"),
    }
    return pd.DataFrame(
        [original_reference_row(reference), current],
        columns=TABLE_FILES["table_1_dataset_scale"],
    )


def table_2(reference: pd.DataFrame) -> pd.DataFrame:
    """Build a conservative methodology comparison."""
    original = {
        "workflow": "Original ChemFluor",
        "molecular representation": "See original ChemFluor paper/manual reference",
        "solvent representation": "See original ChemFluor paper/manual reference",
        "model families": "See original ChemFluor paper/manual reference",
        "split strategies": "See original ChemFluor paper/manual reference",
        "targets": "See original ChemFluor paper/manual reference",
        "uncertainty/applicability-domain handling": (
            "Not asserted here without a manually verified reference"
        ),
    }
    if not reference.empty and "workflow" in reference.columns:
        candidates = reference[
            reference["workflow"].astype(str).str.lower().str.contains("original")
        ]
        if not candidates.empty:
            for column in TABLE_FILES["table_2_methodology"][1:]:
                if column in candidates.columns and pd.notna(candidates.iloc[0][column]):
                    original[column] = candidates.iloc[0][column]
    current = {
        "workflow": "FluorCast",
        "molecular representation": "Morgan fingerprints; graph atom/bond features for graph experiments",
        "solvent representation": "Expanded numeric solvent descriptors with train-only median imputation",
        "model families": "Random forest, Extra Trees, histogram/standard gradient boosting, MLP; optional graph neural models",
        "split strategies": "Random row, grouped molecule, grouped Bemis-Murcko scaffold",
        "targets": "Absorption wavelength, emission wavelength, quantum yield; bright QY classification at QY > 0.25",
        "uncertainty/applicability-domain handling": "Bootstrap confidence intervals, leakage reports, similarity-based applicability-domain diagnostics",
    }
    return pd.DataFrame(
        [original, current], columns=TABLE_FILES["table_2_methodology"]
    )


def table_3(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build emission model comparison table."""
    summary = aggregate_metrics(metrics, "emission_nm")
    rows = []
    for _, row in summary.iterrows():
        split = str(row["split"])
        comment = (
            "Row-level interpolation estimate; repeated molecules may cross partitions."
            if split == "random"
            else "Grouped by chromophore; tests transfer to held-out molecules."
            if split == "molecule"
            else "Grouped by Bemis-Murcko scaffold; strongest structural extrapolation test."
        )
        rows.append(
            {
                "split": split,
                "model": row["model"],
                "MAE_nm": row["mae"],
                "RMSE_nm": row["rmse"],
                "R2": row["r2"],
                "MAE_eV": row["mae_ev"],
                "train_rows": row["train_rows"],
                "test_rows": row["test_rows"],
                "comment": comment,
            }
        )
    return pd.DataFrame(rows, columns=TABLE_FILES["table_3_emission_models"])


def table_4(metrics: pd.DataFrame, classifiers: pd.DataFrame) -> pd.DataFrame:
    """Build quantum-yield regression and classification table."""
    summary = aggregate_metrics(metrics, "quantum_yield")
    classifier_summary = pd.DataFrame()
    if (
        not classifiers.empty
        and {"split", "model", "accuracy", "f1"}.issubset(classifiers.columns)
    ):
        working = classifiers.copy()
        working["accuracy"] = numeric(working["accuracy"])
        working["f1"] = numeric(working["f1"])
        classifier_summary = working.groupby(
            ["split", "model"], as_index=False
        )[["accuracy", "f1"]].mean()
    result = summary.rename(
        columns={"mae": "QY_MAE", "rmse": "QY_RMSE", "r2": "QY_R2"}
    )[["split", "model", "QY_MAE", "QY_RMSE", "QY_R2"]]
    if not classifier_summary.empty:
        result = result.merge(classifier_summary, on=["split", "model"], how="left")
    else:
        result["accuracy"] = np.nan
        result["f1"] = np.nan
    return result.rename(
        columns={
            "accuracy": "bright_classifier_accuracy",
            "f1": "bright_classifier_F1",
        }
    ).reindex(columns=TABLE_FILES["table_4_quantum_yield_models"])


def table_5(regions: pd.DataFrame) -> pd.DataFrame:
    """Pivot wavelength-region MAE into a paper-ready table."""
    columns = TABLE_FILES["table_5_wavelength_regions"]
    required = {"split", "model", "region", "mae"}
    if regions.empty or not required.issubset(regions.columns):
        return pd.DataFrame(columns=columns)
    working = regions.copy()
    working["mae"] = numeric(working["mae"])
    grouped = working.groupby(["split", "model", "region"], as_index=False)["mae"].mean()
    pivot = grouped.pivot(index=["split", "model"], columns="region", values="mae")
    region_map = {
        "UV": "UV_MAE",
        "blue": "blue_MAE",
        "green": "green_MAE",
        "yellow/orange": "yellow_orange_MAE",
        "red/NIR": "red_NIR_MAE",
    }
    result = pivot.rename(columns=region_map).reset_index()
    for column in region_map.values():
        if column not in result.columns:
            result[column] = np.nan
    metric_columns = list(region_map.values())
    result["worst_region"] = result[metric_columns].idxmax(axis=1).str.replace(
        "_MAE", "", regex=False
    )
    result["comment"] = result["worst_region"].map(
        lambda region: (
            f"Highest observed MAE is in {region.replace('_', '/')}."
            if isinstance(region, str)
            else "Insufficient regional coverage to identify a worst region."
        )
    )
    return result.reindex(columns=columns)


def find_graph_table(results_dir: Path) -> tuple[pd.DataFrame, str]:
    """Find a supported graph summary near the paper results directory."""
    candidates = [
        results_dir / "graph_seed_summary_grouped.csv",
        results_dir.parent / "graph_seed_summary_grouped.csv",
        results_dir / "graph_model_comparison.csv",
        results_dir.parent / "graph_model_experiments_fluodb" / "graph_model_comparison.csv",
    ]
    for path in candidates:
        if path.exists():
            return read_optional_csv(path, "graph model summary"), str(path)
    warn(
        "No graph summary found. Expected graph_seed_summary_grouped.csv or "
        "graph_model_comparison.csv near the paper results directory."
    )
    return pd.DataFrame(), ""


def table_6(graph: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported graph-result formats into a manuscript table."""
    columns = TABLE_FILES["table_6_graph_models"]
    if graph.empty or "model" not in graph.columns:
        return pd.DataFrame(columns=columns)
    working = graph.copy()
    if {"target", "mae"}.issubset(working.columns):
        working["mae"] = numeric(working["mae"])
        rows = []
        for model, subset in working.groupby("model"):
            emission = subset[subset["target"].astype(str) == "emission_nm"]
            qy = subset[subset["target"].astype(str) == "quantum_yield"]
            rows.append(
                {
                    "model": model,
                    "seeds": emission["seed"].nunique() if "seed" in emission else len(emission),
                    "emission_MAE_mean": emission["mae"].mean(),
                    "emission_MAE_std": emission["mae"].std(),
                    "best_seed_MAE": emission["mae"].min(),
                    "QY_MAE": qy["mae"].mean(),
                }
            )
        result = pd.DataFrame(rows)
    elif {"target", "mae_mean"}.issubset(working.columns):
        rows = []
        for model, subset in working.groupby("model"):
            emission = subset[subset["target"].astype(str) == "emission_nm"]
            qy = subset[subset["target"].astype(str) == "quantum_yield"]
            first = emission.iloc[0] if not emission.empty else pd.Series(dtype=object)
            rows.append(
                {
                    "model": model,
                    "seeds": first.get("seeds", 0),
                    "emission_MAE_mean": first.get("mae_mean", np.nan),
                    "emission_MAE_std": first.get("mae_std", np.nan),
                    "best_seed_MAE": first.get("mae_min", np.nan),
                    "QY_MAE": (
                        qy.iloc[0].get("mae_mean", np.nan) if not qy.empty else np.nan
                    ),
                }
            )
        result = pd.DataFrame(rows)
    else:
        warn("Graph summary has an unsupported schema; Table 6 will be empty.")
        return pd.DataFrame(columns=columns)
    result["best_use_case"] = (
        "Learning molecular graph representations for held-out-molecule evaluation; "
        "interpret only against matched splits."
    )
    result["limitation"] = (
        "Compute-intensive and not directly comparable to other models unless data, "
        "target, and split are matched."
    )
    return result.reindex(columns=columns)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a dependency-free Markdown table."""
    if frame.empty:
        return "_No supported result rows were available._"
    printable = frame.copy()
    for column in printable.columns:
        if pd.api.types.is_numeric_dtype(printable[column]):
            printable[column] = printable[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4g}"
            )
        else:
            printable[column] = printable[column].fillna("")
    headers = [str(column) for column in printable.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value).replace("|", "/") for value in row) + " |"
        for row in printable.to_numpy()
    )
    return "\n".join(lines)


def claims(regions: pd.DataFrame, graph: pd.DataFrame) -> list[str]:
    """Generate conservative, evidence-conditioned manuscript claims."""
    statements = [
        (
            "FluorCast expands dataset and target coverage beyond a single-source "
            "workflow, while preserving source-specific audit information."
        ),
        (
            "Random-split results should be interpreted as interpolation estimates; "
            "molecule- and scaffold-grouped results provide more stringent tests of "
            "generalization and should be emphasized when available."
        ),
        (
            "The comparison does not establish superiority over the original ChemFluor "
            "paper unless target definitions, data, and split protocols are matched."
        ),
        (
            "Region-wise and applicability-domain analyses make failure modes more "
            "visible than a single aggregate error metric."
        ),
    ]
    if not regions.empty and {"region", "mae"}.issubset(regions.columns):
        means = regions.assign(mae=numeric(regions["mae"])).groupby("region")["mae"].mean()
        if "red/NIR" in means.index and means.idxmax() == "red/NIR":
            statements.append(
                "Red/NIR emission remains the highest-error wavelength region in the "
                "available regional results and should be presented as a current weakness."
            )
    if graph.empty:
        statements.append(
            "No supported graph-model summary was available, so no performance claim "
            "about graph models is made."
        )
    else:
        statements.append(
            "Graph models are promising as representation-learning alternatives, but "
            "their value should be claimed only for targets and matched splits supported "
            "by the reported metrics."
        )
    return statements


def figure_captions() -> list[tuple[str, str]]:
    """Return the seven requested manuscript figure captions."""
    return [
        (
            "Figure 1. Dataset construction workflow.",
            "Source datasets are standardized to common molecular, solvent, and target fields, canonicalized, deduplicated, and merged before feature generation and leakage-aware evaluation.",
        ),
        (
            "Figure 2. Dataset target coverage by source.",
            "Numbers of rows with absorption wavelength, emission wavelength, and quantum-yield measurements are shown separately for each contributing dataset source.",
        ),
        (
            "Figure 3. Emission wavelength distribution by dataset source.",
            "Emission measurements are stratified by source to show coverage differences across ultraviolet, visible, and red/near-infrared wavelengths.",
        ),
        (
            "Figure 4. Random versus molecule versus scaffold split performance.",
            "Emission errors are compared across row-random, held-out-molecule, and held-out-Bemis-Murcko-scaffold tests; grouped splits provide the more stringent generalization estimates.",
        ),
        (
            "Figure 5. Model-family comparison.",
            "Emission performance is compared across the evaluated ensemble, boosting, and neural model families using the same target and split within each comparison.",
        ),
        (
            "Figure 6. Wavelength-region error heatmap.",
            "Mean absolute emission error is shown for UV, blue, green, yellow/orange, and red/NIR regions to identify wavelength-dependent failure modes.",
        ),
        (
            "Figure 7. Applicability-domain benchmark prediction.",
            "Benchmark predictions are interpreted alongside nearest-training-molecule similarity and applicability-domain status; disagreement or low similarity is treated as reduced confidence rather than evidence of model superiority.",
        ),
    ]


def write_outputs(
    tables: dict[str, pd.DataFrame],
    statements: list[str],
    out_md: Path,
    out_csv_dir: Path,
) -> None:
    """Write separate CSV tables and the combined paper summary."""
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        "# Paper-ready comparison summary",
        "",
        (
            "Values are summarized from available result files. Missing original-paper "
            "values are left as not provided rather than inferred."
        ),
        "",
    ]
    for index, (name, table) in enumerate(tables.items(), start=1):
        path = out_csv_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        title = name.replace("_", " ").title()
        sections.extend([f"## {title}", "", markdown_table(table), ""])
    sections.extend(["## Main manuscript claims supported by results", ""])
    sections.extend([f"- {statement}" for statement in statements])
    sections.extend(["", "## Figure captions", ""])
    for title, caption in figure_captions():
        sections.extend([f"### {title}", "", caption, ""])
    out_md.write_text("\n".join(sections), encoding="utf-8")


def main() -> int:
    """Create all paper-ready summary outputs."""
    args = parse_args()
    results_dir = args.paper_results_dir
    if not results_dir.exists():
        warn(f"Paper results directory not found: {results_dir}")

    audit = read_optional_csv(results_dir / "dataset_audit.csv", "dataset audit")
    metrics = read_optional_csv(
        results_dir / "metrics_by_split_model_target.csv", "model metrics"
    )
    regions = read_optional_csv(
        results_dir / "region_metrics_by_split_model.csv", "regional metrics"
    )
    classifiers = read_optional_csv(
        results_dir / "qy_classifier_metrics.csv", "QY classifier metrics"
    )
    reference = (
        read_optional_csv(args.original_chemfluor_reference, "original ChemFluor reference")
        if args.original_chemfluor_reference is not None
        else pd.DataFrame()
    )
    if args.original_chemfluor_reference is None:
        warn(
            "No original ChemFluor reference CSV supplied; original-paper numeric "
            "fields will be marked as not provided."
        )
    graph, graph_source = find_graph_table(results_dir)
    if graph_source:
        print(f"Using graph summary: {graph_source}")

    tables = {
        "table_1_dataset_scale": table_1(audit, reference),
        "table_2_methodology": table_2(reference),
        "table_3_emission_models": table_3(metrics),
        "table_4_quantum_yield_models": table_4(metrics, classifiers),
        "table_5_wavelength_regions": table_5(regions),
        "table_6_graph_models": table_6(graph),
    }
    write_outputs(tables, claims(regions, graph), args.out_md, args.out_csv_dir)
    print(f"Saved paper summary: {args.out_md}")
    print(f"Saved paper tables: {args.out_csv_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
