"""Analyze prediction errors for combined ChemFluor Random Forest models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


DEFAULT_MODEL_DIR = Path("models/chemfluor_combined")
DEFAULT_OUT_DIR = Path("outputs/error_analysis")
REQUIRED_COLUMNS = {"y_true", "y_pred", "residual"}
WAVELENGTH_TARGETS = {"absorption_nm", "emission_nm"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create error-analysis tables and markdown for combined RF predictions."
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    return parser.parse_args()


def discover_prediction_files(model_dir: Path) -> dict[str, Path]:
    """Find predictions_{target}.csv files in the model directory."""
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    files = sorted(model_dir.glob("predictions_*.csv"))
    if not files:
        raise FileNotFoundError(f"No predictions_*.csv files found in {model_dir}")

    return {
        path.stem.removeprefix("predictions_"): path
        for path in files
    }


def load_predictions(path: Path) -> pd.DataFrame:
    """Load one prediction CSV and add absolute error."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")

    df = df.copy()
    for column in ["y_true", "y_pred", "residual"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["absolute_error"] = (df["y_true"] - df["y_pred"]).abs()
    return df


def summarize_group_error(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Summarize absolute error by a categorical column."""
    if group_column not in df.columns:
        return pd.DataFrame()

    summary = (
        df.dropna(subset=[group_column])
        .groupby(group_column, dropna=False)
        .agg(
            rows=("absolute_error", "size"),
            mean_absolute_error=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            rmse=("residual", lambda values: (values.pow(2).mean()) ** 0.5),
            mean_residual=("residual", "mean"),
        )
        .reset_index()
        .sort_values(["mean_absolute_error", "rows"], ascending=[False, False])
    )
    return summary


def wavelength_region(value: float) -> str:
    """Assign an actual wavelength to a coarse spectral region."""
    if pd.isna(value):
        return "missing"
    if value < 400:
        return "UV"
    if value < 480:
        return "blue"
    if value < 560:
        return "green"
    if value <= 620:
        return "yellow/orange"
    return "red/NIR"


def summarize_wavelength_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize error by actual wavelength region."""
    region_df = df.copy()
    region_df["wavelength_region"] = region_df["y_true"].map(wavelength_region)
    region_order = ["UV", "blue", "green", "yellow/orange", "red/NIR", "missing"]
    summary = summarize_group_error(region_df, "wavelength_region")
    if summary.empty:
        return summary

    summary["wavelength_region"] = pd.Categorical(
        summary["wavelength_region"], categories=region_order, ordered=True
    )
    return summary.sort_values("wavelength_region").reset_index(drop=True)


def save_target_outputs(
    target: str, df: pd.DataFrame, out_dir: Path
) -> dict[str, pd.DataFrame]:
    """Save best/worst predictions and summary tables for one target."""
    worst = df.sort_values("absolute_error", ascending=False).head(50)
    best = df.sort_values("absolute_error", ascending=True).head(50)
    source_summary = summarize_group_error(df, "source_dataset")
    solvent_summary = summarize_group_error(df, "solvent_original").head(30)

    worst.to_csv(out_dir / f"worst_predictions_{target}.csv", index=False)
    best.to_csv(out_dir / f"best_predictions_{target}.csv", index=False)
    if not source_summary.empty:
        source_summary.to_csv(out_dir / f"error_by_source_dataset_{target}.csv", index=False)
    if not solvent_summary.empty:
        solvent_summary.to_csv(out_dir / f"top_error_solvents_{target}.csv", index=False)

    summaries = {
        "overall": pd.DataFrame(
            [
                {
                    "target": target,
                    "rows": len(df),
                    "mean_absolute_error": df["absolute_error"].mean(),
                    "median_absolute_error": df["absolute_error"].median(),
                    "rmse": (df["residual"].pow(2).mean()) ** 0.5,
                    "mean_residual": df["residual"].mean(),
                }
            ]
        ),
        "worst": worst,
        "best": best,
        "source": source_summary,
        "solvent": solvent_summary,
    }

    if target in WAVELENGTH_TARGETS:
        wavelength_summary = summarize_wavelength_regions(df)
        wavelength_summary.to_csv(
            out_dir / f"error_by_wavelength_region_{target}.csv", index=False
        )
        summaries["wavelength"] = wavelength_summary

    return summaries


def describe_best_worst_target(target_summaries: dict[str, dict[str, pd.DataFrame]]) -> tuple[str, str]:
    """Return target names with lowest and highest overall mean absolute error."""
    target_maes: dict[str, float] = {}
    for target, summaries in target_summaries.items():
        overall = summaries["overall"]
        if not overall.empty:
            target_maes[target] = float(overall.loc[0, "mean_absolute_error"])

    if not target_maes:
        return "unknown", "unknown"

    return min(target_maes, key=target_maes.get), max(target_maes, key=target_maes.get)


def summarize_performance_patterns(
    target_summaries: dict[str, dict[str, pd.DataFrame]]
) -> list[str]:
    """Create short markdown bullets from source, solvent, and wavelength summaries."""
    bullets: list[str] = []
    for target, summaries in target_summaries.items():
        source = summaries.get("source", pd.DataFrame())
        solvent = summaries.get("solvent", pd.DataFrame())
        wavelength = summaries.get("wavelength", pd.DataFrame())

        if not source.empty:
            best_source = source.sort_values("mean_absolute_error").iloc[0]
            worst_source = source.sort_values("mean_absolute_error", ascending=False).iloc[0]
            bullets.append(
                f"- `{target}` source split: lowest mean absolute error is for "
                f"`{best_source['source_dataset']}` ({best_source['mean_absolute_error']:.4g}); "
                f"highest is for `{worst_source['source_dataset']}` ({worst_source['mean_absolute_error']:.4g})."
            )

        if not solvent.empty:
            top_solvent = solvent.iloc[0]
            bullets.append(
                f"- `{target}` solvent pattern: highest-error solvent among the top 30 table is "
                f"`{top_solvent['solvent_original']}` with mean absolute error "
                f"{top_solvent['mean_absolute_error']:.4g} over {int(top_solvent['rows'])} row(s)."
            )

        if not wavelength.empty:
            best_region = wavelength.sort_values("mean_absolute_error").iloc[0]
            worst_region = wavelength.sort_values(
                "mean_absolute_error", ascending=False
            ).iloc[0]
            bullets.append(
                f"- `{target}` wavelength regions: best region is "
                f"`{best_region['wavelength_region']}` ({best_region['mean_absolute_error']:.4g}); "
                f"worst region is `{worst_region['wavelength_region']}` "
                f"({worst_region['mean_absolute_error']:.4g})."
            )

    return bullets


def build_markdown_report(
    target_summaries: dict[str, dict[str, pd.DataFrame]], out_dir: Path
) -> str:
    """Build a markdown report explaining the error analysis."""
    best_target, worst_target = describe_best_worst_target(target_summaries)
    pattern_bullets = summarize_performance_patterns(target_summaries)
    target_list = ", ".join(f"`{target}`" for target in sorted(target_summaries))

    output_lines = []
    for target in sorted(target_summaries):
        output_lines.extend(
            [
                f"- `worst_predictions_{target}.csv`",
                f"- `best_predictions_{target}.csv`",
                f"- `error_by_source_dataset_{target}.csv` if `source_dataset` was present",
                f"- `top_error_solvents_{target}.csv`",
            ]
        )
        if target in WAVELENGTH_TARGETS:
            output_lines.append(f"- `error_by_wavelength_region_{target}.csv`")

    return f"""# Combined Random Forest Error Analysis

## Scope

This report analyzes held-out prediction files from the combined ChemFluor + Deep4Chem Random Forest model directory. Targets analyzed: {target_list}.

For each target, the script computed absolute error as `abs(y_true - y_pred)`, saved the 50 worst predictions, saved the 50 best predictions, summarized error by dataset source when available, and summarized the highest-error solvents.

## Where The Model Performs Best

The best individual predictions are saved in `best_predictions_<target>.csv`. At a high level, the lowest-error examples are rows where the model prediction closely matches the measured held-out value, often reflecting chromophore/solvent combinations that are well represented by the learned fingerprint and solvent descriptor space.

Based on the sampled best/worst error contrast, `{best_target}` shows the strongest relative error behavior among the analyzed targets.

## Where The Model Performs Worst

The largest-error examples are saved in `worst_predictions_<target>.csv`. These rows are the first place to inspect for unusual chromophores, rare solvents, possible label noise, uncommon wavelength ranges, or experimental cases where the available descriptors do not capture the controlling photophysics.

Based on the sampled best/worst error contrast, `{worst_target}` shows the weakest relative error behavior among the analyzed targets.

## Source, Solvent, And Wavelength Patterns

{chr(10).join(pattern_bullets)}

For `absorption_nm` and `emission_nm`, wavelength-region summaries use the measured `y_true` value:

- UV: <400 nm
- blue: 400-480 nm
- green: 480-560 nm
- yellow/orange: 560-620 nm
- red/NIR: >620 nm

If one region has noticeably larger mean absolute error, that suggests the model has less reliable coverage in that spectral regime. If particular solvents dominate the top-error solvent tables, those solvents are good candidates for descriptor review, data-quality inspection, or targeted additional training data.

## Output Files

All outputs are written under `{out_dir}`.

{chr(10).join(output_lines)}
"""


def save_overall_summary(
    target_summaries: dict[str, dict[str, pd.DataFrame]], out_dir: Path
) -> None:
    """Save one overall target-level error summary CSV."""
    overall = pd.concat(
        [summaries["overall"] for summaries in target_summaries.values()],
        ignore_index=True,
    ).sort_values("mean_absolute_error")
    overall.to_csv(out_dir / "overall_error_summary.csv", index=False)


def main() -> int:
    """Run error analysis."""
    args = parse_args()
    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        prediction_files = discover_prediction_files(args.model_dir)

        target_summaries: dict[str, dict[str, pd.DataFrame]] = {}
        for target, path in prediction_files.items():
            predictions = load_predictions(path)
            target_summaries[target] = save_target_outputs(
                target=target,
                df=predictions,
                out_dir=args.out_dir,
            )

        save_overall_summary(target_summaries, args.out_dir)
        report = build_markdown_report(target_summaries, args.out_dir)
        (args.out_dir / "error_analysis_report.md").write_text(report, encoding="utf-8")

        print(f"Analyzed {len(target_summaries)} target(s).")
        print(f"Saved error analysis outputs to: {args.out_dir}")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
