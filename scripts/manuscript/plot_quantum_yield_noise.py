"""Visualize repeated-measurement noise in experimental quantum-yield data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/processed/fluodb_lite/combined_deduplicated.csv")
DEFAULT_OUT_DIR = Path("outputs/paper_comparison/qy_noise")
MOLECULE_COLUMN = "canonical_chromophore_smiles"
SOLVENT_COLUMN = "canonical_solvent_smiles"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-replicates", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=25)
    return parser.parse_args()


def warn(message: str) -> None:
    """Print a non-fatal warning."""
    print(f"WARNING: {message}", file=sys.stderr)


def _nonblank(series: pd.Series) -> pd.Series:
    """Return a mask selecting non-null, non-whitespace values."""
    return series.notna() & series.astype(str).str.strip().ne("")


def prepare_qy_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Select valid QY rows and resolve the columns used for grouping.

    Returns ``(valid_rows, excluded_physical_values, group_columns)``. Values
    that cannot be parsed as numbers are treated as missing, not as physically
    invalid measurements.
    """
    required = {MOLECULE_COLUMN, "quantum_yield", "source_dataset"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required input columns: {', '.join(missing)}")

    working = data.copy()
    working["quantum_yield"] = pd.to_numeric(working["quantum_yield"], errors="coerce")
    qy_rows = working.loc[working["quantum_yield"].notna()].copy()
    physically_valid = qy_rows["quantum_yield"].between(0.0, 1.0, inclusive="both")
    excluded = qy_rows.loc[~physically_valid].copy()
    valid = qy_rows.loc[physically_valid].copy()

    has_canonical = SOLVENT_COLUMN in valid.columns
    has_original = "solvent_original" in valid.columns
    if has_canonical:
        valid[SOLVENT_COLUMN] = valid[SOLVENT_COLUMN].where(
            _nonblank(valid[SOLVENT_COLUMN]), np.nan
        )
        if has_original:
            fallback = valid["solvent_original"].where(
                _nonblank(valid["solvent_original"]), np.nan
            )
            valid[SOLVENT_COLUMN] = valid[SOLVENT_COLUMN].fillna(fallback)

    if has_canonical and valid[SOLVENT_COLUMN].notna().any():
        # A stable placeholder prevents unknown-solvent records from disappearing
        # silently during pandas groupby.
        valid[SOLVENT_COLUMN] = valid[SOLVENT_COLUMN].fillna("[unknown solvent]")
        group_columns = [MOLECULE_COLUMN, SOLVENT_COLUMN]
    elif has_original and _nonblank(valid["solvent_original"]).any():
        valid[SOLVENT_COLUMN] = valid["solvent_original"].where(
            _nonblank(valid["solvent_original"]), "[unknown solvent]"
        )
        group_columns = [MOLECULE_COLUMN, SOLVENT_COLUMN]
        warn("canonical_solvent_smiles is unavailable; using solvent_original.")
    else:
        group_columns = [MOLECULE_COLUMN]
        warn("No usable solvent column found; grouping quantum yields by molecule only.")

    return valid, excluded, group_columns


def compute_group_statistics(
    data: pd.DataFrame,
    group_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Calculate repeated-measurement statistics for each molecular group."""
    if group_columns is None:
        group_columns = [MOLECULE_COLUMN, SOLVENT_COLUMN]
    group_columns = list(group_columns)
    columns = group_columns + [
        "n_qy_records",
        "n_unique_qy",
        "mean_qy",
        "median_qy",
        "std_qy",
        "min_qy",
        "max_qy",
        "qy_range",
        "qy_iqr",
        "n_source_datasets",
        "source_datasets",
    ]
    if data.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in data.groupby(grouper, dropna=False, sort=True):
        keys = (key,) if len(group_columns) == 1 else tuple(key)
        qy = group["quantum_yield"].astype(float)
        sources = sorted(group["source_dataset"].dropna().astype(str).unique())
        row: dict[str, object] = dict(zip(group_columns, keys))
        row.update(
            {
                "n_qy_records": int(qy.size),
                "n_unique_qy": int(qy.nunique()),
                "mean_qy": qy.mean(),
                "median_qy": qy.median(),
                "std_qy": qy.std(ddof=1),
                "min_qy": qy.min(),
                "max_qy": qy.max(),
                "qy_range": qy.max() - qy.min(),
                "qy_iqr": qy.quantile(0.75) - qy.quantile(0.25) if qy.size >= 4 else np.nan,
                "n_source_datasets": len(sources),
                "source_datasets": "; ".join(sources),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_range_histogram(summary: pd.DataFrame, path: Path) -> None:
    """Plot the distribution of within-group QY ranges."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if summary.empty:
        ax.text(0.5, 0.5, "No repeated groups", ha="center", va="center", transform=ax.transAxes)
    else:
        bins = min(30, max(5, int(np.sqrt(len(summary)) * 2)))
        ax.hist(summary["qy_range"], bins=bins, color="#3b75af", edgecolor="white", linewidth=0.7)
    ax.axvline(0.05, color="#e69f00", linestyle="--", linewidth=1.6, label="0.05")
    ax.axvline(0.10, color="#c44e52", linestyle="--", linewidth=1.6, label="0.10")
    ax.set(xlabel="QY range within molecule–solvent group", ylabel="Number of molecule–solvent groups")
    ax.legend(frameon=False, title="Reference range")
    _style_axes(ax)
    _save_figure(fig, path)


def plot_mean_vs_range(summary: pd.DataFrame, path: Path) -> None:
    """Plot group mean QY against within-group QY range."""
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    if summary.empty:
        ax.text(0.5, 0.5, "No repeated groups", ha="center", va="center", transform=ax.transAxes)
    else:
        sizes = 18 + 7 * np.sqrt(summary["n_qy_records"])
        ax.scatter(summary["mean_qy"], summary["qy_range"], s=sizes, alpha=0.72,
                   color="#2a788e", edgecolors="white", linewidths=0.45)
    ax.set(xlabel="Mean quantum yield", ylabel="QY range within group", xlim=(-0.02, 1.02))
    _style_axes(ax)
    _save_figure(fig, path)


def _shorten(value: object, length: int = 25) -> str:
    text = str(value)
    return text if len(text) <= length else f"{text[: length - 1]}…"


def plot_top_groups(summary: pd.DataFrame, path: Path, top_n: int) -> None:
    """Plot a horizontal ranking of groups with the largest QY ranges."""
    top = summary.nlargest(top_n, "qy_range").sort_values("qy_range") if not summary.empty else summary
    fig_height = max(4.2, 0.34 * max(len(top), 1) + 1.4)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    if top.empty:
        ax.text(0.5, 0.5, "No repeated groups", ha="center", va="center", transform=ax.transAxes)
    else:
        labels = []
        for _, row in top.iterrows():
            molecule = _shorten(row[MOLECULE_COLUMN], 28)
            solvent = _shorten(row.get(SOLVENT_COLUMN, "molecule only"), 18)
            labels.append(f"{molecule} | {solvent} (n={int(row['n_qy_records'])})")
        y = np.arange(len(top))
        ax.barh(y, top["qy_range"], color="#4477aa")
        ax.set_yticks(y, labels=labels, fontsize=8)
    ax.set_xlabel("QY range")
    _style_axes(ax)
    _save_figure(fig, path)


def plot_noise_by_source(summary: pd.DataFrame, path: Path) -> None:
    """Compare ranges in groups represented by one versus multiple sources."""
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    one = summary.loc[summary["n_source_datasets"] == 1, "qy_range"].to_numpy()
    multiple = summary.loc[summary["n_source_datasets"] > 1, "qy_range"].to_numpy()
    available = [("One source", one), ("Multiple sources", multiple)]
    present = [(label, values) for label, values in available if len(values)]
    if not present:
        ax.text(0.5, 0.5, "No repeated groups", ha="center", va="center", transform=ax.transAxes)
    else:
        labels, values = zip(*present)
        boxes = ax.boxplot(values, patch_artist=True, widths=0.55, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1), labels=labels)
        for box in boxes["boxes"]:
            box.set(facecolor="#6baed6", alpha=0.7)
        rng = np.random.default_rng(42)
        for position, observations in enumerate(values, start=1):
            jitter = rng.normal(0, 0.035, len(observations))
            ax.scatter(position + jitter, observations, s=17, color="#24557a", alpha=0.55)
    ax.set_ylabel("QY range within group")
    _style_axes(ax)
    _save_figure(fig, path)


def _percent(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def write_report(
    path: Path,
    total_qy_rows: int,
    valid_rows: pd.DataFrame,
    all_groups: pd.DataFrame,
    repeated: pd.DataFrame,
    group_columns: list[str],
) -> None:
    """Write citable descriptive statistics and the ten noisiest groups."""
    repeated_rows = int(repeated["n_qy_records"].sum()) if not repeated.empty else 0
    median_range = repeated["qy_range"].median() if not repeated.empty else np.nan
    mean_range = repeated["qy_range"].mean() if not repeated.empty else np.nan
    lines = [
        "# Quantum-yield repeated-measurement noise",
        "",
        "## Summary",
        "",
        f"- Total QY rows (non-null numeric values): **{total_qy_rows:,}**",
        f"- Physically valid QY rows (0–1): **{len(valid_rows):,}**",
        f"- Total molecule-solvent groups with valid QY: **{len(all_groups):,}**",
        f"- Repeated molecule-solvent groups: **{len(repeated):,}**",
        f"- QY rows involved in repeated groups: **{repeated_rows:,} ({_percent(repeated_rows, len(valid_rows)):.1f}%)**",
        f"- Median QY range among repeated groups: **{median_range:.3f}**",
        f"- Mean QY range among repeated groups: **{mean_range:.3f}**",
        f"- Repeated groups with QY range > 0.05: **{_percent((repeated['qy_range'] > 0.05).sum(), len(repeated)):.1f}%**",
        f"- Repeated groups with QY range > 0.10: **{_percent((repeated['qy_range'] > 0.10).sum(), len(repeated)):.1f}%**",
        "",
        "## Top 10 noisiest molecule-solvent groups",
        "",
        "| Rank | Molecule | Solvent | Records | QY min | QY max | QY range | Sources |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for rank, (_, row) in enumerate(repeated.nlargest(10, "qy_range").iterrows(), start=1):
        molecule = str(row[MOLECULE_COLUMN]).replace("|", "\\|")
        solvent = str(row.get(SOLVENT_COLUMN, "molecule only")).replace("|", "\\|")
        sources = str(row["source_datasets"]).replace("|", "\\|")
        lines.append(
            f"| {rank} | `{molecule}` | `{solvent}` | {int(row['n_qy_records'])} | "
            f"{row['min_qy']:.3f} | {row['max_qy']:.3f} | {row['qy_range']:.3f} | {sources} |"
        )
    if repeated.empty:
        lines.append("| – | No groups met the replicate threshold | – | – | – | – | – | – |")
    grouping = "molecule-solvent pair" if SOLVENT_COLUMN in group_columns else "molecule"
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Repeated records for the same {grouping} can show substantial variation in reported "
            "quantum yield. This experimental and label disagreement supports treating quantum-yield "
            "prediction as noisier than wavelength prediction; QY model performance and individual "
            "predictions should therefore be interpreted cautiously.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(input_path: Path, out_dir: Path, min_replicates: int = 2, top_n: int = 25) -> None:
    """Run the complete analysis and write tables, report, and figures."""
    if min_replicates < 2:
        raise ValueError("--min-replicates must be at least 2")
    if top_n < 1:
        raise ValueError("--top-n must be at least 1")
    data = pd.read_csv(input_path)
    numeric_qy = pd.to_numeric(data.get("quantum_yield"), errors="coerce")
    total_qy_rows = int(numeric_qy.notna().sum())
    valid, excluded, group_columns = prepare_qy_data(data)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not excluded.empty:
        excluded.to_csv(out_dir / "excluded_qy_values.csv", index=False)

    all_groups = compute_group_statistics(valid, group_columns)
    repeated = all_groups.loc[all_groups["n_qy_records"] >= min_replicates].copy()
    repeated = repeated.sort_values(["qy_range", "n_qy_records"], ascending=[False, False])
    repeated.to_csv(out_dir / "qy_replicate_noise_summary.csv", index=False)
    repeated.head(top_n).to_csv(out_dir / "top_noisy_qy_groups.csv", index=False)

    plot_range_histogram(repeated, out_dir / "qy_replicate_range_histogram.png")
    plot_mean_vs_range(repeated, out_dir / "qy_mean_vs_range_scatter.png")
    plot_top_groups(repeated, out_dir / "top_noisy_qy_groups_bar.png", top_n)
    plot_noise_by_source(repeated, out_dir / "qy_noise_by_source_pair.png")
    write_report(
        out_dir / "qy_noise_report.md",
        total_qy_rows,
        valid,
        all_groups,
        repeated,
        group_columns,
    )
    print(f"Wrote QY noise analysis for {len(repeated):,} repeated groups to {out_dir}")


def main() -> None:
    args = parse_args()
    run_analysis(args.input, args.out_dir, args.min_replicates, args.top_n)


if __name__ == "__main__":
    main()
