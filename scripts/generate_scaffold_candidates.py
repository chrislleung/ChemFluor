"""Rule-based scaffold enumeration for simple fluorophore candidates.

This script uses fixed SMILES templates and substituent fragments. It is not
neural generation.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from rdkit import Chem, RDLogger
except ImportError as exc:  # pragma: no cover - only used when RDKit is missing.
    Chem = None
    RDLogger = None
    _RDKIT_IMPORT_ERROR = exc
else:
    _RDKIT_IMPORT_ERROR = None
    RDLogger.DisableLog("rdApp.*")


DEFAULT_OUTPUT_PATH = Path("data/generated_candidates/scaffold_candidates.csv")

SUBSTITUENTS = {
    "H": "[H]",
    "methyl": "C",
    "ethyl": "CC",
    "methoxy": "OC",
    "ethoxy": "OCC",
    "dimethylamino": "N(C)C",
    "diethylamino": "N(CC)CC",
    "cyano": "C#N",
    "fluoro": "F",
    "chloro": "Cl",
    "trifluoromethyl": "C(F)(F)F",
    "phenyl": "c1ccccc1",
}

SCAFFOLD_TEMPLATES = {
    "coumarin_7_substituted": "O=c1oc2ccc({sub})cc2cc1",
    "coumarin_6_substituted": "O=c1oc2cc({sub})ccc2cc1",
    "coumarin_4_methyl_7_substituted": "Cc1cc(=O)oc2ccc({sub})cc12",
    "naphthalimide_4_substituted": "O=C1N(C)C(=O)c2ccc({sub})cc21",
    "naphthalimide_4_substituted_n_butyl": "CCCCN1C(=O)c2ccc({sub})cc2C1=O",
}


@dataclass(frozen=True)
class GenerationStats:
    """Summary counts for a scaffold enumeration run."""

    scaffold_templates_used: int
    substituents_used: int
    raw_combinations_attempted: int
    unique_valid_molecules_saved: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rule-based enumeration of coumarin-like and naphthalimide-like candidates."
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT_PATH,
        type=Path,
        help=f"Output CSV path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--scaffolds",
        choices=["coumarin", "naphthalimide", "all"],
        default="all",
        help="Scaffold family to enumerate. Defaults to all.",
    )
    parser.add_argument(
        "--substituents",
        default=",".join(SUBSTITUENTS.keys()),
        help=(
            "Comma-separated substituent names to use. Defaults to the full built-in "
            f"set: {', '.join(SUBSTITUENTS.keys())}."
        ),
    )
    return parser.parse_args()


def require_rdkit() -> None:
    """Raise a helpful error if RDKit is unavailable."""
    if Chem is None:
        raise ImportError("RDKit is required to generate scaffold candidates.") from _RDKIT_IMPORT_ERROR


def canonicalize_smiles(smiles: str) -> str | None:
    """Return canonical SMILES, or None for invalid molecules."""
    require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def select_scaffold_templates(scaffold_choice: str) -> dict[str, str]:
    """Select scaffold templates by family."""
    if scaffold_choice == "all":
        return dict(SCAFFOLD_TEMPLATES)
    return {
        name: template
        for name, template in SCAFFOLD_TEMPLATES.items()
        if name.startswith(scaffold_choice)
    }


def select_substituents(substituent_text: str) -> dict[str, str]:
    """Select substituents from a comma-separated CLI value."""
    names = [name.strip() for name in substituent_text.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one substituent name must be provided.")

    unknown = [name for name in names if name not in SUBSTITUENTS]
    if unknown:
        valid = ", ".join(SUBSTITUENTS.keys())
        raise ValueError(
            f"Unknown substituent(s): {', '.join(unknown)}. Valid choices: {valid}"
        )

    return {name: SUBSTITUENTS[name] for name in names}


def generate_candidates(
    scaffold_templates: dict[str, str],
    substituents: dict[str, str],
) -> tuple[pd.DataFrame, GenerationStats]:
    """Enumerate template/substituent combinations and remove duplicates."""
    records: list[dict[str, str]] = []
    attempted = 0

    for scaffold_name, template in scaffold_templates.items():
        for substituent_name, substituent_smiles in substituents.items():
            attempted += 1
            smiles = template.format(sub=substituent_smiles)
            canonical = canonicalize_smiles(smiles)
            if canonical is None:
                continue
            records.append(
                {
                    "name": f"{scaffold_name}_{substituent_name}",
                    "scaffold": scaffold_name,
                    "substituent": substituent_name,
                    "smiles": smiles,
                    "canonical_smiles": canonical,
                }
            )

    candidates = pd.DataFrame.from_records(records)
    if candidates.empty:
        unique_candidates = pd.DataFrame(
            columns=["name", "scaffold", "substituent", "smiles", "canonical_smiles"]
        )
    else:
        unique_candidates = candidates.drop_duplicates(
            subset=["canonical_smiles"]
        ).reset_index(drop=True)

    stats = GenerationStats(
        scaffold_templates_used=len(scaffold_templates),
        substituents_used=len(substituents),
        raw_combinations_attempted=attempted,
        unique_valid_molecules_saved=len(unique_candidates),
    )
    return unique_candidates, stats


def main() -> int:
    """Generate and save scaffold candidates."""
    try:
        args = parse_args()
        scaffold_templates = select_scaffold_templates(args.scaffolds)
        substituents = select_substituents(args.substituents)
        candidates, stats = generate_candidates(scaffold_templates, substituents)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(args.out, index=False)

        print(f"Scaffold templates used: {stats.scaffold_templates_used}")
        print(f"Substituents used: {stats.substituents_used}")
        print(f"Raw combinations attempted: {stats.raw_combinations_attempted}")
        print(f"Unique valid molecules saved: {stats.unique_valid_molecules_saved}")
        print(f"Saved candidates to: {args.out}")
        return 0
    except (ImportError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
