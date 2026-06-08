from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from . import config


@dataclass
class MoleculeIdentity:
    mol: Chem.Mol
    canonical_smiles: str
    scaffold: str


def canonicalize_input_molecule(smiles: str) -> MoleculeIdentity:
    text = str(smiles).strip()
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}. Please check the molecule string.")
    canonical = Chem.MolToSmiles(mol, canonical=True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return MoleculeIdentity(mol=mol, canonical_smiles=canonical, scaffold=scaffold)


def morgan_fingerprint(mol: Chem.Mol, metadata: dict | None = None):
    metadata = metadata or {}
    radius = int(metadata.get("morgan_radius", config.MORGAN_RADIUS))
    bits = int(metadata.get("morgan_bits", config.MORGAN_BITS))
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    return gen.GetFingerprint(mol)


def nearest_training_molecules(mol: Chem.Mol, metadata: dict, top_n: int = 5) -> dict:
    query_fp = morgan_fingerprint(mol, metadata)
    training_fps = [fp for fp in metadata.get("training_morgan_fingerprints", []) if fp is not None]
    if not training_fps:
        return {"max_similarity": 0.0, "top5_average_similarity": 0.0, "nearest_neighbors": []}

    similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(query_fp, training_fps), dtype=float)
    order = np.argsort(similarities)[::-1][:top_n]
    train_df = metadata["cleaned_training_df"].reset_index(drop=True)
    neighbors = []
    for idx in order:
        row = train_df.iloc[int(idx)]
        neighbors.append(
            {
                "canonical_smiles": row.get("canonical_smiles"),
                "solvent": row.get(config.SOLVENT_COL),
                "emission_nm": float(row.get(config.WAVELENGTH_COL)),
                "plqy": float(row.get(config.PLQY_COL)),
                "similarity": float(similarities[idx]),
            }
        )
    top_avg = float(similarities[order].mean()) if len(order) else 0.0
    return {
        "max_similarity": float(similarities.max()) if len(similarities) else 0.0,
        "top5_average_similarity": top_avg,
        "nearest_neighbors": neighbors,
    }


def scaffold_novelty(scaffold: str, metadata: dict) -> dict:
    scaffolds = set(metadata.get("training_scaffolds", []))
    return {"scaffold_seen": scaffold in scaffolds, "scaffold": scaffold}


def solvent_novelty(solvent: str, metadata: dict) -> dict:
    known = {str(s).strip() for s in metadata.get("known_solvents", [])}
    clean_solvent = str(solvent).strip()
    return {"solvent_seen": clean_solvent in known, "solvent": clean_solvent}


def confidence_assessment(
    max_similarity: float,
    scaffold_seen: bool,
    solvent_seen: bool,
    descriptors_missing: bool = False,
    uncertainty_caution: str | None = None,
) -> dict:
    domain_score = 0.6 * float(max_similarity) + 0.25 * float(scaffold_seen) + 0.15 * float(solvent_seen)
    domain_score = float(np.clip(domain_score, 0.0, 1.0))

    if scaffold_seen and max_similarity >= 0.60 and solvent_seen and not descriptors_missing:
        level = "High"
        warning = "High confidence: this molecule, scaffold, and solvent are well represented in the training domain."
    elif max_similarity >= 0.40 and solvent_seen and not descriptors_missing:
        level = "Medium"
        if scaffold_seen:
            warning = "Medium confidence: this molecule is moderately similar to training molecules."
        else:
            warning = (
                "Medium confidence: this molecule is moderately similar to training molecules, "
                "but its scaffold was not seen during training."
            )
    else:
        level = "Low"
        reasons = []
        if max_similarity < 0.40:
            reasons.append("low similarity to the training set")
        if not scaffold_seen:
            reasons.append("a new scaffold")
        if not solvent_seen:
            reasons.append("an unknown solvent")
        if descriptors_missing:
            reasons.append("missing solvent descriptors that were median-imputed")
        detail = " and ".join(reasons) if reasons else "limited training-domain support"
        warning = f"Low confidence: this molecule has {detail}. Treat the prediction as a rough estimate."

    if uncertainty_caution:
        warning = f"{warning} {uncertainty_caution}"
        if level == "High":
            level = "Medium"
        elif level == "Medium":
            level = "Low"

    return {"confidence_level": level, "domain_score": domain_score, "warning": warning}
