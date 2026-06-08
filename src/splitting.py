from __future__ import annotations

import random

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

from . import config


def get_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def random_split_indices(df: pd.DataFrame, test_size: float = config.TEST_SIZE, random_state: int = config.RANDOM_STATE):
    train_idx, test_idx = train_test_split(
        df.index.to_numpy(), test_size=test_size, random_state=random_state
    )
    print(f"Random split sizes: train={len(train_idx)}, test={len(test_idx)}")
    return train_idx, test_idx


def scaffold_train_test_split(
    df: pd.DataFrame, test_size: float = config.TEST_SIZE, random_state: int = config.RANDOM_STATE
):
    scaffolds: dict[str, list[int]] = {}
    for idx, smi in df["canonical_smiles"].items():
        scaffolds.setdefault(get_scaffold(smi), []).append(idx)

    groups = list(scaffolds.values())
    rng = random.Random(random_state)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    target_test = max(1, int(round(len(df) * test_size)))
    train_idx: list[int] = []
    test_idx: list[int] = []
    for group in groups:
        if len(test_idx) < target_test:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    train_scaffolds = {get_scaffold(df.loc[i, "canonical_smiles"]) for i in train_idx}
    test_scaffolds = {get_scaffold(df.loc[i, "canonical_smiles"]) for i in test_idx}
    leakage = train_scaffolds.intersection(test_scaffolds)
    if leakage:
        raise RuntimeError(f"Scaffold leakage detected: {len(leakage)} overlapping scaffolds.")

    print(
        "Scaffold split sizes: "
        f"train={len(train_idx)}, test={len(test_idx)}, unique_scaffolds={len(scaffolds)}"
    )
    return train_idx, test_idx

