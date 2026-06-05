from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from sklearn.impute import SimpleImputer

from . import config
from .utils import sanitize_feature_columns


DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "TPSA": Descriptors.TPSA,
    "RingCount": Descriptors.RingCount,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "FractionCSP3": Descriptors.FractionCSP3,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NHOHCount": Descriptors.NHOHCount,
    "NOCount": Descriptors.NOCount,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "NumAliphaticRings": Descriptors.NumAliphaticRings,
    "MolMR": Descriptors.MolMR,
    "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT,
}


def mols_from_smiles(smiles: pd.Series) -> list[Chem.Mol]:
    mols = []
    bad = 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            bad += 1
        mols.append(mol)
    if bad:
        raise ValueError(f"Feature generation received {bad} invalid SMILES after cleaning.")
    return mols


def morgan_fingerprints(mols: list[Chem.Mol]) -> pd.DataFrame:
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=config.MORGAN_RADIUS, fpSize=config.MORGAN_BITS
    )
    arr = np.zeros((len(mols), config.MORGAN_BITS), dtype=np.uint8)
    for i, mol in enumerate(mols):
        fp = gen.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return pd.DataFrame(arr, columns=[f"morgan_{i}" for i in range(config.MORGAN_BITS)])


def maccs_keys(mols: list[Chem.Mol]) -> pd.DataFrame:
    # RDKit MACCS is 167 bits including bit 0. Keep all bits and name them explicitly.
    values = []
    for mol in mols:
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        values.append(arr)
    return pd.DataFrame(values, columns=[f"maccs_{i}" for i in range(len(values[0]))])


def rdkit_descriptors(mols: list[Chem.Mol]) -> pd.DataFrame:
    rows = []
    for mol in mols:
        row = {}
        for name, func in DESCRIPTOR_FUNCS.items():
            try:
                row[name] = float(func(mol))
            except Exception:
                row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_or_create_solvent_descriptors(df: pd.DataFrame) -> tuple[pd.DataFrame | None, bool]:
    path = config.SOLVENT_DESCRIPTOR_PATH
    descriptor_cols = [
        "dielectric_constant",
        "refractive_index",
        "dipole_moment",
        "hbond_donor",
        "hbond_acceptor",
        "polarity_ET30",
    ]
    if not path.exists():
        print(f"Solvent descriptor file not found. Creating template: {path}")
        template = pd.DataFrame({config.SOLVENT_COL: sorted(df[config.SOLVENT_COL].unique())})
        for col in descriptor_cols:
            template[col] = np.nan
        template.to_csv(path, index=False)
        return None, False

    solvent_df = pd.read_csv(path)
    if config.SOLVENT_COL not in solvent_df.columns:
        raise ValueError(f"{path} must contain a '{config.SOLVENT_COL}' column.")
    usable_cols = [c for c in descriptor_cols if c in solvent_df.columns]
    if not usable_cols:
        print("Solvent descriptor file exists but has no descriptor columns filled; using one-hot solvents only.")
        return None, False
    for col in usable_cols:
        solvent_df[col] = pd.to_numeric(solvent_df[col], errors="coerce")
    if solvent_df[usable_cols].isna().all().all():
        print("Solvent descriptor template is blank; using one-hot solvents only.")
        return None, False
    return solvent_df[[config.SOLVENT_COL] + usable_cols], True


def solvent_features(df: pd.DataFrame) -> pd.DataFrame:
    one_hot = pd.get_dummies(df[config.SOLVENT_COL], prefix="solvent", dtype=int)
    solvent_df, has_real_descriptors = load_or_create_solvent_descriptors(df)
    if not has_real_descriptors or solvent_df is None:
        return one_hot

    merged = df[[config.SOLVENT_COL]].merge(solvent_df, on=config.SOLVENT_COL, how="left")
    descriptor_cols = [c for c in merged.columns if c != config.SOLVENT_COL]
    missing_rows = merged[descriptor_cols].isna().any(axis=1)
    if missing_rows.any():
        missing_solvents = sorted(df.loc[missing_rows, config.SOLVENT_COL].unique())
        print(f"Warning: missing solvent descriptor values for solvents: {missing_solvents}")
    all_blank = [c for c in descriptor_cols if merged[c].isna().all()]
    if all_blank:
        print(f"Warning: dropping all-blank solvent descriptor columns: {all_blank}")
        descriptor_cols = [c for c in descriptor_cols if c not in all_blank]
    if not descriptor_cols:
        return one_hot
    imputer = SimpleImputer(strategy="median")
    desc = pd.DataFrame(imputer.fit_transform(merged[descriptor_cols]), columns=[f"solvdesc_{c}" for c in descriptor_cols])
    return pd.concat([one_hot.reset_index(drop=True), desc.reset_index(drop=True)], axis=1)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    print("Building features: Morgan fingerprints, MACCS keys, RDKit descriptors, and solvent features.")
    mols = mols_from_smiles(df["canonical_smiles"])
    parts = [
        morgan_fingerprints(mols),
        maccs_keys(mols),
        rdkit_descriptors(mols),
        solvent_features(df),
    ]
    X = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    X = sanitize_feature_columns(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
    print(f"Feature matrix shape: {X.shape}")
    return X
