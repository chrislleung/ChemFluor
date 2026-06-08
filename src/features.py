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
    solvent_df[config.SOLVENT_COL] = solvent_df[config.SOLVENT_COL].astype(str).str.strip()
    duplicate_solvents = solvent_df[config.SOLVENT_COL].duplicated().sum()
    if duplicate_solvents:
        print(f"Warning: solvent descriptor file has {duplicate_solvents} duplicate solvent rows; keeping the first.")
        solvent_df = solvent_df.drop_duplicates(subset=[config.SOLVENT_COL], keep="first")
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


def _normalized_solvent_series(df: pd.DataFrame) -> pd.Series:
    return df[config.SOLVENT_COL].astype(str).str.strip()


def solvent_features_train(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df[config.SOLVENT_COL] = _normalized_solvent_series(df)
    one_hot = pd.get_dummies(df[config.SOLVENT_COL], prefix="solvent", dtype=int)
    solvent_df, has_real_descriptors = load_or_create_solvent_descriptors(df)
    artifact = {
        "known_solvents": sorted(df[config.SOLVENT_COL].dropna().unique().tolist()),
        "one_hot_columns": one_hot.columns.tolist(),
        "solvent_descriptor_columns": [],
        "prefixed_solvent_descriptor_columns": [],
        "solvent_descriptor_table": solvent_df,
        "solvent_descriptor_imputer": None,
        "has_solvent_descriptors": bool(has_real_descriptors and solvent_df is not None),
    }
    if not has_real_descriptors or solvent_df is None:
        return one_hot, artifact

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
        artifact["has_solvent_descriptors"] = False
        return one_hot, artifact
    imputer = SimpleImputer(strategy="median")
    desc = pd.DataFrame(imputer.fit_transform(merged[descriptor_cols]), columns=[f"solvdesc_{c}" for c in descriptor_cols])
    artifact.update(
        {
            "solvent_descriptor_columns": descriptor_cols,
            "prefixed_solvent_descriptor_columns": desc.columns.tolist(),
            "solvent_descriptor_imputer": imputer,
            "has_solvent_descriptors": True,
        }
    )
    return pd.concat([one_hot.reset_index(drop=True), desc.reset_index(drop=True)], axis=1), artifact


def solvent_features_inference(df: pd.DataFrame, solvent_artifact: dict) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df[config.SOLVENT_COL] = _normalized_solvent_series(df)
    raw_one_hot = pd.get_dummies(df[config.SOLVENT_COL], prefix="solvent", dtype=int)
    one_hot = pd.DataFrame(0, index=df.index, columns=solvent_artifact.get("one_hot_columns", []), dtype=int)
    for col in raw_one_hot.columns:
        if col in one_hot.columns:
            one_hot[col] = raw_one_hot[col].to_numpy()

    status = {"missing_descriptor_solvents": [], "solvent_descriptors_imputed": False}
    if not solvent_artifact.get("has_solvent_descriptors"):
        return one_hot.reset_index(drop=True), status

    solvent_df = solvent_artifact.get("solvent_descriptor_table")
    descriptor_cols = solvent_artifact.get("solvent_descriptor_columns", [])
    imputer = solvent_artifact.get("solvent_descriptor_imputer")
    if solvent_df is None or not descriptor_cols or imputer is None:
        return one_hot.reset_index(drop=True), status

    merged = df[[config.SOLVENT_COL]].merge(solvent_df, on=config.SOLVENT_COL, how="left")
    missing_rows = merged[descriptor_cols].isna().any(axis=1)
    if missing_rows.any():
        status["missing_descriptor_solvents"] = sorted(df.loc[missing_rows, config.SOLVENT_COL].unique().tolist())
        status["solvent_descriptors_imputed"] = True
    desc = pd.DataFrame(
        imputer.transform(merged[descriptor_cols]),
        columns=solvent_artifact.get("prefixed_solvent_descriptor_columns", [f"solvdesc_{c}" for c in descriptor_cols]),
    )
    return pd.concat([one_hot.reset_index(drop=True), desc.reset_index(drop=True)], axis=1), status


def build_feature_matrix_train(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("Building features: Morgan fingerprints, MACCS keys, RDKit descriptors, and solvent features.")
    df = df.copy()
    df[config.SOLVENT_COL] = _normalized_solvent_series(df)
    mols = mols_from_smiles(df["canonical_smiles"])
    solv, solvent_artifact = solvent_features_train(df)
    parts = [
        morgan_fingerprints(mols),
        maccs_keys(mols),
        rdkit_descriptors(mols),
        solv,
    ]
    X_raw = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    X_raw = sanitize_feature_columns(X_raw)
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)
    artifacts = {
        "feature_columns": X.columns.tolist(),
        "final_imputer": imputer,
        "solvent": solvent_artifact,
        "morgan_radius": config.MORGAN_RADIUS,
        "morgan_bits": config.MORGAN_BITS,
    }
    print(f"Feature matrix shape: {X.shape}")
    return X, artifacts


def build_feature_matrix_inference(df: pd.DataFrame, feature_artifacts: dict) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df[config.SOLVENT_COL] = _normalized_solvent_series(df)
    mols = mols_from_smiles(df["canonical_smiles"])
    solv, status = solvent_features_inference(df, feature_artifacts.get("solvent", {}))
    parts = [
        morgan_fingerprints(mols),
        maccs_keys(mols),
        rdkit_descriptors(mols),
        solv,
    ]
    X_raw = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    X_raw = sanitize_feature_columns(X_raw)
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)

    feature_columns = feature_artifacts["feature_columns"]
    X_aligned = X_raw.reindex(columns=feature_columns, fill_value=0)
    X = pd.DataFrame(feature_artifacts["final_imputer"].transform(X_aligned), columns=feature_columns)
    status["added_missing_columns"] = int(len(set(feature_columns) - set(X_raw.columns)))
    status["dropped_extra_columns"] = int(len(set(X_raw.columns) - set(feature_columns)))
    return X, status


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X, _ = build_feature_matrix_train(df)
    return X
