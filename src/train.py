from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import clean_data, load_raw_data
from src.evaluate import classification_metrics, regression_metrics
from src.features import build_feature_matrix_train
from src.models import (
    best_three_average,
    compare_classifiers,
    compare_regressors,
    lgbm_regressor,
    tune_lgbm_regressor,
)
from src.plots import confusion_matrix_plot, error_by_solvent, predicted_vs_actual, residuals_vs_actual
from src.splitting import get_scaffold, random_split_indices, scaffold_train_test_split
from src.utils import ensure_output_dirs, save_json, save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the original ChemFluor prediction workflow."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=(
            "ChemFluor dataset CSV. Defaults to data/chemfluor_data.csv, "
            "then chemfluor_data.csv."
        ),
    )
    parser.add_argument(
        "--solvent-descriptors",
        type=Path,
        default=None,
        help=(
            "Solvent descriptor CSV. Defaults to data/solvent_descriptors.csv, "
            "then solvent_descriptors.csv. If neither exists, a template is "
            "created at data/solvent_descriptors.csv."
        ),
    )
    return parser.parse_args()


def nm_to_ev(y):
    return 1240.0 / np.asarray(y, dtype=float)


def ev_to_nm(y):
    y = np.clip(np.asarray(y, dtype=float), 1e-6, None)
    return 1240.0 / y


def plqy_logit(y):
    y = np.clip(np.asarray(y, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(y / (1 - y))


def inv_plqy_logit(y):
    pred = 1.0 / (1.0 + np.exp(-np.asarray(y, dtype=float)))
    return np.clip(pred, 0.0, 1.0)


def save_worst_predictions(df, y_true, y_pred, value_name: str, path: Path) -> None:
    out = df[[config.SMILES_COL, "canonical_smiles", config.SOLVENT_COL]].copy()
    out[f"actual_{value_name}"] = np.asarray(y_true)
    out[f"predicted_{value_name}"] = np.asarray(y_pred)
    out["absolute_error"] = np.abs(out[f"actual_{value_name}"] - out[f"predicted_{value_name}"])
    out.sort_values("absolute_error", ascending=False).head(20).to_csv(path, index=False)


def feature_importance_csv(model, feature_names, path: Path) -> None:
    if hasattr(model, "feature_importances_"):
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=False
        ).to_csv(path, index=False)


def shap_summary_if_available(model, X, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import shap
    except Exception:
        print("[optional] shap is not installed; skipping SHAP plots.")
        return
    sample = X.sample(min(len(X), 500), random_state=config.RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    shap.summary_plot(values, sample, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def seed_ensemble(
    X_train,
    y_train,
    X_test,
    y_true_eval,
    inverse,
    value_name: str,
    meta_df: pd.DataFrame,
    models_path: Path | None = None,
) -> pd.DataFrame:
    preds = []
    models = []
    for seed in config.SEED_ENSEMBLE_SEEDS:
        print(f"Seed ensemble {value_name}: training LightGBM seed={seed}")
        model = lgbm_regressor(seed)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        preds.append(inverse(pred) if inverse else pred)
        models.append(model)
    if models_path is not None:
        save_model(models, models_path)
    pred_arr = np.vstack(preds)
    out = meta_df[[config.SMILES_COL, "canonical_smiles", config.SOLVENT_COL]].copy()
    out["actual"] = np.asarray(y_true_eval)
    out["mean_prediction"] = pred_arr.mean(axis=0)
    out["prediction_std"] = pred_arr.std(axis=0)
    out["absolute_error"] = np.abs(out["actual"] - out["mean_prediction"])
    return out


def run_split(split_name: str, df: pd.DataFrame, X: pd.DataFrame, train_idx, test_idx) -> dict:
    print(f"\n================ {split_name.upper()} SPLIT ================")
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    split_df_test = df.loc[test_idx].reset_index(drop=True)
    y_wave_train = df.loc[train_idx, config.WAVELENGTH_COL].to_numpy()
    y_wave_test = df.loc[test_idx, config.WAVELENGTH_COL].to_numpy()
    y_plqy_train = df.loc[train_idx, config.PLQY_COL].to_numpy()
    y_plqy_test = df.loc[test_idx, config.PLQY_COL].to_numpy()

    split_metrics: dict[str, object] = {}

    tuned_nm_params = tune_lgbm_regressor(X_train, y_wave_train, X_test, y_wave_test)
    direct_model = lgbm_regressor(params=tuned_nm_params)
    direct_model.fit(X_train, y_wave_train)
    direct_pred = direct_model.predict(X_test)
    split_metrics["wavelength_direct_nm"] = regression_metrics(y_wave_test, direct_pred)

    y_ev_train = nm_to_ev(y_wave_train)
    y_ev_test = nm_to_ev(y_wave_test)
    tuned_ev_params = tune_lgbm_regressor(X_train, y_ev_train, X_test, y_ev_test)
    ev_model = lgbm_regressor(params=tuned_ev_params)
    ev_model.fit(X_train, y_ev_train)
    ev_pred_nm = ev_to_nm(ev_model.predict(X_test))
    split_metrics["wavelength_ev_to_nm"] = regression_metrics(y_wave_test, ev_pred_nm)

    wave_cmp, wave_models = compare_regressors(X_train, y_wave_train, X_test, y_wave_test)
    wave_ens_pred, wave_ens_names = best_three_average(wave_cmp, wave_models, X_test)
    split_metrics["wavelength_model_comparison"] = wave_cmp.to_dict(orient="records")
    split_metrics["wavelength_best3_ensemble"] = {
        "members": wave_ens_names,
        **regression_metrics(y_wave_test, wave_ens_pred),
    }

    raw_plqy_model = lgbm_regressor()
    raw_plqy_model.fit(X_train, y_plqy_train)
    raw_plqy_pred = np.clip(raw_plqy_model.predict(X_test), 0, 1)
    split_metrics["plqy_raw"] = regression_metrics(y_plqy_test, raw_plqy_pred)

    logit_model = lgbm_regressor()
    logit_model.fit(X_train, plqy_logit(y_plqy_train))
    logit_pred = inv_plqy_logit(logit_model.predict(X_test))
    split_metrics["plqy_logit"] = regression_metrics(y_plqy_test, logit_pred)

    plqy_cmp, plqy_models = compare_regressors(X_train, y_plqy_train, X_test, y_plqy_test)
    plqy_ens_pred, plqy_ens_names = best_three_average(plqy_cmp, plqy_models, X_test)
    plqy_ens_pred = np.clip(plqy_ens_pred, 0, 1)
    split_metrics["plqy_model_comparison"] = plqy_cmp.to_dict(orient="records")
    split_metrics["plqy_best3_ensemble"] = {
        "members": plqy_ens_names,
        **regression_metrics(y_plqy_test, plqy_ens_pred),
    }

    y_cls_train = (y_plqy_train > config.BRIGHT_THRESHOLD).astype(int)
    y_cls_test = (y_plqy_test > config.BRIGHT_THRESHOLD).astype(int)
    cls_cmp, cls_models = compare_classifiers(X_train, y_cls_train, X_test, y_cls_test)
    best_cls_name = cls_cmp.iloc[0]["model"]
    cls_pred = cls_models[best_cls_name].predict(X_test)
    split_metrics["plqy_classifier_comparison"] = cls_cmp.to_dict(orient="records")
    split_metrics["plqy_classifier_best"] = {"model": best_cls_name, **classification_metrics(y_cls_test, cls_pred)}

    if split_name == "scaffold":
        predicted_vs_actual(y_wave_test, direct_pred, "Wavelength: Predicted vs Actual", config.PLOTS_DIR / "predicted_vs_actual_wavelength.png")
        residuals_vs_actual(y_wave_test, direct_pred, "Wavelength Residuals", config.PLOTS_DIR / "residuals_vs_actual_wavelength.png")
        error_by_solvent(split_df_test, y_wave_test, direct_pred, "Wavelength Error by Solvent", config.PLOTS_DIR / "error_by_solvent_wavelength.png")
        save_worst_predictions(split_df_test, y_wave_test, direct_pred, "wavelength_nm", config.PLOTS_DIR / "worst_20_wavelength_predictions.csv")

        predicted_vs_actual(y_plqy_test, raw_plqy_pred, "PLQY: Predicted vs Actual", config.PLOTS_DIR / "predicted_vs_actual_plqy.png")
        residuals_vs_actual(y_plqy_test, raw_plqy_pred, "PLQY Residuals", config.PLOTS_DIR / "residuals_vs_actual_plqy.png")
        error_by_solvent(split_df_test, y_plqy_test, raw_plqy_pred, "PLQY Error by Solvent", config.PLOTS_DIR / "error_by_solvent_plqy.png")
        save_worst_predictions(split_df_test, y_plqy_test, raw_plqy_pred, "plqy", config.PLOTS_DIR / "worst_20_plqy_predictions.csv")
        confusion_matrix_plot(split_metrics["plqy_classifier_best"]["confusion_matrix"], config.PLOTS_DIR / "confusion_matrix_plqy_classifier.png")

        feature_importance_csv(direct_model, X.columns, config.METRICS_DIR / "wavelength_feature_importance.csv")
        feature_importance_csv(raw_plqy_model, X.columns, config.METRICS_DIR / "plqy_feature_importance.csv")
        shap_summary_if_available(direct_model, X_test, config.PLOTS_DIR / "wavelength_shap_summary.png")
        shap_summary_if_available(raw_plqy_model, X_test, config.PLOTS_DIR / "plqy_shap_summary.png")

        seed_ensemble(
            X_train,
            y_wave_train,
            X_test,
            y_wave_test,
            None,
            "wavelength",
            split_df_test,
            config.MODEL_DIR / "wavelength_seed_models.pkl",
        ).to_csv(
            config.METRICS_DIR / "wavelength_uncertainty.csv", index=False
        )
        seed_ensemble(
            X_train,
            y_plqy_train,
            X_test,
            y_plqy_test,
            None,
            "plqy",
            split_df_test,
            config.MODEL_DIR / "plqy_seed_models.pkl",
        ).to_csv(
            config.METRICS_DIR / "plqy_uncertainty.csv", index=False
        )
        save_model(direct_model, config.MODEL_DIR / "best_wavelength_lightgbm.pkl")
        save_model(raw_plqy_model, config.MODEL_DIR / "best_plqy_lightgbm.pkl")
        save_model(cls_models[best_cls_name], config.MODEL_DIR / "best_plqy_classifier.pkl")

    split_metrics["best_models"] = {
        "wavelength": wave_cmp.iloc[0]["model"],
        "plqy_regression": plqy_cmp.iloc[0]["model"],
        "plqy_classification": best_cls_name,
    }
    return split_metrics


def flatten_metrics(metrics: dict) -> pd.DataFrame:
    rows = []
    for split, split_data in metrics.items():
        if split == "dataset":
            continue
        for task, values in split_data.items():
            if isinstance(values, dict):
                row = {"split": split, "task": task}
                for key, value in values.items():
                    if isinstance(value, (int, float, str)):
                        row[key] = value
                rows.append(row)
    return pd.DataFrame(rows)


def print_summary(metrics: dict) -> None:
    ds = metrics["dataset"]
    random = metrics["random"]
    scaffold = metrics["scaffold"]
    print("\n================ ChemFluor Results Summary ================")
    print("\nPrevious baseline:")
    print(f"- Wavelength MAE: ~{config.BASELINE_RESULTS['wavelength_mae_nm']} nm")
    print(f"- PLQY regression MAE: ~{config.BASELINE_RESULTS['plqy_regression_mae']}")
    print(f"- PLQY classifier accuracy: ~{config.BASELINE_RESULTS['plqy_classifier_accuracy']:.2%}")
    print("\nDataset:")
    print(f"- Raw rows: {ds['raw_rows']}")
    print(f"- Cleaned rows: {ds['cleaned_rows']}")
    print(f"- Unique molecules: {ds['unique_molecules']}")
    print(f"- Unique solvents: {ds['unique_solvents']}")
    print(f"- Unique scaffolds: {ds['unique_scaffolds']}")
    for name, block in [("Random split", random), ("Scaffold split", scaffold)]:
        print(f"\n{name}:")
        print(f"- Best wavelength direct nm MAE: {block['wavelength_direct_nm']['MAE']:.4f}")
        print(f"- Best wavelength eV-to-nm MAE: {block['wavelength_ev_to_nm']['MAE']:.4f}")
        print(f"- Best PLQY raw MAE: {block['plqy_raw']['MAE']:.4f}")
        print(f"- Best PLQY logit MAE: {block['plqy_logit']['MAE']:.4f}")
        cls = block["plqy_classifier_best"]
        print(f"- Best PLQY classifier accuracy/F1: {cls['accuracy']:.4f}/{cls['F1']:.4f}")
    print("\nBest models:")
    print(f"- Wavelength: {scaffold['best_models']['wavelength']}")
    print(f"- PLQY regression: {scaffold['best_models']['plqy_regression']}")
    print(f"- PLQY classification: {scaffold['best_models']['plqy_classification']}")
    print("\nSaved outputs:")
    print(f"- metrics: {config.METRICS_DIR}")
    print(f"- plots: {config.PLOTS_DIR}")
    print(f"- models: {config.MODEL_DIR}")
    print("\nShort read: scaffold split is the honest score. Accuracy gains usually come from the richer descriptor set, solvent descriptors when filled, and choosing the target transform that best matches the physics/noise of each property.")


def training_morgan_fingerprints(canonical_smiles: pd.Series) -> list:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=config.MORGAN_RADIUS, fpSize=config.MORGAN_BITS)
    fps = []
    for smi in canonical_smiles:
        mol = Chem.MolFromSmiles(str(smi))
        fps.append(gen.GetFingerprint(mol) if mol is not None else None)
    return fps


def save_inference_metadata(df: pd.DataFrame, X: pd.DataFrame, feature_artifacts: dict, metrics: dict | None = None) -> None:
    keep_cols = [
        config.SMILES_COL,
        "canonical_smiles",
        config.SOLVENT_COL,
        config.WAVELENGTH_COL,
        config.PLQY_COL,
        "scaffold",
    ]
    metadata = {
        "feature_columns": X.columns.tolist(),
        "cleaned_training_df": df[keep_cols].copy(),
        "training_morgan_fingerprints": training_morgan_fingerprints(df["canonical_smiles"]),
        "training_canonical_smiles": df["canonical_smiles"].tolist(),
        "training_scaffolds": sorted(df["scaffold"].fillna("").unique().tolist()),
        "known_solvents": sorted(df[config.SOLVENT_COL].astype(str).str.strip().unique().tolist()),
        "solvent_descriptor_column_names": feature_artifacts.get("solvent", {}).get("solvent_descriptor_columns", []),
        "plqy_bright_threshold": config.BRIGHT_THRESHOLD,
        "model_result_summary": metrics,
        "morgan_radius": config.MORGAN_RADIUS,
        "morgan_bits": config.MORGAN_BITS,
    }
    save_model(metadata, config.MODEL_DIR / "inference_metadata.pkl")


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    raw = load_raw_data(args.data_path)
    df, dataset_stats = clean_data(raw)
    df = df.reset_index(drop=True)
    df["scaffold"] = df["canonical_smiles"].map(get_scaffold)
    dataset_stats["unique_scaffolds"] = df["scaffold"].nunique()
    X, feature_artifacts = build_feature_matrix_train(
        df, solvent_descriptor_path=args.solvent_descriptors
    )
    save_model(feature_artifacts, config.MODEL_DIR / "feature_artifacts.pkl")
    save_inference_metadata(df, X, feature_artifacts)

    random_train, random_test = random_split_indices(df)
    scaffold_train, scaffold_test = scaffold_train_test_split(df)

    metrics = {
        "dataset": dataset_stats,
        "random": run_split("random", df, X, random_train, random_test),
        "scaffold": run_split("scaffold", df, X, scaffold_train, scaffold_test),
    }
    save_json(metrics, config.METRICS_DIR / "metrics.json")
    flatten_metrics(metrics).to_csv(config.METRICS_DIR / "metrics.csv", index=False)
    save_inference_metadata(df, X, feature_artifacts, metrics)
    print_summary(metrics)


if __name__ == "__main__":
    main()
