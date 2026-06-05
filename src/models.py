from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from . import config
from .evaluate import classification_metrics, regression_metrics


def lgbm_regressor(random_state: int = config.RANDOM_STATE, params: dict | None = None) -> LGBMRegressor:
    base = {
        "n_estimators": 700,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "verbose": -1,
    }
    if params:
        base.update(params)
    return LGBMRegressor(**base)


def lgbm_classifier(random_state: int = config.RANDOM_STATE) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        verbose=-1,
    )


def available_regressors(random_state: int = config.RANDOM_STATE) -> dict[str, object]:
    models: dict[str, object] = {
        "LightGBM": lgbm_regressor(random_state),
        "RandomForest": RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=500, random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
        "SVR": Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=10.0, epsilon=0.1))]),
    }
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.9,
            colsample_bytree=0.9, random_state=random_state, objective="reg:squarederror", n_jobs=-1
        )
    except Exception:
        print("[optional] xgboost is not installed; skipping XGBoost regressors.")
    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            iterations=600, learning_rate=0.03, depth=6, random_seed=random_state, verbose=False
        )
    except Exception:
        print("[optional] catboost is not installed; skipping CatBoost regressors.")
    return models


def available_classifiers(random_state: int = config.RANDOM_STATE) -> dict[str, object]:
    models: dict[str, object] = {
        "LightGBM": lgbm_classifier(random_state),
        "RandomForest": RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=500, random_state=random_state, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.9,
            colsample_bytree=0.9, random_state=random_state, eval_metric="logloss", n_jobs=-1
        )
    except Exception:
        print("[optional] xgboost is not installed; skipping XGBoost classifiers.")
    try:
        from catboost import CatBoostClassifier

        models["CatBoost"] = CatBoostClassifier(
            iterations=600, learning_rate=0.03, depth=6, random_seed=random_state, verbose=False
        )
    except Exception:
        print("[optional] catboost is not installed; skipping CatBoost classifiers.")
    return models


def tune_lgbm_regressor(X_train, y_train, X_valid, y_valid) -> dict:
    if not config.USE_OPTUNA:
        return {}
    try:
        import optuna
        from sklearn.metrics import mean_absolute_error
    except Exception:
        print("[optional] optuna is not installed; using default LightGBM parameters.")
        return {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", -1, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = lgbm_regressor(config.RANDOM_STATE, params)
        model.fit(X_train, y_train)
        return mean_absolute_error(y_valid, model.predict(X_valid))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config.N_OPTUNA_TRIALS)
    print(f"Best Optuna LightGBM MAE: {study.best_value:.4f}")
    return study.best_params


def compare_regressors(X_train, y_train, X_test, y_test, inverse: Callable | None = None):
    results = []
    fitted = {}
    true_eval = inverse(y_test) if inverse else y_test
    for name, model in available_regressors().items():
        print(f"Training regressor: {name}")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred_eval = inverse(pred) if inverse else pred
        metrics = regression_metrics(true_eval, pred_eval)
        results.append({"model": name, **metrics})
        fitted[name] = model
        print(f"  {name} MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
    results_df = pd.DataFrame(results).sort_values("MAE")
    return results_df, fitted


def best_three_average(results_df: pd.DataFrame, fitted: dict[str, object], X_test, inverse: Callable | None = None):
    names = results_df.head(3)["model"].tolist()
    preds = np.vstack([fitted[name].predict(X_test) for name in names])
    pred = preds.mean(axis=0)
    return inverse(pred) if inverse else pred, names


def compare_classifiers(X_train, y_train, X_test, y_test):
    rows = []
    fitted = {}
    for name, model in available_classifiers().items():
        print(f"Training classifier: {name}")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = classification_metrics(y_test, pred)
        rows.append({"model": name, **metrics})
        fitted[name] = model
        print(f"  {name} accuracy={metrics['accuracy']:.4f}, F1={metrics['F1']:.4f}")
    return pd.DataFrame(rows).sort_values("F1", ascending=False), fitted

