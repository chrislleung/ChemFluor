from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from . import config


def predicted_vs_actual(y_true, y_pred, title: str, path):
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def residuals_vs_actual(y_true, y_pred, title: str, path):
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, residuals, alpha=0.7)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def error_by_solvent(df: pd.DataFrame, y_true, y_pred, title: str, path):
    tmp = pd.DataFrame(
        {
            "solvent": df[config.SOLVENT_COL].to_numpy(),
            "absolute_error": np.abs(np.asarray(y_true) - np.asarray(y_pred)),
        }
    )
    grouped = tmp.groupby("solvent")["absolute_error"].mean().sort_values(ascending=False).head(25)
    if grouped.empty:
        return
    plt.figure(figsize=(10, 6))
    grouped.iloc[::-1].plot(kind="barh")
    plt.xlabel("Mean absolute error")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def confusion_matrix_plot(cm, path, title: str = "PLQY Bright/Dim Confusion Matrix"):
    disp = ConfusionMatrixDisplay(confusion_matrix=np.asarray(cm), display_labels=["dim", "bright"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

