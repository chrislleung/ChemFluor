import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config


def ensure_output_dirs() -> None:
    for path in [config.OUTPUT_DIR, config.MODEL_DIR, config.METRICS_DIR, config.PLOTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def sanitize_feature_name(name: object) -> str:
    text = str(name)
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "feature"


def sanitize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: dict[str, int] = {}
    names: list[str] = []
    for col in df.columns:
        clean = sanitize_feature_name(col)
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        names.append(clean)
    out = df.copy()
    out.columns = names
    return out


def save_json(data: dict[str, Any], path: Path) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(data), indent=2), encoding="utf-8")


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)


def optional_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        print(f"[optional] {module_name} is not installed; skipping related models/features.")
        return None

