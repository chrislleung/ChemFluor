from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
BRIGHT_THRESHOLD = 0.25

USE_OPTUNA = False
N_OPTUNA_TRIALS = 50

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
SEED_ENSEMBLE_SEEDS = [0, 1, 2, 3, 4, 42, 100, 123]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHEMFLUOR_DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "chemfluor_data.csv",
    PROJECT_ROOT / "chemfluor_data.csv",
]
DEFAULT_SOLVENT_DESCRIPTOR_CANDIDATES = [
    PROJECT_ROOT / "data" / "solvent_descriptors.csv",
    PROJECT_ROOT / "solvent_descriptors.csv",
]
DATA_PATH = DEFAULT_CHEMFLUOR_DATA_CANDIDATES[0]
SOLVENT_DESCRIPTOR_PATH = DEFAULT_SOLVENT_DESCRIPTOR_CANDIDATES[0]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"

SMILES_COL = "SMILES"
SOLVENT_COL = "solvent"
WAVELENGTH_COL = "Emission/nm"
PLQY_COL = "PLQY"

BASELINE_RESULTS = {
    "wavelength_mae_nm": 21.99,
    "plqy_regression_mae": 0.1339,
    "plqy_classifier_accuracy": 0.8269,
}


def resolve_existing_path(
    explicit_path: str | Path | None,
    fallback_paths: list[Path],
    description: str,
) -> Path:
    """Resolve an explicit path or the first existing fallback path."""
    if explicit_path is not None:
        path = Path(explicit_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"{description} not found at explicit path: {path}")

    for path in fallback_paths:
        if path.exists():
            return path

    searched = "\n".join(f"- {path}" for path in fallback_paths)
    raise FileNotFoundError(
        f"{description} not found. Searched these locations:\n{searched}"
    )


def resolve_chemfluor_data_path(explicit_path: str | Path | None = None) -> Path:
    """Resolve the original ChemFluor dataset path."""
    return resolve_existing_path(
        explicit_path=explicit_path,
        fallback_paths=DEFAULT_CHEMFLUOR_DATA_CANDIDATES,
        description="ChemFluor dataset",
    )


def resolve_solvent_descriptor_path(
    explicit_path: str | Path | None = None,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve the original solvent descriptor path."""
    if must_exist:
        return resolve_existing_path(
            explicit_path=explicit_path,
            fallback_paths=DEFAULT_SOLVENT_DESCRIPTOR_CANDIDATES,
            description="Solvent descriptor file",
        )
    if explicit_path is not None:
        path = Path(explicit_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Solvent descriptor file not found at explicit path: {path}")
    for path in DEFAULT_SOLVENT_DESCRIPTOR_CANDIDATES:
        if path.exists():
            return path
    return DEFAULT_SOLVENT_DESCRIPTOR_CANDIDATES[0]
