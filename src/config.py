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
DATA_PATH = PROJECT_ROOT / "chemfluor_data.csv"
SOLVENT_DESCRIPTOR_PATH = PROJECT_ROOT / "solvent_descriptors.csv"
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
