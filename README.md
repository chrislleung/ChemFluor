# ChemFluor Molecular Property Pipeline

This project predicts fluorescent molecule properties from `SMILES` plus `solvent`.

Expected dataset: `chemfluor_data.csv`

Required columns:
- `SMILES`
- `solvent`
- `Emission/nm`
- `PLQY`

## What The Pipeline Does

Run training with either:

```bash
python -m src.train
```

or:

```bash
python src/train.py
```

The pipeline cleans the dataset, canonicalizes SMILES with RDKit, merges duplicate molecule-solvent pairs by averaging `Emission/nm` and `PLQY`, builds features, trains models, evaluates random and scaffold splits, and saves outputs.

## Features

Molecular features:
- Morgan fingerprints, radius 2, 2048 bits
- MACCS keys
- RDKit descriptors such as molecular weight, logP, TPSA, ring counts, aromatic rings, rotatable bonds, fraction sp3, MolMR, BalabanJ, and BertzCT

Solvent features:
- One-hot solvent identity
- Optional physical solvent descriptors from `solvent_descriptors.csv`

If `solvent_descriptors.csv` does not exist, the pipeline creates a template with all solvents found in the dataset. Fill in columns such as:

```text
dielectric_constant,refractive_index,dipole_moment,hbond_donor,hbond_acceptor,polarity_ET30
```

Blank solvent descriptors are allowed. The model will continue with one-hot solvent encoding only. Partially missing numeric values are reported and median-imputed.

## Targets

Wavelength is modeled two ways:
- Direct nanometer prediction: `Emission/nm`
- Energy-space prediction: `eV = 1240 / nm`, then predictions are converted back with `nm = 1240 / eV`

The energy model is scientifically useful because molecular emission is often more linear in energy than wavelength.

PLQY is modeled two ways:
- Raw PLQY regression
- Logit-transformed PLQY regression, which respects PLQY as a bounded 0 to 1 quantity

PLQY classification predicts bright versus dim using `PLQY > 0.25`. Change this in `src/config.py`.

## Splits

The pipeline reports both:
- Random split: useful for baseline comparison, but can overestimate performance when similar molecules appear in train and test
- Bemis-Murcko scaffold split: keeps molecular scaffolds separated and is the more honest generalization score

## Models

Regression models:
- LightGBM
- RandomForest
- ExtraTrees
- GradientBoosting
- SVR
- XGBoost, if installed
- CatBoost, if installed

Classification models:
- LightGBM
- RandomForest
- ExtraTrees
- XGBoost, if installed
- CatBoost, if installed

The best three regressors by validation MAE are also averaged as a simple ensemble.

Optional Optuna tuning is controlled in `src/config.py`:

```python
USE_OPTUNA = False
N_OPTUNA_TRIALS = 50
```

## Outputs

Saved files:
- `outputs/metrics/metrics.json`
- `outputs/metrics/metrics.csv`
- `outputs/metrics/wavelength_uncertainty.csv`
- `outputs/metrics/plqy_uncertainty.csv`
- `outputs/metrics/wavelength_feature_importance.csv`
- `outputs/metrics/plqy_feature_importance.csv`
- `outputs/plots/*.png`
- `outputs/plots/worst_20_wavelength_predictions.csv`
- `outputs/plots/worst_20_plqy_predictions.csv`
- `outputs/models/*.pkl`

Plots include predicted-vs-actual, residuals, error by solvent, and the PLQY classifier confusion matrix. SHAP summary plots are created if `shap` is installed.

## Interpreting Metrics

Regression:
- MAE: average absolute error, easiest to interpret
- RMSE: penalizes large misses more strongly
- R²: variance explained
- Spearman: rank correlation

Classification:
- Accuracy: overall correct bright/dim predictions
- Precision: how often predicted bright is actually bright
- Recall: how many bright molecules were found
- F1: balance between precision and recall
- Confusion matrix: true/false bright/dim counts

## Installation

```bash
pip install -r requirements.txt
```

If RDKit is difficult to install with pip, use conda:

```bash
conda install -c conda-forge rdkit
```

## Known Limitations

- PLQY is experimentally noisy and can be hard to predict.
- Solvent descriptors may be incomplete.
- Random split can overestimate real performance.
- Scaffold split is more honest for new chemistry.
- The dataset size is probably modest for deep learning, so strong classical ML baselines are a good fit.

