# ChemFluor

ChemFluor is a machine-learning workflow for predicting fluorescence-related properties of organic chromophores from molecular structure and solvent information. The project has been expanded to combine the original ChemFluor dataset with the Deep4Chem chromophore dataset, train solvent-aware prediction models, and perform first-pass candidate screening for target fluorescence wavelengths.

The current workflow supports:

```text
molecule + solvent → predicted optical properties
```

and an early inverse-design workflow:

```text
target emission + solvent + candidate molecules → ranked candidate fluorophores
```

This is not full neural molecular generation yet. The current candidate-generation step uses rule-based scaffold enumeration, then the trained model ranks the generated candidates.

---

## Repository Structure

```text
ChemFluor_Project_synced/
├── data/                         # Input and processed data
├── models/                       # Local trained models; ignored by Git
├── notebooks/                    # Optional notebooks
├── outputs/                      # Generated reports/plots; ignored by Git
├── scripts/                      # Command-line scripts
├── src/chemfluor/                # Reusable ChemFluor Python package
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
├── README.md
└── run_chemfluor.sh              # Slurm helper script for Compute Canada
```

Important: trained model files and generated outputs are intentionally ignored by Git because they can be large and are reproducible from the scripts.

---

## Installation

Create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux / Compute Canada

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

RDKit is required for SMILES canonicalization, molecular fingerprints, and descriptor generation.

---

## Main Workflow

The complete command-line workflow is:

```text
1. Analyze Deep4Chem dataset
2. Build expanded solvent descriptors
3. Train combined ChemFluor + Deep4Chem models
4. Generate model reports and plots
5. Compare model types
6. Analyze prediction errors
7. Generate scaffold-based candidate molecules
8. Screen candidates for target emission wavelengths
```

---

## 1. Analyze the Deep4Chem Dataset

This script inspects the raw Deep4Chem chromophore dataset. It reports dataset size, missing values, target coverage, common solvents, invalid solvent labels, and SMILES validity.

```powershell
python scripts/analyze_deep4chem_dataset.py `
  --input "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv"
```

Default output directory:

```text
outputs/deep4chem_analysis/
```

Generated files include:

```text
deep4chem_summary.txt
top_solvents.csv
missing_values.csv
numeric_summary.csv
invalid_solvents.csv
```

---

## 2. Build Expanded Solvent Descriptors

This script creates an expanded solvent descriptor table for the Deep4Chem solvents. It extracts unique solvents, validates solvent SMILES with RDKit, computes RDKit solvent descriptors, and merges any existing physical solvent descriptors.

```powershell
python scripts/make_deep4chem_solvent_descriptors.py `
  --deep4chem "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv" `
  --existing-solvents data/solvent_descriptors.csv `
  --output data/solvent_descriptors_expanded_deep4chem.csv
```

Optional report path:

```powershell
python scripts/make_deep4chem_solvent_descriptors.py `
  --deep4chem "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv" `
  --existing-solvents data/solvent_descriptors.csv `
  --output data/solvent_descriptors_expanded_deep4chem.csv `
  --report outputs/deep4chem_analysis/solvent_descriptor_report.txt
```

The resulting solvent descriptor CSV is used during model training and candidate screening.

---

## 3. Train Combined ChemFluor + Deep4Chem Models

This script standardizes the original ChemFluor dataset and Deep4Chem dataset into one common schema, merges solvent descriptors, generates Morgan fingerprints, and trains one model per target property.

Targets:

```text
absorption_nm
emission_nm
lifetime_ns
quantum_yield
log_extinction
```

The train/test split is grouped by canonical chromophore SMILES, meaning the same chromophore is not allowed to appear in both train and test sets.

### Train Random Forest Models

```powershell
python scripts/train_combined_predictors.py `
  --deep4chem "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv" `
  --chemfluor data/chemfluor_data.csv `
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv `
  --out-dir models/chemfluor_combined `
  --model rf
```

### Train HistGradientBoosting Models

```powershell
python scripts/train_combined_predictors.py `
  --deep4chem "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv" `
  --chemfluor data/chemfluor_data.csv `
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv `
  --out-dir models/chemfluor_combined_histgb `
  --model histgb
```

Useful options:

```powershell
--n-bits 2048
--radius 2
```

Model outputs are saved under the chosen model directory, including:

```text
metrics.json
feature_metadata.json
predictions_absorption_nm.csv
predictions_emission_nm.csv
predictions_lifetime_ns.csv
predictions_quantum_yield.csv
predictions_log_extinction.csv
*_rf.joblib or *_histgb.joblib
```

These outputs are generated artifacts and are ignored by Git.

---

## 4. Generate Model Report

This script creates a model summary, metrics table, and diagnostic plots.

```powershell
python scripts/report_combined_model_results.py `
  --model-dir models/chemfluor_combined `
  --out-dir outputs/combined_model_report
```

Generated outputs:

```text
outputs/combined_model_report/model_summary.md
outputs/combined_model_report/metrics_table.csv
outputs/combined_model_report/figures/
```

Each target gets:

```text
predicted vs actual plot
residual histogram
residual vs predicted plot
```

---

## 5. Compare Random Forest vs HistGradientBoosting

After training both model types, compare them:

```powershell
python scripts/compare_model_results.py `
  --rf-dir models/chemfluor_combined `
  --histgb-dir models/chemfluor_combined_histgb `
  --out-dir outputs/model_comparison_report
```

Generated outputs:

```text
outputs/model_comparison_report/model_comparison.csv
outputs/model_comparison_report/model_comparison.md
outputs/model_comparison_report/mae_comparison.png
outputs/model_comparison_report/rmse_comparison.png
outputs/model_comparison_report/r2_comparison.png
```

The current main baseline is Random Forest because it gives the strongest MAE profile overall, while HistGradientBoosting remains useful as a comparison model.

---

## 6. Analyze Prediction Errors

This script identifies where the model performs best and worst. It saves best/worst predictions, error summaries by source dataset, top-error solvents, and wavelength-region summaries for absorption/emission.

```powershell
python scripts/analyze_prediction_errors.py `
  --model-dir models/chemfluor_combined `
  --out-dir outputs/error_analysis
```

Generated outputs include:

```text
overall_error_summary.csv
error_analysis_report.md
worst_predictions_<target>.csv
best_predictions_<target>.csv
error_by_source_dataset_<target>.csv
top_error_solvents_<target>.csv
error_by_wavelength_region_absorption_nm.csv
error_by_wavelength_region_emission_nm.csv
```

Use this step to understand where the model is less reliable, such as unusual solvents, rare chromophore classes, or red/NIR wavelength regions.

---

## 7. Generate Scaffold-Based Candidate Molecules

This script performs rule-based scaffold enumeration. It is not neural molecular generation.

It combines predefined coumarin-like and naphthalimide-like scaffold templates with substituent fragments, validates the molecules with RDKit, removes duplicates, and saves the candidate library.

```powershell
python scripts/generate_scaffold_candidates.py
```

Default output:

```text
data/generated_candidates/scaffold_candidates.csv
```

Expected default run summary:

```text
Scaffold templates used: 5
Substituents used: 12
Raw combinations attempted: 60
Unique valid molecules saved: 59
Saved candidates to: data/generated_candidates/scaffold_candidates.csv
```

Choose only coumarin candidates:

```powershell
python scripts/generate_scaffold_candidates.py `
  --scaffolds coumarin `
  --out data/generated_candidates/coumarin_candidates.csv
```

Choose only naphthalimide candidates:

```powershell
python scripts/generate_scaffold_candidates.py `
  --scaffolds naphthalimide `
  --out data/generated_candidates/naphthalimide_candidates.csv
```

Choose custom substituents:

```powershell
python scripts/generate_scaffold_candidates.py `
  --scaffolds all `
  --substituents cyano,methoxy,diethylamino,phenyl `
  --out data/generated_candidates/custom_candidates.csv
```

Candidate output columns:

```text
name
scaffold
substituent
smiles
canonical_smiles
```

---

## 8. Screen Candidate Molecules

This script scores and ranks candidate molecules using trained ChemFluor models.

Inputs:

```text
candidate molecule CSV
solvent SMILES
target emission wavelength
trained model directory
solvent descriptor CSV
```

Outputs:

```text
ranked candidate CSV
```

The screening script predicts:

```text
predicted_absorption_nm
predicted_emission_nm
predicted_quantum_yield
predicted_log_extinction
```

and ranks candidates using a score that rewards closeness to the target emission wavelength and higher predicted quantum yield.

### Screen for 450 nm emission in ethanol

```powershell
python scripts/screen_candidate_molecules.py `
  --candidates data/generated_candidates/scaffold_candidates.csv `
  --solvent-smiles CCO `
  --target-emission 450 `
  --model-dir models/chemfluor_combined `
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv `
  --out outputs/candidate_screening/ranked_scaffold_candidates_ethanol_450.csv
```

### Screen for 520 nm emission in ethanol

```powershell
python scripts/screen_candidate_molecules.py `
  --candidates data/generated_candidates/scaffold_candidates.csv `
  --solvent-smiles CCO `
  --target-emission 520 `
  --model-dir models/chemfluor_combined `
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv `
  --out outputs/candidate_screening/ranked_scaffold_candidates_ethanol_520.csv
```

### Screen for 600 nm emission in ethanol

```powershell
python scripts/screen_candidate_molecules.py `
  --candidates data/generated_candidates/scaffold_candidates.csv `
  --solvent-smiles CCO `
  --target-emission 600 `
  --model-dir models/chemfluor_combined `
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv `
  --out outputs/candidate_screening/ranked_scaffold_candidates_ethanol_600.csv
```

Ranked output columns:

```text
name
scaffold
substituent
smiles
canonical_smiles
solvent_smiles
predicted_absorption_nm
predicted_emission_nm
predicted_quantum_yield
predicted_log_extinction
emission_error_from_target
score
estimated_brightness_score
```

---

## Example Candidate-Screening Result

Using the 59 generated coumarin/naphthalimide candidates in ethanol:

| Target emission | Top candidate                                     | Top scaffold          | Substituent   | Predicted emission | Predicted QY |
| --------------: | ------------------------------------------------- | --------------------- | ------------- | -----------------: | -----------: |
|          450 nm | naphthalimide_4_substituted_n_butyl_phenyl        | N-butyl naphthalimide | phenyl        |           458.0 nm |        0.437 |
|          520 nm | naphthalimide_4_substituted_n_butyl_cyano         | N-butyl naphthalimide | cyano         |           504.5 nm |        0.420 |
|          600 nm | naphthalimide_4_substituted_n_butyl_dimethylamino | N-butyl naphthalimide | dimethylamino |           562.3 nm |        0.428 |

The 600 nm result shows that the current small candidate library does not reach far enough into the red/orange region. Future work should expand the scaffold library with more red-shifted fluorophores.

---

## Reusable Package Modules

The `src/chemfluor/` folder contains reusable code used by the scripts.

Important modules include:

```text
data_standardization.py
    Standardizes ChemFluor and Deep4Chem into a shared schema.

features.py
    Builds molecular and solvent features for the original ChemFluor workflow.

models.py
    Defines model utilities for the original ChemFluor workflow.

evaluate.py
    Computes model metrics.

plots.py
    Creates plots for model evaluation.

splitting.py
    Handles train/test splitting.

predict.py
    Prediction utilities.

applicability.py
    Applicability-domain utilities.

utils.py
    Shared helper functions.
```

---

## Testing

Run tests with:

```powershell
python -m pytest tests
```

Run a specific test file:

```powershell
python -m pytest tests/test_data_standardization.py
```

---

## Git / Large File Policy

The repository should include:

```text
source code
scripts
tests
documentation
small sanitized datasets
configuration files
```

The repository should not include:

```text
virtual environments
trained model files
joblib/pickle files
generated outputs
plots
large raw datasets
private unsanitized data
Slurm logs
cache folders
```

Before pushing:

```powershell
git status
```

Check staged files carefully. If model or output files were staged accidentally:

```powershell
git reset models/
git reset outputs/
```

If a large file is already tracked and should be removed from Git while staying on your computer:

```powershell
git rm -r --cached models/
git rm -r --cached outputs/
```

Then commit:

```powershell
git add .gitignore README.md requirements.txt scripts src tests
git commit -m "Add combined ChemFluor Deep4Chem workflow and candidate screening"
git push
```

---

## Current Status

The project currently supports:

```text
combined ChemFluor + Deep4Chem training
solvent-aware optical-property prediction
Random Forest and HistGradientBoosting comparison
model reporting and error analysis
rule-based scaffold candidate generation
target-emission candidate screening
```

The next major development step is to expand the candidate generator with additional red-shifted fluorophore scaffolds, such as BODIPY-like, rhodamine-like, fluorescein-like, cyanine-like, and larger donor-acceptor systems.
