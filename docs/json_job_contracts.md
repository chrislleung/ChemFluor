# FluorCast JSON job contracts

These command-line entrypoints run one portal job from an input JSON file and
always attempt to write an output JSON file. They do not make network calls or
submit Slurm jobs.

## Prediction jobs

```powershell
python scripts/run_prediction_job.py --input job_input.json --output job_output.json
```

Input:

```json
{
  "job_id": "job-123",
  "user_id": "user-123",
  "molecule_smiles": "c1ccccc1",
  "solvent_smiles": "CCO",
  "model_choice": "all",
  "requested_at": "2026-06-22T12:00:00Z"
}
```

`model_choice` accepts `all`, `rf`, `extratrees`, `histgb`, or
`graph_model_later`. The last value is reserved and currently returns
`PREDICTION_BACKEND_NOT_CONNECTED`.

Successful output:

```json
{
  "status": "success",
  "job_id": "job-123",
  "canonical_molecule_smiles": "c1ccccc1",
  "canonical_solvent_smiles": "CCO",
  "predictions": [
    {
      "model_name": "rf",
      "predicted_absorption_nm": null,
      "predicted_emission_nm": 450.0,
      "predicted_quantum_yield": 0.2,
      "nearest_training_similarity": 0.8,
      "nearest_training_smiles": "c1ccccc1",
      "warnings": []
    }
  ],
  "warnings": []
}
```

The runner adapts `scripts/predict_all_models.py`. It never invents values: if
no requested model artifacts produce a prediction, the job fails with
`PREDICTION_BACKEND_NOT_CONNECTED`. The existing models do not currently emit
absorption predictions, so that property is `null`.

## Duplicate-check jobs

```powershell
python scripts/run_duplicate_check_job.py --input duplicate_input.json --output duplicate_output.json --dataset data/processed/fluodb_lite/combined_deduplicated.csv --max-matches 5
```

The checked-in combined deduplicated CSV is used by default when present.
Custom datasets must contain a recognized molecule SMILES column and either
`canonical_solvent_smiles` or `solvent_smiles`.

Input:

```json
{
  "submission_id": "submission-123",
  "user_id": "user-123",
  "molecule_smiles": "c1ccccc1",
  "solvent_smiles": "CCO",
  "submitted_at": "2026-06-22T12:00:00Z"
}
```

Successful output contains `exact_duplicate_found`, an optional exact record
ID, canonical molecule and solvent SMILES, and up to `max-matches` nearest
records. Each nearest record contains its molecule/solvent SMILES, Morgan
Tanimoto similarity, optional emission and quantum-yield values, and an
optional DOI.

## Failure output

Both runners use this shape on validation, configuration, parsing, and runtime
errors:

```json
{
  "status": "failed",
  "job_id": "job-123",
  "error_code": "INVALID_INPUT",
  "error_message": "Missing required field(s): molecule_smiles",
  "traceback": "Traceback (most recent call last): ...",
  "warnings": []
}
```

Duplicate-check failures use `submission_id` in place of `job_id`. A missing
dataset produces `DATASET_NOT_CONFIGURED`.

## Portal and NIBI use

The future portal can write an input JSON file, submit a Slurm job that invokes
one of these commands on NIBI, and read the resulting JSON file after the job
finishes. Slurm submission, status polling, file transfer, authentication, and
portal persistence all happen outside these scripts.
