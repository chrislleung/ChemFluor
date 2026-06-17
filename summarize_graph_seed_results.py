from pathlib import Path
import pandas as pd

rows = []

paths = [
    ("gin_emission_seed0", "outputs/graph_gin_emission_3seeds_gpu/seed_0/graph_model_comparison.csv"),
    ("gin_emission_seed1", "outputs/graph_gin_emission_3seeds_gpu/seed_1/graph_model_comparison.csv"),
    ("gin_emission_seed2", "outputs/graph_gin_emission_3seeds_gpu/seed_2/graph_model_comparison.csv"),
    ("gcn_emission_seed0", "outputs/graph_gcn_emission_3seeds_gpu/seed_0/graph_model_comparison.csv"),
    ("gcn_emission_seed1", "outputs/graph_gcn_emission_3seeds_gpu/seed_1/graph_model_comparison.csv"),
    ("gcn_emission_seed2", "outputs/graph_gcn_emission_3seeds_gpu/seed_2/graph_model_comparison.csv"),
    ("gin_qy_seed0", "outputs/graph_gin_qy_gpu/graph_model_comparison.csv"),
    ("gcn_qy_seed0", "outputs/graph_gcn_qy_gpu/graph_model_comparison.csv"),
]

for label, path in paths:
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
        df.insert(0, "run_label", label)
        rows.append(df)

out = pd.concat(rows, ignore_index=True)
out.to_csv("outputs/graph_seed_summary.csv", index=False)

summary = (
    out.groupby(["model", "target"])
    .agg(
        seeds=("seed", "count"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        mae_min=("mae", "min"),
        mae_max=("mae", "max"),
        r2_mean=("r2", "mean"),
        r2_min=("r2", "min"),
        r2_max=("r2", "max"),
    )
    .reset_index()
)

summary.to_csv("outputs/graph_seed_summary_grouped.csv", index=False)

print("Saved:")
print("outputs/graph_seed_summary.csv")
print("outputs/graph_seed_summary_grouped.csv")
print()
print(summary.to_string(index=False))
