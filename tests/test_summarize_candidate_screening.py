from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "summarize_candidate_screening.py"

spec = importlib.util.spec_from_file_location("summarize_candidate_screening", SCRIPT_PATH)
summarizer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(summarizer)


def test_summarize_screening_writes_outputs_and_assigns_ranks(tmp_path: Path) -> None:
    first_input = tmp_path / "ranked_450.csv"
    second_input = tmp_path / "ranked_520.csv"
    out_path = tmp_path / "nested" / "summary.csv"
    markdown_path = tmp_path / "nested" / "summary.md"

    pd.DataFrame(
        {
            "name": ["blue_a", "blue_b", "blue_c"],
            "scaffold": ["coumarin", "coumarin", "bodipy"],
            "substituent": ["F", "Me", "F"],
            "smiles": ["C", "CC", "CCC"],
            "canonical_smiles": ["C", "CC", "CCC"],
            "solvent_smiles": ["CCO", "CCO", "CCO"],
            "predicted_absorption_nm": [360, 365, 370],
            "predicted_emission_nm": [452, 460, 470],
            "predicted_quantum_yield": [0.8, 0.7, 0.6],
            "predicted_log_extinction": [4.1, 4.0, 3.9],
            "emission_error_from_target": [2, 10, 20],
            "score": [0.1, 0.2, 0.3],
            "estimated_brightness_score": [1000, 900, 800],
        }
    ).to_csv(first_input, index=False)

    pd.DataFrame(
        {
            "name": ["green_a", "green_b"],
            "scaffold": ["xanthene", "xanthene"],
            "smiles": ["O", "CO"],
            "predicted_emission_nm": [560, 530],
            "predicted_quantum_yield": [0.5, 0.4],
            "emission_error_from_target": [40, 10],
            "score": [0.4, 0.5],
        }
    ).to_csv(second_input, index=False)

    summary = summarizer.summarize_screening(
        input_paths=[first_input, second_input],
        targets=[450, 520],
        out_path=out_path,
        markdown_path=markdown_path,
        top_n=2,
    )

    assert out_path.exists()
    assert markdown_path.exists()
    assert len(summary) == 4

    written = pd.read_csv(out_path)
    assert list(written["rank"]) == [1, 2, 1, 2]
    assert list(written["target_emission_nm"]) == [450, 450, 520, 520]
    assert "substituent" in written.columns
    assert pd.isna(written.loc[2, "substituent"])

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Candidate Screening Summary" in markdown
    assert "## Target emission: 450 nm" in markdown
    assert "current candidate library may not contain candidates close enough" in markdown
