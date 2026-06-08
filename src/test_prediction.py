from __future__ import annotations

import subprocess
import sys


TEST_COMMANDS = [
    ["python", "-m", "src.predict", "--smiles", "CCO", "--solvent", "MeOH"],
    ["python", "-m", "src.predict", "--smiles", "c1ccccc1", "--solvent", "MeCN"],
    ["python", "-m", "src.predict", "--smiles", "CCO", "--solvent", "unknown_solvent"],
    ["python", "-m", "src.predict", "--smiles", "invalid_smiles", "--solvent", "MeOH"],
]


def main() -> int:
    print("Running ChemFluor prediction sanity checks.\n")
    for command in TEST_COMMANDS:
        print("=" * 72)
        print(" ".join(command))
        completed = subprocess.run(command, text=True, capture_output=True)
        print(completed.stdout)
        if completed.stderr:
            print(completed.stderr)
        if "invalid_smiles" in command:
            if completed.returncode == 0:
                print("Expected invalid SMILES to fail cleanly, but it succeeded.")
                return 1
        elif completed.returncode != 0:
            print("Prediction command failed. If models/metadata are missing, run python -m src.train first.")
            return completed.returncode
    print("Sanity checks finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
