# Change python behavior
from __future__ import annotations

# Set default cores to hide warning thrown from joblib 
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Standard library imports
import argparse
import subprocess
import sys
from pathlib import Path


# Parse the provided command-line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full cardiovasc pipeline in numeric order."
    )
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Skip script 3 (UMAP) for a faster run.",
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip script 11 (final visualizations).",
    )
    parser.add_argument(
        "--skip-dependencies",
        action="store_true",
        help="Skip dependencies install (pip).",
    )
    return parser.parse_args()


# Helper function to run another script file
def run_script(script_path: Path) -> None:
    print(f"\n>>> Running: {script_path.name}")
    subprocess.run([sys.executable, str(script_path)], check=True)


def clear_generated_outputs(root: Path) -> None:
    print("Clearing generated files for a fresh run...")
    processed = root / "data" / "processed"
    models = root / "models"
    metrics = root / "results" / "metrics"
    figures = root / "results" / "figures"

    for folder in [processed, models, metrics, figures]:
        folder.mkdir(parents=True, exist_ok=True)
        for p in folder.iterdir():
            if p.is_file():
                p.unlink()


# Parse any argument provided and define src folder
args = parse_args()
root = Path(__file__).resolve().parent
src = root / "src"

# Set the files sequence
sequence = [
    src / "1_eda.py",
    src / "2_preprocessing.py",
    src / "3_umap_visualization.py",
    src / "4_random_forest.py",
    src / "5_logistic_regression.py",
    src / "6_mlp.py",
    src / "7_hist_gradient_boosting.py",
    src / "8_xgboost.py",
    src / "9_hyperparameter_tuning.py",
    src / "10_stacking.py",
    src / "11_results_comparison.py",
    src / "12_experimental_comparison.py",
    src / "13_visualizations.py",
    src / "14_kmeans.py",
    src / "15_stacking_risk_score.py",
]

# Based on arguments provided, skip certain files
if args.skip_umap:
    sequence = [s for s in sequence if s.name != "3_umap_visualization.py"]
if args.skip_visuals:
    sequence = [s for s in sequence if s.name != "13_visualizations.py"]
if not args.skip_dependencies:
    print(f"\n>>> Installing dependencies <<<")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True
    )

# Clear previous run ML generated files
clear_generated_outputs(root)

# Run remaining sequence files
for script in sequence:
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")
    run_script(script)

# Print confirmation
print("\nFull pipeline completed.")
print(f"Outputs are under: {root / 'results'}")