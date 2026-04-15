from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


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
    return parser.parse_args()


def run_script(script_path: Path) -> None:
    print(f"\n>>> Running: {script_path.name}")
    subprocess.run([sys.executable, str(script_path)], check=True)


def clear_generated_outputs(root: Path) -> None:
    processed = root / "data" / "processed"
    metrics = root / "results" / "metrics"
    figures = root / "results" / "figures"

    for folder in [processed, metrics, figures]:
        folder.mkdir(parents=True, exist_ok=True)
        for p in folder.iterdir():
            if p.is_file():
                p.unlink()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    src = root / "src"

    print("Clearing generated files for a fresh run...")
    clear_generated_outputs(root)

    sequence = [
        src / "1_eda.py",
        src / "2_preprocessing.py",
        src / "3_umap_visualization.py",
        src / "4_random_forest.py",
        src / "5_logistic_regression.py",
        src / "6_mlp.py",
        src / "7_hist_gradient_boosting.py",
        src / "8_hyperparameter_tuning.py",
        src / "9_results_comparison.py",
        src / "10_round_delta_report.py",
        src / "11_visualizations.py",
    ]

    if args.skip_umap:
        sequence = [s for s in sequence if s.name != "3_umap_visualization.py"]
    if args.skip_visuals:
        sequence = [s for s in sequence if s.name != "11_visualizations.py"]

    for script in sequence:
        if not script.exists():
            raise FileNotFoundError(f"Missing script: {script}")
        run_script(script)

    print("\nFull pipeline completed.")
    print(f"Outputs are under: {root / 'results'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
