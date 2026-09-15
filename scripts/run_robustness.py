"""
scripts/run_robustness.py
──────────────────────────
Measurement noise robustness experiment script.

Loads fitted models and evaluates them on progressively noisier
versions of the held-out test set.

Usage:
    python scripts/run_robustness.py
    python scripts/run_robustness.py --config configs/config.yaml
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_config
from src.experiments.robustness import run_full_robustness
from src.visualization.plots import plot_robustness_curves

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("run_robustness")


def parse_args():
    parser = argparse.ArgumentParser(description="Run noise robustness experiment.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    results_base = Path(cfg["results"]["metrics"])
    models_dir = Path(cfg["results"]["models"])
    robustness_dir = Path(cfg["results"]["robustness"])
    figs_dir = Path(cfg["results"]["figures"])
    robustness_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Load test data (clean)
    logger.info("Loading preprocessed test data...")
    X_test = pd.read_csv(results_base / "X_test.csv")
    y_test = pd.read_csv(results_base / "y_test.csv").iloc[:, 0]

    # Load fitted models
    robust_cfg = cfg.get("robustness", {})
    model_subset = robust_cfg.get("models_to_test", None)
    noise_levels = robust_cfg.get("noise_levels", [0.0, 0.02, 0.05, 0.10, 0.15])
    n_reps = robust_cfg.get("n_repetitions", 5)

    fitted_models = {}
    for model_path in sorted(models_dir.glob("*.pkl")):
        model_name = model_path.stem
        if model_name == "preprocessor":
            continue
        if model_subset and model_name not in model_subset:
            continue
        with open(model_path, "rb") as f:
            fitted_models[model_name] = pickle.load(f)
        logger.info(f"Loaded model: {model_name}")

    if not fitted_models:
        logger.error("No fitted models found. Run scripts/train_models.py first.")
        sys.exit(1)

    # Run robustness experiment
    logger.info("═══ Running Noise Robustness Experiment ═══")
    robustness_df = run_full_robustness(
        fitted_models=fitted_models,
        X_test=X_test,
        y_test=y_test,
        noise_levels=noise_levels,
        n_repetitions=n_reps,
        output_dir=str(robustness_dir),
        model_subset=model_subset,
    )

    # Generate robustness plots
    plot_robustness_curves(
        robustness_df,
        metrics=["f1_macro_mean", "roc_auc_mean"],
        output_path=str(figs_dir / "robustness_curves.png")
    )

    # Print summary
    logger.info("\n═══ Robustness Results Summary ═══")
    summary_cols = ["model", "noise_level", "f1_macro_mean", "f1_macro_std", "roc_auc_mean"]
    print(robustness_df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
