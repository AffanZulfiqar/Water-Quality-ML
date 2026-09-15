"""
scripts/run_ablation.py
────────────────────────
Feature ablation experiment script.

Loads SHAP-ranked feature list from previous explainability run,
builds feature configurations, re-trains models on each subset,
and evaluates on the held-out test set.

Usage:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --config configs/config.yaml
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_config
from src.experiments.ablation import build_ablation_configs, run_full_ablation
from src.visualization.plots import plot_ablation_curves

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("run_ablation")


def parse_args():
    parser = argparse.ArgumentParser(description="Run feature ablation experiment.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    results_base = Path(cfg["results"]["metrics"])
    models_dir = Path(cfg["results"]["models"])
    shap_dir = Path(cfg["results"]["shap"])
    ablation_dir = Path(cfg["results"]["ablation"])
    figs_dir = Path(cfg["results"]["figures"])
    ablation_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    X_train = pd.read_csv(results_base / "X_train.csv")
    y_train = pd.read_csv(results_base / "y_train.csv").iloc[:, 0]
    X_test = pd.read_csv(results_base / "X_test.csv")
    y_test = pd.read_csv(results_base / "y_test.csv").iloc[:, 0]

    # Load SHAP-ranked features (from training data explainability)
    ranked_features_path = shap_dir / "ranked_features_RandomForest.json"
    if not ranked_features_path.exists():
        logger.error(
            f"Ranked features file not found: {ranked_features_path}\n"
            "Please run scripts/run_explainability.py first."
        )
        sys.exit(1)

    with open(ranked_features_path, "r") as f:
        ranked_features = json.load(f)
    logger.info(f"SHAP feature ranking: {ranked_features}")

    # Build feature configurations
    configurations = build_ablation_configs(ranked_features, cfg)

    # Load models for ablation (using un-fitted versions from registry)
    from src.models.registry import build_models
    SEED = cfg.get("random_seed", 42)
    models = build_models(cfg, random_seed=SEED)

    # Run ablation experiment
    logger.info("═══ Running Feature Ablation Experiment ═══")
    ablation_df = run_full_ablation(
        models=models,
        configurations=configurations,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_dir=str(ablation_dir),
    )

    # Generate ablation curves
    plot_ablation_curves(
        ablation_df,
        metrics=["f1_macro", "roc_auc"],
        output_path=str(figs_dir / "ablation_curves.png")
    )

    # Print summary
    logger.info("\n═══ Ablation Results Summary ═══")
    summary = ablation_df[["model", "ablation_config", "n_features", "f1_macro", "roc_auc", "accuracy"]]
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
