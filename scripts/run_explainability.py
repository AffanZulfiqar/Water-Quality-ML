"""
scripts/run_explainability.py
──────────────────────────────
SHAP-based explainability analysis for all trained models.

Produces:
- Global SHAP feature importance tables
- SHAP summary plots (beeswarm)
- SHAP bar charts (mean |SHAP|)
- Local SHAP waterfall plots for selected examples
- Cross-model SHAP ranking comparison table

Usage:
    python scripts/run_explainability.py
    python scripts/run_explainability.py --config configs/config.yaml --model RandomForest
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_config
from src.explainability.shap_analysis import (
    get_shap_explainer, compute_shap_values,
    get_global_feature_importance, get_ranked_features,
    select_local_examples, save_shap_values,
    compare_feature_rankings
)
from src.visualization.plots import plot_shap_summary, plot_shap_bar, plot_shap_waterfall

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("run_explainability")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SHAP explainability analysis.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default=None, help="Specific model name to analyze (or all if omitted)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    results_base = Path(cfg["results"]["metrics"])
    models_dir = Path(cfg["results"]["models"])
    shap_dir = Path(cfg["results"]["shap"])
    figs_dir = Path(cfg["results"]["figures"])
    shap_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    X_train = pd.read_csv(results_base / "X_train.csv")
    X_test = pd.read_csv(results_base / "X_test.csv")
    y_test = pd.read_csv(results_base / "y_test.csv").iloc[:, 0]
    feature_names = list(X_train.columns)

    shap_cfg = cfg.get("shap", {})
    bg_samples = shap_cfg.get("background_samples", 100)
    max_display = shap_cfg.get("max_display", 9)

    # Determine which models to analyze
    tree_models = ["RandomForest", "XGBoost", "LightGBM", "DecisionTree"]
    other_models = ["LogisticRegression", "SVM", "MLP"]
    all_model_names = tree_models + other_models

    if args.model:
        all_model_names = [args.model]

    all_rankings = {}

    for model_name in all_model_names:
        model_path = models_dir / f"{model_name}.pkl"
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}. Skipping.")
            continue

        logger.info(f"═══ SHAP Analysis: {model_name} ═══")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        try:
            # Get explainer
            explainer = get_shap_explainer(
                model, X_train, model_name=model_name, background_samples=bg_samples
            )

            # Compute SHAP values on test set
            shap_values = compute_shap_values(explainer, X_test, model_name=model_name)

            # Save raw SHAP values
            save_shap_values(
                shap_values, feature_names,
                str(shap_dir / f"shap_values_{model_name}.csv"),
                model_name=model_name
            )

            # Global importance
            importance_df = get_global_feature_importance(shap_values, feature_names)
            importance_df.to_csv(shap_dir / f"global_importance_{model_name}.csv", index=False)
            logger.info(f"\nTop features for {model_name}:\n{importance_df.to_string(index=False)}")

            # Ranked features (for ablation)
            ranked = get_ranked_features(shap_values, feature_names)
            all_rankings[model_name] = ranked

            # Visualizations
            plot_shap_bar(
                importance_df, model_name,
                str(figs_dir / f"shap_bar_{model_name}.png")
            )
            plot_shap_summary(
                shap_values, X_test, model_name,
                str(figs_dir / f"shap_summary_{model_name}.png"),
                max_display=max_display
            )

            # Local explanations
            example_cfg = shap_cfg.get("local_examples", {})
            examples = select_local_examples(
                model, X_test, y_test,
                n_correct=example_cfg.get("n_correct", 2),
                n_fp=example_cfg.get("n_fp", 2),
                n_fn=example_cfg.get("n_fn", 2),
                n_high_conf=example_cfg.get("n_high_confidence", 2),
            )
            for category, df_ex in examples.items():
                if len(df_ex) == 0:
                    continue
                for i, (idx, row) in enumerate(df_ex.iterrows()):
                    plot_shap_waterfall(
                        explainer, row, model_name,
                        example_label=f"{category}_{i}",
                        output_path=str(figs_dir / f"shap_waterfall_{model_name}_{category}_{i}.png")
                    )

        except Exception as e:
            logger.error(f"SHAP analysis failed for {model_name}: {e}", exc_info=True)
            continue

    # Cross-model feature ranking comparison
    if len(all_rankings) >= 2:
        ranking_table = compare_feature_rankings(all_rankings)
        ranking_table.to_csv(shap_dir / "cross_model_ranking_comparison.csv", index=False)
        logger.info(f"\nCross-model feature ranking comparison:\n{ranking_table.to_string(index=False)}")

    # Save global SHAP importance for the primary model (Random Forest) for ablation
    if "RandomForest" in all_rankings:
        ranked_features_path = shap_dir / "ranked_features_RandomForest.json"
        import json
        with open(ranked_features_path, "w") as f:
            json.dump(all_rankings["RandomForest"], f)
        logger.info(f"Ranked features saved: {ranked_features_path}")

    logger.info("═══ Explainability Analysis Complete ═══")


if __name__ == "__main__":
    main()
