"""
scripts/train_models.py
────────────────────────
Main training script: loads data, preprocesses, trains all benchmark models,
runs cross-validation, saves fitted models and metrics.

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --config configs/config.yaml

This script implements the complete RQ1 experimental pipeline.
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import (
    load_config, load_raw_dataset, validate_dataset,
    split_dataset, get_feature_names
)
from src.preprocessing.pipeline import WaterQualityPreprocessor, remove_duplicates
from src.models.registry import build_models, get_hyperparameter_search_spaces
from src.evaluation.metrics import (
    evaluate_single_split, cross_validate_model,
    save_metrics_json, save_metrics_csv, format_cv_result_for_display
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("train_models")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate all benchmark models.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter optimization for top models")
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation (faster)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    SEED = cfg.get("random_seed", 42)
    np.random.seed(SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base = Path(cfg["results"]["metrics"])
    models_dir = Path(cfg["results"]["models"])
    results_base.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load and validate dataset ─────────────────────────────────────────
    logger.info("═══ PHASE 1: Data Loading ═══")
    data_path = cfg["data"]["primary_dataset"]
    df = load_raw_dataset(data_path)
    validate_dataset(df, target_col=cfg["data"]["target_column"])

    # ── 2. Remove duplicates ──────────────────────────────────────────────────
    df, n_dupes = remove_duplicates(df)
    logger.info(f"Dataset after deduplication: {df.shape}")

    # ── 3. Train/validation/test split ────────────────────────────────────────
    logger.info("═══ PHASE 2: Data Splitting ═══")
    split_cfg = cfg.get("splitting", {})
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df,
        target_col=cfg["data"]["target_column"],
        test_size=split_cfg.get("test_size", 0.15),
        validation_size=split_cfg.get("validation_size", 0.15),
        random_seed=SEED
    )

    # ── 4. Preprocessing (fit on train ONLY) ──────────────────────────────────
    logger.info("═══ PHASE 3: Preprocessing ═══")
    preprocessor = WaterQualityPreprocessor(cfg)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = list(X_train.columns)
    logger.info(f"Features: {feature_names}")

    # Save preprocessor for later use (SHAP, ablation, robustness)
    with open(models_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info("Preprocessor saved.")

    # Save splits for reproducibility
    X_train_proc.to_csv(results_base / "X_train.csv", index=False)
    X_val_proc.to_csv(results_base / "X_val.csv", index=False)
    X_test_proc.to_csv(results_base / "X_test.csv", index=False)
    y_train.to_csv(results_base / "y_train.csv", index=False)
    y_val.to_csv(results_base / "y_val.csv", index=False)
    y_test.to_csv(results_base / "y_test.csv", index=False)
    logger.info("Preprocessed data splits saved.")

    # ── 5. Build models ───────────────────────────────────────────────────────
    logger.info("═══ PHASE 4: Model Training ═══")
    models = build_models(cfg, random_seed=SEED)

    # ── 6. Hyperparameter optimization (optional) ─────────────────────────────
    if args.tune:
        logger.info("═══ PHASE 4b: Hyperparameter Optimization ═══")
        tune_models = cfg.get("hyperparameter_optimization", {}).get("models_to_tune", [])
        search_spaces = get_hyperparameter_search_spaces()
        n_iter = cfg.get("hyperparameter_optimization", {}).get("n_iter", 50)
        cv_folds = cfg.get("hyperparameter_optimization", {}).get("cv_folds", 5)

        for model_name in tune_models:
            if model_name not in models or model_name not in search_spaces:
                continue
            logger.info(f"Tuning {model_name}...")
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
            search = RandomizedSearchCV(
                models[model_name],
                param_distributions=search_spaces[model_name],
                n_iter=n_iter,
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1,
                random_state=SEED,
                verbose=1
            )
            search.fit(X_train_proc, y_train)
            models[model_name] = search.best_estimator_
            logger.info(f"{model_name} best params: {search.best_params_}")
            save_metrics_json(
                {"model": model_name, "best_params": search.best_params_, "best_cv_score": search.best_score_},
                str(results_base / f"hpopt_{model_name}.json")
            )

    # ── 7. Train all models, collect metrics ──────────────────────────────────
    logger.info("═══ PHASE 5: Training & Evaluation ═══")
    all_val_metrics = []
    all_test_metrics = []
    all_cv_results = []
    fitted_models = {}

    for model_name, model in models.items():
        logger.info(f"── Training: {model_name} ──")

        # Fit on training data
        model.fit(X_train_proc, y_train)
        fitted_models[model_name] = model

        # Validation set metrics
        val_metrics = evaluate_single_split(
            model, X_val_proc, y_val, split_name="val", model_name=model_name
        )
        val_metrics["timestamp"] = timestamp
        all_val_metrics.append(val_metrics)
        save_metrics_json(val_metrics, str(results_base / f"val_{model_name}.json"))

        # Cross-validation on training data (skipped if --skip-cv)
        if not args.skip_cv:
            cv_result = cross_validate_model(
                model, X_train_proc, y_train,
                n_splits=cfg.get("cross_validation", {}).get("n_splits", 5),
                random_seed=SEED,
                model_name=model_name
            )
            cv_result["timestamp"] = timestamp
            all_cv_results.append(cv_result)
            logger.info(format_cv_result_for_display(cv_result))

        # Save fitted model
        model_path = models_dir / f"{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved: {model_path}")

    # ── 8. Final test set evaluation (ONE TIME) ───────────────────────────────
    logger.info("═══ PHASE 6: Final Test Set Evaluation ═══")
    for model_name, model in fitted_models.items():
        test_metrics = evaluate_single_split(
            model, X_test_proc, y_test, split_name="test", model_name=model_name
        )
        test_metrics["timestamp"] = timestamp
        all_test_metrics.append(test_metrics)
        save_metrics_json(test_metrics, str(results_base / f"test_{model_name}.json"))

    # ── 9. Save aggregate results ─────────────────────────────────────────────
    save_metrics_csv(all_val_metrics, str(results_base / "all_val_metrics.csv"))
    save_metrics_csv(all_test_metrics, str(results_base / "all_test_metrics.csv"))
    if all_cv_results:
        # Flatten CV results to CSV
        cv_rows = []
        for cvr in all_cv_results:
            row = {"model": cvr["model"]}
            for m in ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "average_precision"]:
                row[f"{m}_mean"] = cvr.get(f"{m}_mean", None)
                row[f"{m}_std"] = cvr.get(f"{m}_std", None)
            cv_rows.append(row)
        save_metrics_csv(cv_rows, str(results_base / "all_cv_results.csv"))

    logger.info("═══ Training Complete ═══")
    logger.info(f"Results saved to: {results_base}")
    logger.info(f"Models saved to: {models_dir}")
    logger.info("\nTest set results summary:")
    test_df = pd.DataFrame(all_test_metrics)[["model", "accuracy", "f1_macro", "roc_auc"]]
    print(test_df.to_string(index=False))


if __name__ == "__main__":
    main()
