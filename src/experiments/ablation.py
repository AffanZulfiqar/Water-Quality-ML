"""
src/experiments/ablation.py
─────────────────────────────
Feature ablation experiment: evaluate performance under
progressively reduced feature sets.

Research Question RQ3:
How does reducing the number of measured parameters affect predictive performance?

DESIGN:
- Feature importance ranking is computed from TRAINING data SHAP values only.
- The same ranking (computed once) is applied to all ablation configurations.
- The test set is evaluated only after the configuration is finalized.
- No information from the test set influences feature selection.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_single_split, save_metrics_json

logger = logging.getLogger(__name__)


def build_ablation_configs(
    ranked_features: List[str],
    config: dict,
) -> Dict[str, List[str]]:
    """
    Build feature subset configurations from SHAP-ranked feature list.

    Parameters
    ----------
    ranked_features : list of str
        Feature names ordered by descending SHAP importance (computed on training data).
    config : dict
        Project config (ablation section defines configuration sizes).

    Returns
    -------
    dict mapping config_name → list of feature names to use
    """
    ablation_cfg = config.get("ablation", {}).get("configurations", {})
    n_total = len(ranked_features)

    configurations = {}

    # Config A: all features
    configurations["A_all"] = ranked_features[:]

    # Config B: Top 7 (or whatever n is specified)
    n_b = ablation_cfg.get("B", min(7, n_total))
    if isinstance(n_b, int):
        configurations[f"B_top{n_b}"] = ranked_features[:n_b]

    # Config C: Top 5
    n_c = ablation_cfg.get("C", min(5, n_total))
    if isinstance(n_c, int):
        configurations[f"C_top{n_c}"] = ranked_features[:n_c]

    # Config D: Top 3
    n_d = ablation_cfg.get("D", min(3, n_total))
    if isinstance(n_d, int):
        configurations[f"D_top{n_d}"] = ranked_features[:n_d]

    # Config E: Proxy low-cost sensors (from config or default)
    e_features = ablation_cfg.get("E", None)
    if isinstance(e_features, list):
        valid_e = [f for f in e_features if f in ranked_features]
        if valid_e:
            configurations["E_lowcost"] = valid_e
    elif isinstance(e_features, int):
        configurations[f"E_top{e_features}"] = ranked_features[:e_features]

    logger.info(f"Ablation configurations built: {list(configurations.keys())}")
    for name, feats in configurations.items():
        logger.info(f"  {name}: {feats}")

    return configurations


def run_ablation_experiment(
    model: Any,
    model_name: str,
    configurations: Dict[str, List[str]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "results/ablation/",
) -> List[Dict]:
    """
    Train and evaluate a model on each feature configuration.

    For each configuration:
    1. Select features from training set.
    2. Fit model on training features.
    3. Evaluate on test set (same feature subset).

    NOTE: The model is re-fitted from scratch for each configuration.
    This is required to avoid information leakage from features
    not included in a given configuration.

    Parameters
    ----------
    model : sklearn-compatible model (un-fitted)
    model_name : str
    configurations : dict mapping config_name → list of feature names
    X_train, y_train : training data
    X_test, y_test : test data (held-out)
    output_dir : str

    Returns
    -------
    List of metric dicts, one per configuration
    """
    from sklearn.base import clone

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []

    for config_name, features in configurations.items():
        logger.info(f"[Ablation] Model={model_name}, Config={config_name}, Features={features}")

        # Validate features exist in both train and test
        available = [f for f in features if f in X_train.columns]
        if len(available) != len(features):
            missing = set(features) - set(X_train.columns)
            logger.warning(f"Features not found in data: {missing}. Skipping.")
            continue

        # Select feature subsets
        X_train_sub = X_train[available]
        X_test_sub = X_test[available]

        # Clone and re-fit model
        model_clone = clone(model)
        model_clone.fit(X_train_sub, y_train)

        # Evaluate on test
        metrics = evaluate_single_split(
            model_clone, X_test_sub, y_test,
            split_name="test",
            model_name=model_name
        )
        metrics["ablation_config"] = config_name
        metrics["n_features"] = len(available)
        metrics["features_used"] = available

        results.append(metrics)

        save_metrics_json(
            metrics,
            str(Path(output_dir) / f"{model_name}_{config_name}.json")
        )

    logger.info(f"[Ablation] Completed {len(results)} configurations for {model_name}")
    return results


def run_full_ablation(
    models: Dict[str, Any],
    configurations: Dict[str, List[str]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "results/ablation/",
) -> pd.DataFrame:
    """
    Run feature ablation for all models and configurations.

    Returns
    -------
    pd.DataFrame with all ablation results.
    """
    all_results = []

    for model_name, model in models.items():
        if model_name == "Dummy":
            continue  # Skip dummy model in ablation
        model_results = run_ablation_experiment(
            model=model,
            model_name=model_name,
            configurations=configurations,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            output_dir=output_dir,
        )
        all_results.extend(model_results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(str(Path(output_dir) / "ablation_results_all.csv"), index=False)
    logger.info(f"Full ablation results saved to {output_dir}/ablation_results_all.csv")
    return results_df
