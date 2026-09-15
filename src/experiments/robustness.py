"""
src/experiments/robustness.py
──────────────────────────────
Measurement noise robustness experiment.

Research Question RQ4:
How robust are the developed models to measurement noise that
could occur in practical sensor-based monitoring?

NOISE MODEL:
Multiplicative Gaussian noise is applied to each feature independently:
    x_noisy = x * (1 + N(0, noise_level))

This model simulates proportional sensor uncertainty — where error
scales with the magnitude of the measurement. This is common for
electrochemical sensors (pH probes, conductivity sensors).

IMPORTANT NOTES:
- Noise is applied ONLY to feature values, NEVER to ground-truth labels.
- The model is fitted on clean training data.
- Noisy versions are only created at evaluation time on the test set.
- The clean test set is the baseline (noise_level = 0.0).

References:
  Leuenberger & Kanevski (2022), Environmental Modelling & Software, 149.
  Chen et al. (2023), IEEE Transactions on Instrumentation and Measurement, 72.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_single_split, save_metrics_json

logger = logging.getLogger(__name__)


def inject_multiplicative_noise(
    X: pd.DataFrame,
    noise_level: float,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Apply multiplicative Gaussian noise to feature values.

    For each sample i and feature j:
        X_noisy[i,j] = X[i,j] * (1 + eps)
        eps ~ N(0, noise_level)

    Parameters
    ----------
    X : pd.DataFrame — clean feature matrix
    noise_level : float — standard deviation of the multiplicative noise.
        0.0 → no noise (identity transform)
        0.05 → 5% proportional noise
    random_seed : int

    Returns
    -------
    pd.DataFrame of the same shape and column names
    """
    if noise_level == 0.0:
        return X.copy()

    rng = np.random.default_rng(random_seed)
    noise = rng.normal(loc=0.0, scale=noise_level, size=X.shape)
    X_noisy = X.values * (1.0 + noise)
    return pd.DataFrame(X_noisy, columns=X.columns, index=X.index)


def run_robustness_experiment(
    fitted_model: Any,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    noise_levels: List[float],
    n_repetitions: int = 5,
    output_dir: str = "results/robustness/",
) -> List[Dict]:
    """
    Evaluate a fitted model at multiple noise levels.

    For each noise level:
    - Repeat `n_repetitions` times with different random seeds.
    - Average performance across repetitions.

    Parameters
    ----------
    fitted_model : sklearn-compatible fitted estimator
    model_name : str
    X_test : pd.DataFrame — clean held-out test features
    y_test : pd.Series — true labels (never modified)
    noise_levels : list of float — noise std levels to evaluate
    n_repetitions : int — number of random seeds per noise level
    output_dir : str

    Returns
    -------
    list of dicts, one per (model, noise_level)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    for noise_level in noise_levels:
        rep_metrics = []
        for rep in range(n_repetitions):
            seed = 100 + rep  # Deterministic seeds per repetition
            X_noisy = inject_multiplicative_noise(X_test, noise_level, random_seed=seed)
            metrics = evaluate_single_split(
                fitted_model, X_noisy, y_test,
                split_name="noisy_test",
                model_name=model_name
            )
            metrics["noise_level"] = noise_level
            metrics["repetition"] = rep
            rep_metrics.append(metrics)

        # Average over repetitions
        key_metrics = [
            "accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "pr_auc"
        ]
        averaged = {
            "model": model_name,
            "noise_level": noise_level,
            "n_repetitions": n_repetitions,
        }
        for m in key_metrics:
            values = [r[m] for r in rep_metrics]
            averaged[f"{m}_mean"] = float(np.mean(values))
            averaged[f"{m}_std"] = float(np.std(values))

        all_results.append(averaged)
        logger.info(
            f"[Robustness] {model_name} @ noise={noise_level:.2f}: "
            f"MacroF1={averaged['f1_macro_mean']:.3f}±{averaged['f1_macro_std']:.3f}"
        )

        save_metrics_json(
            averaged,
            str(Path(output_dir) / f"{model_name}_noise{noise_level:.2f}.json")
        )

    return all_results


def run_full_robustness(
    fitted_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    noise_levels: List[float],
    n_repetitions: int = 5,
    output_dir: str = "results/robustness/",
    model_subset: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run noise robustness experiment for all (or selected) models.

    Parameters
    ----------
    fitted_models : dict of model_name → fitted estimator
    X_test, y_test : clean held-out test data
    noise_levels : list of noise std values
    n_repetitions : int
    output_dir : str
    model_subset : optional list of model names to restrict evaluation

    Returns
    -------
    pd.DataFrame with all robustness results
    """
    all_results = []

    models_to_evaluate = model_subset if model_subset else list(fitted_models.keys())

    for model_name in models_to_evaluate:
        if model_name not in fitted_models:
            logger.warning(f"Model '{model_name}' not found. Skipping.")
            continue
        if model_name == "Dummy":
            continue

        model_results = run_robustness_experiment(
            fitted_model=fitted_models[model_name],
            model_name=model_name,
            X_test=X_test,
            y_test=y_test,
            noise_levels=noise_levels,
            n_repetitions=n_repetitions,
            output_dir=output_dir,
        )
        all_results.extend(model_results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(str(Path(output_dir) / "robustness_results_all.csv"), index=False)
    logger.info(f"Full robustness results saved to {output_dir}/robustness_results_all.csv")
    return results_df
