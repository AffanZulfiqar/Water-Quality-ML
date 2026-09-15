"""
src/explainability/shap_analysis.py
─────────────────────────────────────
SHAP-based explainability analysis for the water quality classifiers.

Implements:
- Global SHAP feature importance (summary and bar plots)
- Local SHAP explanations for selected examples
- Cross-model SHAP consistency comparison
- SHAP value persistence for post-hoc analysis

References:
  Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions.
  NeurIPS 30.
  https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning(
        "SHAP library not installed. Install with: pip install shap\n"
        "SHAP analysis will be skipped."
    )


def _check_shap():
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is required for explainability analysis.\n"
            "Install with: pip install shap"
        )


def get_shap_explainer(
    model: Any,
    X_background: pd.DataFrame,
    model_name: str = "model",
    background_samples: int = 100,
) -> Any:
    """
    Create an appropriate SHAP explainer for the given model type.

    Uses TreeExplainer for tree-based models (exact, fast).
    Uses KernelExplainer for other models (approximate, slower).

    Parameters
    ----------
    model : fitted sklearn/XGBoost/LightGBM estimator
    X_background : pd.DataFrame
        Training data used as background for KernelSHAP.
        A random subsample of `background_samples` rows is used.
    model_name : str
        Name string for logging.
    background_samples : int
        Number of background samples for KernelSHAP.

    Returns
    -------
    SHAP Explainer object
    """
    _check_shap()

    tree_model_names = ["RandomForest", "DecisionTree", "XGBoost", "LightGBM"]
    is_tree_model = any(name in model_name for name in tree_model_names)

    if is_tree_model:
        try:
            explainer = shap.TreeExplainer(model)
            logger.info(f"Created TreeExplainer for {model_name}")
            return explainer
        except Exception as e:
            logger.warning(f"TreeExplainer failed for {model_name}: {e}. Falling back to KernelExplainer.")

    # KernelExplainer for non-tree models
    n_bg = min(background_samples, len(X_background))
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_background), size=n_bg, replace=False)
    background = shap.sample(X_background, n_bg)

    def predict_fn(X):
        X_df = pd.DataFrame(X, columns=X_background.columns)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_df)[:, 1]
        return model.predict(X_df).astype(float)

    explainer = shap.KernelExplainer(predict_fn, background)
    logger.info(f"Created KernelExplainer for {model_name} with {n_bg} background samples")
    return explainer


def compute_shap_values(
    explainer: Any,
    X: pd.DataFrame,
    model_name: str = "model",
) -> np.ndarray:
    """
    Compute SHAP values for a dataset.

    Parameters
    ----------
    explainer : SHAP Explainer
    X : pd.DataFrame — data to explain
    model_name : str — for logging

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    _check_shap()
    logger.info(f"Computing SHAP values for {model_name} on {len(X)} samples...")
    shap_vals = explainer.shap_values(X)

    # For TreeExplainer with binary classification, shap_values returns a list [class0, class1]
    if isinstance(shap_vals, list):
        if len(shap_vals) == 2:
            shap_vals = shap_vals[1]  # Use class 1 (potable) SHAP values
        else:
            shap_vals = shap_vals[0]

    logger.info(f"SHAP values computed. Shape: {shap_vals.shape}")
    return shap_vals


def get_global_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Compute global SHAP feature importance as mean absolute SHAP value.

    Parameters
    ----------
    shap_values : np.ndarray of shape (n_samples, n_features)
    feature_names : list of str

    Returns
    -------
    pd.DataFrame with columns ['feature', 'mean_abs_shap', 'rank']
    sorted by importance descending.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    })
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def get_ranked_features(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> List[str]:
    """
    Return feature names ordered by descending mean absolute SHAP value.
    Used for feature ablation configuration.
    """
    importance_df = get_global_feature_importance(shap_values, feature_names)
    return importance_df["feature"].tolist()


def select_local_examples(
    model: Any,
    X: pd.DataFrame,
    y_true: pd.Series,
    n_correct: int = 2,
    n_fp: int = 2,
    n_fn: int = 2,
    n_high_conf: int = 2,
) -> Dict[str, pd.DataFrame]:
    """
    Select representative examples for local SHAP explanation.

    Categories:
    - 'correct': correctly classified samples
    - 'false_positive': predicted potable but actually not potable
    - 'false_negative': predicted not potable but actually potable
    - 'high_confidence': samples with highest predicted probability for class 1

    Parameters
    ----------
    model : fitted model with predict_proba
    X : pd.DataFrame
    y_true : pd.Series
    n_* : int — number of examples for each category

    Returns
    -------
    dict mapping category name → pd.DataFrame of selected rows
    """
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    y_true_arr = y_true.values
    idx = np.arange(len(y_true_arr))

    correct_mask = (y_pred == y_true_arr)
    fp_mask = (y_pred == 1) & (y_true_arr == 0)
    fn_mask = (y_pred == 0) & (y_true_arr == 1)

    rng = np.random.default_rng(42)

    def sample_idx(mask, n):
        candidates = idx[mask]
        if len(candidates) == 0:
            return pd.DataFrame()
        chosen = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
        return X.iloc[chosen].copy()

    correct_examples = sample_idx(correct_mask, n_correct)
    fp_examples = sample_idx(fp_mask, n_fp)
    fn_examples = sample_idx(fn_mask, n_fn)

    # High confidence: top probability scores
    top_conf_idx = np.argsort(y_prob)[-n_high_conf:][::-1]
    high_conf_examples = X.iloc[top_conf_idx].copy()

    examples = {
        "correct": correct_examples,
        "false_positive": fp_examples,
        "false_negative": fn_examples,
        "high_confidence": high_conf_examples,
    }

    for cat, df_ex in examples.items():
        logger.info(f"Local examples — {cat}: {len(df_ex)} samples selected")

    return examples


def save_shap_values(
    shap_values: np.ndarray,
    feature_names: List[str],
    output_path: str,
    model_name: str = "model",
) -> None:
    """Save SHAP values to a CSV file for reproducibility."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(shap_values, columns=feature_names)
    df.to_csv(output_path, index=False)
    logger.info(f"SHAP values saved: {output_path}")


def compare_feature_rankings(
    rankings_dict: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Compare SHAP feature rankings across multiple models.

    Parameters
    ----------
    rankings_dict : dict mapping model_name → list of features (ranked best first)

    Returns
    -------
    pd.DataFrame where rows are ranks and columns are model names.
    """
    max_len = max(len(v) for v in rankings_dict.values())
    rows = []
    for rank in range(1, max_len + 1):
        row = {"rank": rank}
        for model_name, ranked_features in rankings_dict.items():
            if rank - 1 < len(ranked_features):
                row[model_name] = ranked_features[rank - 1]
            else:
                row[model_name] = None
        rows.append(row)
    return pd.DataFrame(rows)
