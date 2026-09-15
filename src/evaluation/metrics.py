"""
src/evaluation/metrics.py
──────────────────────────
Comprehensive model evaluation utilities.

Implements:
- Single-split evaluation (accuracy, precision, recall, F1, ROC-AUC, PR-AUC)
- Cross-validation evaluation with mean ± std reporting
- Confusion matrix generation
- Results serialization to JSON/CSV

IMPORTANT: All evaluation is performed on pre-split data.
The test set is used ONLY for final evaluation after all
hyperparameter tuning and model selection are complete.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Single-split evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_single_split(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str = "test",
    model_name: str = "model",
) -> Dict:
    """
    Evaluate a fitted model on a single data split.

    Parameters
    ----------
    model : fitted sklearn estimator
    X : pd.DataFrame  — feature matrix
    y : pd.Series     — true labels
    split_name : str  — for logging ("val", "test")
    model_name : str  — for logging

    Returns
    -------
    dict with all metrics
    """
    start = time.time()
    y_pred = model.predict(X)
    inference_time_ms = (time.time() - start) * 1000 / len(X)  # per sample

    # Probability scores for AUC metrics
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X)
    else:
        y_prob = y_pred.astype(float)

    metrics = {
        "model": model_name,
        "split": split_name,
        "n_samples": len(y),
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "precision_macro": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "precision_class0": float(precision_score(y, y_pred, pos_label=0, zero_division=0)),
        "recall_class0": float(recall_score(y, y_pred, pos_label=0, zero_division=0)),
        "f1_class0": float(f1_score(y, y_pred, pos_label=0, zero_division=0)),
        "precision_class1": float(precision_score(y, y_pred, pos_label=1, zero_division=0)),
        "recall_class1": float(recall_score(y, y_pred, pos_label=1, zero_division=0)),
        "f1_class1": float(f1_score(y, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "pr_auc": float(average_precision_score(y, y_prob)),
        "inference_time_ms_per_sample": float(inference_time_ms),
    }

    logger.info(
        f"[{model_name} / {split_name}] "
        f"Acc={metrics['accuracy']:.3f}, "
        f"MacroF1={metrics['f1_macro']:.3f}, "
        f"ROC-AUC={metrics['roc_auc']:.3f}"
    )

    return metrics


def get_confusion_matrix(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series
) -> np.ndarray:
    """Return confusion matrix for a fitted model."""
    y_pred = model.predict(X)
    return confusion_matrix(y, y_pred)


def get_classification_report(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series
) -> str:
    """Return sklearn classification report as string."""
    y_pred = model.predict(X)
    return classification_report(y, y_pred, target_names=["Not Potable", "Potable"])


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation evaluation
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_seed: int = 42,
    model_name: str = "model",
) -> Dict:
    """
    Stratified K-Fold cross-validation.

    IMPORTANT: This function must only be called on training data
    (X_train, y_train). The test set must not be passed here.

    Returns dict with mean ± std for each metric.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }

    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    result = {"model": model_name, "n_splits": n_splits}
    for metric_name in scoring.keys():
        key = f"test_{metric_name}"
        values = cv_results[key]
        result[f"{metric_name}_mean"] = float(np.mean(values))
        result[f"{metric_name}_std"] = float(np.std(values))
        result[f"{metric_name}_values"] = [float(v) for v in values]

    logger.info(
        f"[{model_name} / CV-{n_splits}] "
        f"MacroF1={result['f1_macro_mean']:.3f}±{result['f1_macro_std']:.3f}, "
        f"ROC-AUC={result['roc_auc_mean']:.3f}±{result['roc_auc_std']:.3f}"
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Results serialization
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_json(metrics: Dict, output_path: str) -> None:
    """Save a metrics dictionary to a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {output_path}")


def save_metrics_csv(metrics_list: List[Dict], output_path: str) -> None:
    """Save a list of metrics dicts to a CSV file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics_list)
    df.to_csv(output_path, index=False)
    logger.info(f"Metrics table saved: {output_path}")


def format_cv_result_for_display(cv_result: Dict) -> str:
    """Format cross-validation results as a human-readable string."""
    model = cv_result["model"]
    lines = [f"Model: {model}"]
    key_metrics = ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "average_precision"]
    for m in key_metrics:
        mean = cv_result.get(f"{m}_mean", None)
        std = cv_result.get(f"{m}_std", None)
        if mean is not None:
            lines.append(f"  {m}: {mean:.4f} ± {std:.4f}")
    return "\n".join(lines)
