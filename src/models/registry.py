"""
src/models/registry.py
───────────────────────
Model factory and registry for all benchmark classifiers.

Provides a consistent interface for instantiating all models
with default hyperparameters from the config file.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

def build_models(config: dict, random_seed: int = 42) -> Dict[str, Any]:
    """
    Instantiate all benchmark models with settings from config.

    Parameters
    ----------
    config : dict
        Full project config (loaded from configs/config.yaml).
    random_seed : int
        Random state for reproducibility.

    Returns
    -------
    dict mapping model name → fitted-ready sklearn estimator
    """
    mc = config.get("models", {})

    models = {}

    # ── 1. Dummy (majority-class baseline) ───────────────────────────────────
    models["Dummy"] = DummyClassifier(
        strategy="most_frequent",
        random_state=random_seed
    )

    # ── 2. Logistic Regression ────────────────────────────────────────────────
    lr_cfg = mc.get("LogisticRegression", {})
    models["LogisticRegression"] = LogisticRegression(
        C=lr_cfg.get("C", 1.0),
        max_iter=lr_cfg.get("max_iter", 1000),
        class_weight=lr_cfg.get("class_weight", "balanced"),
        random_state=random_seed,
        solver="lbfgs",
        n_jobs=-1
    )

    # ── 3. Decision Tree ──────────────────────────────────────────────────────
    dt_cfg = mc.get("DecisionTree", {})
    models["DecisionTree"] = DecisionTreeClassifier(
        max_depth=dt_cfg.get("max_depth", 6),
        min_samples_leaf=dt_cfg.get("min_samples_leaf", 10),
        class_weight=dt_cfg.get("class_weight", "balanced"),
        random_state=random_seed
    )

    # ── 4. Random Forest ──────────────────────────────────────────────────────
    rf_cfg = mc.get("RandomForest", {})
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=rf_cfg.get("n_estimators", 200),
        max_depth=rf_cfg.get("max_depth", None),
        min_samples_leaf=rf_cfg.get("min_samples_leaf", 5),
        class_weight=rf_cfg.get("class_weight", "balanced"),
        n_jobs=rf_cfg.get("n_jobs", -1),
        random_state=random_seed
    )

    # ── 5. Support Vector Machine ─────────────────────────────────────────────
    svm_cfg = mc.get("SVM", {})
    models["SVM"] = SVC(
        C=svm_cfg.get("C", 1.0),
        kernel=svm_cfg.get("kernel", "rbf"),
        gamma=svm_cfg.get("gamma", "scale"),
        class_weight=svm_cfg.get("class_weight", "balanced"),
        probability=True,          # Required for ROC-AUC computation
        random_state=random_seed
    )

    # ── 6. XGBoost ────────────────────────────────────────────────────────────
    if XGBOOST_AVAILABLE:
        xgb_cfg = mc.get("XGBoost", {})
        models["XGBoost"] = XGBClassifier(
            n_estimators=xgb_cfg.get("n_estimators", 300),
            max_depth=xgb_cfg.get("max_depth", 6),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            scale_pos_weight=xgb_cfg.get("scale_pos_weight", 1.5),
            eval_metric="logloss",
            random_state=random_seed,
            verbosity=0
        )
    else:
        logger.warning("XGBoost not installed. Skipping XGBoost model.")

    # ── 7. LightGBM ───────────────────────────────────────────────────────────
    if LIGHTGBM_AVAILABLE:
        lgbm_cfg = mc.get("LightGBM", {})
        models["LightGBM"] = LGBMClassifier(
            n_estimators=lgbm_cfg.get("n_estimators", 300),
            max_depth=lgbm_cfg.get("max_depth", -1),
            learning_rate=lgbm_cfg.get("learning_rate", 0.05),
            num_leaves=lgbm_cfg.get("num_leaves", 31),
            class_weight=lgbm_cfg.get("class_weight", "balanced"),
            n_jobs=lgbm_cfg.get("n_jobs", -1),
            verbosity=-1,
            random_state=random_seed
        )
    else:
        logger.warning("LightGBM not installed. Skipping LightGBM model.")

    # ── 8. MLP Neural Network ─────────────────────────────────────────────────
    mlp_cfg = mc.get("MLP", {})
    hidden_layers = mlp_cfg.get("hidden_layer_sizes", [128, 64, 32])
    models["MLP"] = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layers),
        activation=mlp_cfg.get("activation", "relu"),
        max_iter=mlp_cfg.get("max_iter", 500),
        early_stopping=mlp_cfg.get("early_stopping", True),
        validation_fraction=mlp_cfg.get("validation_fraction", 0.1),
        random_state=random_seed,
        learning_rate_init=1e-3
    )

    logger.info(f"Built {len(models)} models: {list(models.keys())}")
    return models


def get_hyperparameter_search_spaces() -> Dict[str, Dict]:
    """
    Define hyperparameter search spaces for RandomizedSearchCV.
    All spaces documented here for reproducibility.
    """
    spaces = {
        "RandomForest": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 8, 12, 16, 20],
            "min_samples_leaf": [1, 2, 5, 10],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2", 0.5, 0.7],
            "class_weight": ["balanced", "balanced_subsample"]
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "scale_pos_weight": [1.0, 1.5, 2.0]
        },
        "LightGBM": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [-1, 6, 8, 10, 12],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "num_leaves": [20, 31, 50, 70, 100],
            "min_child_samples": [5, 10, 20, 30],
            "reg_alpha": [0, 0.01, 0.1],
            "reg_lambda": [0, 0.01, 0.1, 1.0],
            "subsample": [0.7, 0.8, 0.9, 1.0]
        }
    }
    return spaces
