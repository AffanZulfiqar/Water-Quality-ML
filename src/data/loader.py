"""
src/data/loader.py
──────────────────
Utilities for loading the water potability dataset and performing
train/validation/test splitting with leakage prevention.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the project YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_raw_dataset(data_path: str) -> pd.DataFrame:
    """
    Load the raw CSV dataset and perform minimal structural validation.

    Parameters
    ----------
    data_path : str
        Path to the water_potability.csv file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Please download it from:\n"
            "  https://www.kaggle.com/datasets/adityakadiwal/water-potability\n"
            "and place water_potability.csv in the data/ directory.\n\n"
            "Alternatively, run:\n"
            "  kaggle datasets download -d adityakadiwal/water-potability -p data/ --unzip"
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def validate_dataset(df: pd.DataFrame, target_col: str = "Potability") -> None:
    """
    Assert that the dataset has the expected columns and target values.
    Raises ValueError if validation fails.
    """
    expected_features = [
        "pH", "Hardness", "Solids", "Chloramines", "Sulfate",
        "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
    ]
    missing_cols = [c for c in expected_features + [target_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing expected columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )
    unique_targets = set(df[target_col].dropna().unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Target column '{target_col}' contains unexpected values: {unique_targets}"
        )
    logger.info("Dataset validation passed.")


def split_dataset(
    df: pd.DataFrame,
    target_col: str = "Potability",
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Perform stratified train/validation/test split.

    IMPORTANT: The test set is held out and must NOT be used for any
    preprocessing fitting, hyperparameter tuning, or model selection.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including target column.
    target_col : str
        Name of the target column.
    test_size : float
        Proportion for held-out test set.
    validation_size : float
        Proportion for validation set (from remaining data after test split).
    random_seed : int
        Random state for reproducibility.

    Returns
    -------
    X_train, X_val, X_test : pd.DataFrame
    y_train, y_val, y_test : pd.Series
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: separate held-out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y
    )

    # Second split: separate validation from training
    # validation_size is relative to the full dataset; adjust for remaining
    val_proportion_of_temp = validation_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_proportion_of_temp,
        random_state=random_seed,
        stratify=y_temp
    )

    logger.info(
        f"Split sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )
    logger.info(
        f"Class distribution in train: {y_train.value_counts(normalize=True).to_dict()}"
    )
    logger.info(
        f"Class distribution in test: {y_test.value_counts(normalize=True).to_dict()}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_feature_names(df: pd.DataFrame, target_col: str = "Potability") -> list:
    """Return the list of feature column names."""
    return [c for c in df.columns if c != target_col]
