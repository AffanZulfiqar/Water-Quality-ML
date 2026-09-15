"""
src/preprocessing/pipeline.py
──────────────────────────────
Reproducible preprocessing pipeline for the water quality dataset.

CRITICAL LEAKAGE PREVENTION RULES:
1. All statistics (median, IQR, scaler parameters) are ONLY computed from
   training data.
2. The same statistics are applied to validation and test sets WITHOUT
   re-fitting.
3. Nothing from the validation or test sets influences preprocessing decisions.
"""

import logging
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class DuplicateRemover(BaseEstimator, TransformerMixin):
    """Remove exact duplicate rows. Only applied during fit (training)."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Duplicates are removed during the data-loading phase, not here.
        # This transformer is a no-op placeholder.
        return X


class MedianImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values with the training-set median.
    Fitted on training data only; applied to all splits.
    """

    def __init__(self):
        self.medians_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y=None):
        self.medians_ = X.median()
        logger.info(f"MedianImputer fitted. Missing value statistics (medians):\n{self.medians_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise ValueError("MedianImputer must be fitted before transform.")
        X = X.copy()
        for col in X.columns:
            n_missing = X[col].isna().sum()
            if n_missing > 0:
                X[col] = X[col].fillna(self.medians_[col])
                logger.debug(f"Imputed {n_missing} missing values in '{col}' with median={self.medians_[col]:.4f}")
        return X


class IQROutlierCapper(BaseEstimator, TransformerMixin):
    """
    Cap outliers at [Q1 - 1.5*IQR, Q3 + 1.5*IQR] computed from training data.
    Values beyond the fence are clipped to the fence value.
    """

    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier
        self.lower_bounds_: Optional[pd.Series] = None
        self.upper_bounds_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y=None):
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        self.lower_bounds_ = Q1 - self.multiplier * IQR
        self.upper_bounds_ = Q3 + self.multiplier * IQR
        logger.info("IQROutlierCapper fitted on training set.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.lower_bounds_ is None:
            raise ValueError("IQROutlierCapper must be fitted before transform.")
        X = X.copy()
        for col in X.columns:
            n_below = (X[col] < self.lower_bounds_[col]).sum()
            n_above = (X[col] > self.upper_bounds_[col]).sum()
            if n_below + n_above > 0:
                X[col] = X[col].clip(
                    lower=self.lower_bounds_[col],
                    upper=self.upper_bounds_[col]
                )
                logger.debug(f"'{col}': capped {n_below} low, {n_above} high outliers.")
        return X


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Wrap scikit-learn scalers and preserve DataFrame column names.
    """

    def __init__(self, method: str = "StandardScaler"):
        self.method = method
        self._scaler = None
        self.feature_names_: Optional[List[str]] = None

    def _get_scaler(self):
        if self.method == "StandardScaler":
            return StandardScaler()
        elif self.method == "RobustScaler":
            return RobustScaler()
        elif self.method == "MinMaxScaler":
            return MinMaxScaler()
        elif self.method == "none":
            return None
        else:
            raise ValueError(f"Unknown scaler: {self.method}")

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_ = list(X.columns)
        self._scaler = self._get_scaler()
        if self._scaler is not None:
            self._scaler.fit(X.values)
        logger.info(f"FeatureScaler fitted: {self.method}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._scaler is None:
            return X
        scaled = self._scaler.transform(X.values)
        return pd.DataFrame(scaled, columns=self.feature_names_, index=X.index)


class WaterQualityPreprocessor:
    """
    High-level preprocessing manager for the water quality dataset.

    Usage:
        preprocessor = WaterQualityPreprocessor(config)
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)

    NEVER call preprocessor.fit() on val or test data.
    """

    def __init__(self, config: dict):
        self.config = config
        prep_cfg = config.get("preprocessing", {})
        self.missing_strategy = prep_cfg.get("missing_value_strategy", "median")
        self.outlier_handling = prep_cfg.get("outlier_handling", "iqr_cap")
        self.scaler_method = prep_cfg.get("scaler", "StandardScaler")

        # Initialize pipeline components
        self.imputer = MedianImputer()
        self.outlier_capper = IQROutlierCapper() if self.outlier_handling == "iqr_cap" else None
        self.scaler = FeatureScaler(method=self.scaler_method)
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "WaterQualityPreprocessor":
        """Fit all pipeline components on training data ONLY."""
        logger.info("Fitting preprocessing pipeline on training data...")
        X_temp = X.copy()
        self.imputer.fit(X_temp)
        X_temp = self.imputer.transform(X_temp)
        if self.outlier_capper is not None:
            self.outlier_capper.fit(X_temp)
            X_temp = self.outlier_capper.transform(X_temp)
        self.scaler.fit(X_temp)
        self._fitted = True
        logger.info("Preprocessing pipeline fitted successfully.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformations to any data split."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        X_out = self.imputer.transform(X.copy())
        if self.outlier_capper is not None:
            X_out = self.outlier_capper.transform(X_out)
        X_out = self.scaler.transform(X_out)
        return X_out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit and transform training data in one step."""
        self.fit(X)
        return self.transform(X)


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove exact duplicate rows from the full dataset before splitting.
    Returns the cleaned dataframe and the number of rows removed.
    """
    n_before = len(df)
    df_clean = df.drop_duplicates()
    n_removed = n_before - len(df_clean)
    logger.info(f"Duplicate removal: {n_removed} rows removed ({n_before} → {len(df_clean)})")
    return df_clean, n_removed
