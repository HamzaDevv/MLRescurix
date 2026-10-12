from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence
import numpy as np
import pandas as pd


class CustomEstimator(ABC):
    """
    Contract for agent-authored custom tabular estimators.
    Must accept raw pandas DataFrame features (X) and pandas Series target (y).
    Must provide predict and optional predict_proba returning pure numpy arrays.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CustomEstimator":
        """Fit estimator to tabular training features and labels."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point predictions for rows in X."""
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Generate class probability predictions for rows in X.
        Return 2D numpy array of shape (n_samples, n_classes) summing to 1.0 along axis 1.
        """
        return None


class CustomTransformer(ABC):
    """
    Contract for agent-authored tabular feature transformers and preprocessors.
    Must accept pandas DataFrame and return transformed pandas DataFrame.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CustomTransformer":
        """Fit transformer to training data without data leakage."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features DataFrame into transformed DataFrame."""
        raise NotImplementedError

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
