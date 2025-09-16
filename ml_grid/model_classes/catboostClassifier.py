from typing import Any, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class CatBoostSKLearnWrapper(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible wrapper for the CatBoostClassifier."""

    def __init__(self, **kwargs: Any):
        """Initializes the CatBoostSKLearnWrapper.

        Args:
            **kwargs (Any): Keyword arguments passed directly to the
                `catboost.CatBoostClassifier`.
        """
        self.model = CatBoostClassifier(**kwargs)

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> "CatBoostSKLearnWrapper":
        """Fits the CatBoost model.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The training input samples.
            y (Union[pd.Series, np.ndarray]): The target values.

        Returns:
            CatBoostSKLearnWrapper: The fitted estimator.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts class labels for samples in X.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The input samples to predict.

        Returns:
            np.ndarray: The predicted class labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts class probabilities for samples in X.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The input samples.

        Returns:
            np.ndarray: The class probabilities of the input samples.
        """
        return self.model.predict_proba(X)
