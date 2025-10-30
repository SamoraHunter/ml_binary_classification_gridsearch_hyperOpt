"""LightGBM Classifier Wrapper.

This module provides a scikit-learn compatible wrapper for the LightGBM classifier,
handling feature name sanitization.
"""
from typing import Any, Optional, Union

import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import re
import numpy as np


class LightGBMClassifier(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible wrapper for the LightGBM classifier.

    This wrapper handles potential issues with special characters in feature
    names that LightGBM does not support by sanitizing column names before
    fitting and predicting.
    """

    def __init__(
        self,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        objective: str = "binary",
        num_class: Optional[int] = None,
        metric: str = "logloss",
        feature_fraction: float = 0.9,
        early_stopping_rounds: Optional[int] = None,
        verbosity: int = -1,
    ):
        """Initializes the LightGBMClassifier wrapper.

        Args:
            boosting_type (str): The type of boosting to use.
            num_leaves (int): Maximum number of leaves in one tree.
            learning_rate (float): Boosting learning rate.
            n_estimators (int): Number of boosting rounds.
            objective (str): The learning objective.
            num_class (Optional[int]): The number of classes for multiclass
                classification. Not needed for binary. Defaults to None.
            metric (str): The metric to be used for evaluation. Defaults to 'logloss'.
            feature_fraction (float): Fraction of features to be considered for each
                tree.
            early_stopping_rounds (Optional[int]): Activates early stopping.
                Defaults to None.
            verbosity (int): Controls the level of LightGBM's verbosity.
        """
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.objective = objective
        self.num_class = num_class
        self.metric = metric
        self.feature_fraction = feature_fraction
        self.early_stopping_rounds = early_stopping_rounds

        self.model: Optional[lgb.LGBMClassifier] = None
        self.verbosity = verbosity
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]
    ) -> "LightGBMClassifier":
        """Fits the LightGBM model.

        This method sanitizes the feature names in `X` before fitting the
        underlying `lgb.LGBMClassifier`.

        Args:
            X (pd.DataFrame): The training input samples.
            y (Union[pd.Series, np.ndarray]): The target values.

        Returns:
            LightGBMClassifier: The fitted estimator.
        """
        # --- ROBUSTNESS FIX for num_leaves > 1 constraint ---
        # This check is moved from __init__ to fit to ensure that values set by
        # hyperparameter search tools (which use set_params) are also validated.
        num_leaves = self.num_leaves if self.num_leaves > 1 else 2

        self.model = lgb.LGBMClassifier(
            boosting_type=self.boosting_type,
            num_leaves=num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            objective=self.objective,
            num_class=self.num_class,
            metric=self.metric,
            feature_fraction=self.feature_fraction,
            # early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbosity,
        )
        
        X_fit = X
        if isinstance(X, pd.DataFrame):
            # Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
            new_names = {col: re.sub(r"[^A-Za-z0-9_]+", "", col) for col in X.columns}
            new_n_list = list(new_names.values())
            # [LightGBM] Feature appears more than one time.
            new_names = {
                col: f"{new_col}_{i}" if new_col in new_n_list[:i] else new_col
                for i, (col, new_col) in enumerate(new_names.items())
            }
            X_fit = X.rename(columns=new_names)

        y = np.ravel(y)

        self.model.fit(X_fit, y)
        if self.objective == "binary":
            self.classes_ = np.unique(y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class labels for samples in X.

        This method sanitizes the feature names in `X` to match those used
        during training.

        Args:
            X (pd.DataFrame): The input samples to predict.

        Raises:
            ValueError: If the model has not been fitted yet.

        Returns:
            np.ndarray: The predicted class labels.
        """
        if self.model is None:
            raise ValueError(
                "Model has not been fitted yet. Call 'fit' before 'predict'."
            )

        X_pred = X
        if isinstance(X, pd.DataFrame):
            # Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
            new_names = {
                col: re.sub(r"[^A-Za-z0-9_]+", "", col) for col in X.columns
            }
            new_n_list = list(new_names.values())
            # [LightGBM] Feature appears more than one time.
            new_names = {
                col: f"{new_col}_{i}" if new_col in new_n_list[:i] else new_col
                for i, (col, new_col) in enumerate(new_names.items())
            }
            X_pred = X.rename(columns=new_names)
        return self.model.predict(X_pred)

    def score(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> float:
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X (pd.DataFrame): Test samples.
            y (Union[pd.Series, np.ndarray]): True labels for X.

        Raises:
            ValueError: If the model has not been fitted yet.

        Returns:
            float: Mean accuracy of self.predict(X) wrt. y.
        """
        if self.model is None:
            raise ValueError(
                "Model has not been fitted yet. Call 'fit' before 'score'."
            )
        return self.model.score(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities for samples in X.

        This method sanitizes the feature names in `X` to match those used
        during training.

        Args:
            X (pd.DataFrame): The input samples to predict.

        Raises:
            ValueError: If the model has not been fitted yet.

        Returns:
            np.ndarray: The predicted class probabilities.
        """
        if self.model is None:
            raise ValueError(
                "Model has not been fitted yet. Call 'fit' before 'predict_proba'."
            )

        X_pred = X
        if isinstance(X, pd.DataFrame):
            # Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
            new_names = {
                col: re.sub(r"[^A-Za-z0-9_]+", "", col) for col in X.columns
            }
            new_n_list = list(new_names.values())
            # [LightGBM] Feature appears more than one time.
            new_names = {
                col: f"{new_col}_{i}" if new_col in new_n_list[:i] else new_col
                for i, (col, new_col) in enumerate(new_names.items())
            }
            X_pred = X.rename(columns=new_names)
        return self.model.predict_proba(X_pred)
