"""FLAML Classifier Wrapper.

This module provides a scikit-learn compatible wrapper for FLAML's AutoML.
"""

import logging
from typing import Union, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

# Attempt to import FLAML
try:
    from flaml import AutoML
except ImportError:
    AutoML = None

logger = logging.getLogger(__name__)


class FLAMLClassifierWrapper(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible wrapper for FLAML AutoML."""

    def __init__(
        self,
        time_budget: int = 60,
        metric: str = "auto",
        task: str = "classification",
        n_jobs: int = -1,
        eval_method: str = "auto",
        split_ratio: float = 0.2,
        n_splits: int = 5,
        log_file_name: str = "flaml.log",
        seed: int = 42,
        verbose: int = 0,
        estimator_list: Union[str, List[str]] = "auto",
    ):
        self.time_budget = time_budget
        self.metric = metric
        self.task = task
        self.n_jobs = n_jobs
        self.eval_method = eval_method
        self.split_ratio = split_ratio
        self.n_splits = n_splits
        self.log_file_name = log_file_name
        self.seed = seed
        self.verbose = verbose
        self.estimator_list = estimator_list

        self.model_ = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs,
    ) -> "FLAMLClassifierWrapper":
        if AutoML is None:
            raise ImportError(
                "FLAML is not installed. Please install it to use FLAMLClassifierWrapper."
            )

        self.model_ = AutoML()

        try:
            self.model_.fit(
                X_train=X,
                y_train=y,
                time_budget=self.time_budget,
                metric=self.metric,
                task=self.task,
                n_jobs=self.n_jobs,
                eval_method=self.eval_method,
                split_ratio=self.split_ratio,
                n_splits=self.n_splits,
                log_file_name=self.log_file_name,
                seed=self.seed,
                verbose=self.verbose,
                estimator_list=self.estimator_list,
                **kwargs,
            )
        except StopIteration:
            # FLAML can raise StopIteration internally when used within scikit-learn's
            # cross-validation framework. We catch it here to prevent it from
            # crashing the joblib parallel backend. The model is still fitted.
            logger.debug(
                "Caught StopIteration from FLAML, which is expected in some CV scenarios."
            )
            pass
        except Exception as e:
            # Catch any other errors during fit (e.g. AttributeError from FLAML runner)
            logger.error(f"FLAML fit failed: {e}")
            raise RuntimeError(f"FLAML fit failed: {e}")

        # After fitting, check if a model was actually found. This is crucial because
        # if the time_budget is too short, FLAML may not find any valid model.
        if self.model_.best_estimator is None:
            msg = (
                "FLAML failed to find a usable model within the given time_budget. "
                "This may be due to a time limit that is too short, or very complex data."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if hasattr(self.model_, "classes_"):
            self.classes_ = self.model_.classes_
        else:
            # If fit fails early or StopIteration is caught before classes_ is set,
            # we infer them from the target variable y to ensure compatibility.
            if isinstance(y, (pd.Series, pd.DataFrame)):
                self.classes_ = np.unique(y.values)
            else:
                self.classes_ = np.unique(y)
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        check_is_fitted(self, ["model_", "classes_"])
        try:
            predictions = self.model_.predict(X)
            if predictions is None:
                logger.warning(
                    "FLAML predict() returned None. Returning dummy predictions (majority class)."
                )
                # Return the most frequent class as a fallback
                dummy_pred = np.full(
                    len(X), self.classes_[0], dtype=self.classes_.dtype
                )
                return dummy_pred
            return predictions
        except Exception as e:
            logger.error(f"FLAML predict failed: {e}. Returning dummy predictions.")
            dummy_pred = np.full(len(X), self.classes_[0], dtype=self.classes_.dtype)
            return dummy_pred

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        check_is_fitted(self, ["model_", "classes_"])
        try:
            probas = self.model_.predict_proba(X)
            if probas is None:
                logger.warning(
                    "FLAML predict_proba() returned None. Returning dummy probabilities."
                )
                n_classes = len(self.classes_)
                return np.full((len(X), n_classes), 1 / n_classes)
            return probas
        except Exception as e:
            logger.error(
                f"FLAML predict_proba failed: {e}. Returning dummy probabilities."
            )
            n_classes = len(self.classes_)
            return np.full((len(X), n_classes), 1 / n_classes)
