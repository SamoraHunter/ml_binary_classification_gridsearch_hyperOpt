from typing import Any, Dict, List
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class _DummyClassifier(BaseEstimator, ClassifierMixin):
    """A dummy classifier to act as a placeholder for a missing model."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        proba = np.zeros((len(X), len(self.classes_)))
        if proba.shape[1] > 0:
            proba[:, 0] = 1.0
        return proba


try:
    from aeon.classification.distance_based import ShapeDTW
except ImportError:
    # ShapeDTW was removed in aeon v0.11.0. Use a dummy placeholder.
    ShapeDTW = _DummyClassifier
from ml_grid.pipeline.data import pipe


class ShapeDTW_class:
    """A wrapper for the aeon ShapeDTW time-series classifier.

    This class provides a consistent interface for the ShapeDTW classifier,
    including defining a hyperparameter search space. This classifier is
    intended for univariate time series only.

    Attributes:
        algorithm_implementation: An instance of the aeon ShapeDTW classifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: Any
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the ShapeDTW_class.
        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """
        if ShapeDTW == _DummyClassifier:
            ml_grid_object.logger.warning(
                "ShapeDTW not found in aeon (removed in v0.11.0). "
                "Using a dummy classifier to prevent a crash. This model will be skipped with a low score."
            )
            self.algorithm_implementation = _DummyClassifier()
            self.method_name = "ShapeDTW"
            self.parameter_space = {}
        else:
            n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

            self.algorithm_implementation = ShapeDTW()
            self.method_name = "ShapeDTW"
            self.parameter_space = {
                "shape_descriptor_function": ["raw", "derivative", "paa", "dwt"],
                "n_jobs": [n_jobs_model_val],
            }
