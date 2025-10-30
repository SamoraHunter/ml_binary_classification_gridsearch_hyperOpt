# from sklearn.neighbors import KNeighborsClassifier
"""KNN Wrapper for GPU-accelerated KNN.

This module provides a scikit-learn compatible wrapper for the
simbsig.neighbors.KNeighborsClassifier, with GPU support.
"""
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from simbsig.neighbors import KNeighborsClassifier as SimbsigKNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier as SklearnKNeighborsClassifier
from sklearn import metrics
import logging


class KNNWrapper:
    """A scikit-learn compatible wrapper for the GPU-accelerated KNN from simbsig.

    This class allows the `simbsig.neighbors.KNeighborsClassifier` to be used
    as a standard scikit-learn classifier, automatically detecting and using a
    GPU if available.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = "minkowski",
        metric_params: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        """Initializes the KNNWrapper.

        Args:
            n_neighbors (int): Number of neighbors to use.
            weights (str): Weight function used in prediction.
            algorithm (str): Algorithm used to compute the nearest neighbors.
            leaf_size (int): Leaf size passed to BallTree or KDTree.
            p (int): Power parameter for the Minkowski metric.
            metric (str): The distance metric to use for the tree.
            metric_params (Optional[Dict[str, Any]]): Additional keyword arguments
                for the metric function. Defaults to None.
            device (Optional[str]): The device to use ('gpu' or 'cpu'). If None,
                it auto-detects GPU availability. Defaults to None.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self._init_device = device  # Store the original device parameter
        self.device = device

        # Auto-detect device if not specified, or validate if specified
        self._set_device(device)

        self.model: Optional[
            Union[SimbsigKNeighborsClassifier, SklearnKNeighborsClassifier]
        ] = None

    def _set_device(self, device: Optional[str]):
        """Helper to set the device, falling back to CPU if GPU is not available."""
        gpu_available = torch.cuda.is_available()
        if device == "gpu" and not gpu_available:
            logging.getLogger("ml_grid").warning(
                "GPU requested for KNNWrapper, but torch.cuda is not available. Falling back to CPU."
            )
            self.device = "cpu"
        else:
            self.device = device if device else ("gpu" if gpu_available else "cpu")

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> "KNNWrapper":
        """Fits the KNN model.

        Initializes and fits the `simbsig.neighbors.KNeighborsClassifier`.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The training input samples.
            y (Union[pd.Series, np.ndarray]): The target values.

        Returns:
            KNNWrapper: The fitted estimator.
        """
        # If the device is CPU, use the standard scikit-learn implementation
        # to completely avoid any simbsig/torch/cuda calls.
        if self.device == "cpu":
            logging.getLogger("ml_grid").info(
                "Using scikit-learn's KNeighborsClassifier for CPU execution."
            )
            self.model = SklearnKNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                p=self.p,
                metric=self.metric,
                metric_params=self.metric_params,
            )
        else:
            # If GPU is intended and available, use the simbsig implementation.
            self.model = SimbsigKNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                p=self.p,
                metric=self.metric,
                metric_params=self.metric_params,
                device=self.device,
            )
        self.model.fit(X, y)
        return self

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Gets parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and
                contained subobjects that are estimators.

        Returns:
            Dict[str, Any]: Parameter names mapped to their original values.
        """
        return {
            "device": self._init_device,
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "p": self.p,
            "metric": self.metric,
            "metric_params": self.metric_params,
        }

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

    def score(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> float:
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Test samples.
            y (Union[pd.Series, np.ndarray]): True labels for X.

        Returns:
            float: Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return metrics.accuracy_score(y, y_pred)

    def set_params(self, **parameters: Any) -> "KNNWrapper":
        """Sets the parameters of this estimator.

        Args:
            **parameters (Any): Estimator parameters.

        Returns:
            KNNWrapper: The instance with updated parameters.
        """
        for parameter, value in parameters.items():
            # Special handling for device to re-validate availability
            if parameter == "device":
                # Update both the initial and current device setting
                self._init_device = value
                self._set_device(value)
            setattr(self, parameter, value)
        return self
