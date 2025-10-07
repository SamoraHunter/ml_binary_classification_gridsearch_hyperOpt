# from sklearn.neighbors import KNeighborsClassifier
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from simbsig.neighbors import KNeighborsClassifier
from sklearn import metrics

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

        # Auto-detect device
        gpu_available = torch.cuda.is_available()
        if device == "gpu" and not gpu_available:
            print("Warning: GPU requested for KNNWrapper, but torch.cuda.is_available() is False. Falling back to CPU.")
            self.device = "cpu"
        elif device:
            self.device = device
        else:
            self.device = "gpu" if gpu_available else "cpu"

        self.model: Optional[KNeighborsClassifier] = None

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
        self.model = KNeighborsClassifier(
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
            Dict[str, Any]: Parameter names mapped to their values.
        """
        return {
            "device": self.device,
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "p": self.p,
            "metric": self.metric,
            "metric_params": self.metric_params,
            "n_neighbors": self.n_neighbors,
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
            setattr(self, parameter, value)
        return self
