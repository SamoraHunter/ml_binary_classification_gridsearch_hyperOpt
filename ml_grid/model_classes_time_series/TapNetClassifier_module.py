from typing import Any, Dict, List

import numpy as np
import keras
from sklearn.base import BaseEstimator, ClassifierMixin


class _DummyClassifier(BaseEstimator, ClassifierMixin):
    """A dummy classifier to act as a placeholder for a missing model."""

    def __init__(self, kernel_size=None, filter_sizes=None, layers=None, **kwargs):
        self.kernel_size = kernel_size
        self.filter_sizes = filter_sizes
        self.layers = layers

    def fit(self, X, y, **kwargs):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X, **kwargs):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X, **kwargs):
        proba = np.zeros((len(X), len(self.classes_)))
        if proba.shape[1] > 0:
            proba[:, 0] = 1.0
        return proba

    def _fit(self, X, y):
        return self.fit(X, y)

    def _predict(self, X, **kwargs):
        return self.predict(X, **kwargs)

    def _predict_proba(self, X, **kwargs):
        return self.predict_proba(X, **kwargs)


try:
    from aeon.classification.deep_learning import TapNetClassifier
except ImportError:
    try:
        from aeon.classification.deep_learning._tapnet import TapNetClassifier
    except ImportError:
        TapNetClassifier = _DummyClassifier

from skopt.space import Categorical
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


from ml_grid.pipeline.data import pipe
from ml_grid.util.param_space import ParamSpace


class TimeSeriesStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon=1e-6):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.epsilon = epsilon

    def fit(self, X, y=None):
        n_samples, n_dims, n_timesteps = X.shape
        # Reshape to (n_samples * n_timesteps, n_dims) to scale each dimension
        X_reshaped = X.transpose(0, 2, 1).reshape(-1, n_dims)
        # Handle infs/nans before anything else
        X_reshaped = np.where(np.isinf(X_reshaped), np.nan, X_reshaped)
        # Impute missing values before scaling
        X_imputed = self.imputer.fit_transform(X_reshaped)
        self.scaler.fit(X_imputed)
        # If scale is near zero (constant feature), use 1.0 to avoid division by zero and value explosion
        self.scaler.scale_ = np.where(
            self.scaler.scale_ < self.epsilon, 1.0, self.scaler.scale_
        )
        return self

    def transform(self, X):
        n_samples, n_dims, n_timesteps = X.shape
        X_reshaped = X.transpose(0, 2, 1).reshape(-1, n_dims)
        # Handle infs/nans
        X_reshaped = np.where(np.isinf(X_reshaped), np.nan, X_reshaped)
        X_imputed = self.imputer.transform(X_reshaped)
        X_scaled_reshaped = self.scaler.transform(X_imputed)
        # Clip to prevent extreme values (e.g. > 20 sigma) which cause gradient explosion
        X_scaled_reshaped = np.clip(X_scaled_reshaped, -20, 20)
        # Reshape back to original 3D format
        X_scaled = X_scaled_reshaped.reshape(n_samples, n_timesteps, n_dims).transpose(
            0, 2, 1
        )
        return X_scaled


class TapNetClassifierWrapper(TapNetClassifier):
    def fit(self, X, y, **kwargs):
        if isinstance(self.kernel_size, tuple):
            self.kernel_size = list(self.kernel_size)
        if isinstance(self.filter_sizes, tuple):
            self.filter_sizes = list(self.filter_sizes)
        if isinstance(self.layers, tuple):
            self.layers = list(self.layers)

        return super().fit(X, y, **kwargs)

    def _fit(self, X, y):
        if self.metrics is None:
            self._metrics = []
        elif isinstance(self.metrics, str):
            self._metrics = [self.metrics]
        else:
            self._metrics = self.metrics

        # Clone optimizer to ensure a fresh instance for each fit/fold
        if hasattr(self, "optimizer") and isinstance(
            self.optimizer, keras.optimizers.Optimizer
        ):
            self.optimizer = self.optimizer.from_config(self.optimizer.get_config())

        return super()._fit(X, y)

    def _predict_proba(self, X, **kwargs):
        # Safety net: intercept NaNs in probability outputs
        proba = super()._predict_proba(X, **kwargs)
        if np.isnan(proba).any():
            # Replace NaNs with uniform probability (1/n_classes)
            n_classes = len(self.classes_)
            uniform_prob = 1.0 / n_classes
            proba = np.where(np.isnan(proba), uniform_prob, proba)
            # Normalize rows to sum to 1
            row_sums = proba.sum(axis=1)
            # Avoid division by zero if a row is all zeros
            row_sums[row_sums == 0] = 1
            proba = proba / row_sums[:, np.newaxis]
        return proba

    def _predict(self, X, **kwargs):
        # Use our safe predict_proba to generate predictions
        probs = self._predict_proba(X, **kwargs)
        return np.array([self.classes_[np.argmax(prob)] for prob in probs])


class TapNetClassifier_class:
    """A wrapper for the aeon TapNetClassifier time-series classifier.

    This class provides a consistent interface for the TapNetClassifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon TapNetClassifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: TapNetClassifier
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the TapNetClassifier_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """

        verbose_param = ml_grid_object.verbose
        param_space = ParamSpace(
            ml_grid_object.local_param_dict.get("param_space_size")
        )

        log_epoch = param_space.param_dict.get("log_epoch")
        if isinstance(log_epoch, list):
            log_epoch = log_epoch[0]
        random_state_val = ml_grid_object.global_params.random_state_val

        tapnet_model = TapNetClassifierWrapper()
        self.algorithm_implementation = Pipeline(
            [("scaler", TimeSeriesStandardScaler()), ("model", tapnet_model)]
        )
        self.method_name = "TapNetClassifier"

        if getattr(ml_grid_object.global_params, "test_mode", False):
            self.parameter_space = {
                "model__n_epochs": [1],
                "model__layers": [(64, 64)],
                "model__filter_sizes": [(64, 64)],
                "model__kernel_size": [(3, 3)],
                "model__verbose": [0],
                "model__optimizer": [keras.optimizers.Adam(learning_rate=0.001)],
            }
            return

        if ml_grid_object.global_params.bayessearch:
            base_params = {
                "filter_sizes": Categorical([(256, 256, 128), (128, 128, 64)]),
                "kernel_size": Categorical([(8, 5, 3), (4, 3, 2)]),
                "layers": Categorical([(500, 300, 100), (400, 200, 100)]),
                "n_epochs": log_epoch,
                "batch_size": Categorical([16, 32]),
                "dropout": Categorical([0.5, 0.3, 0.2]),
                "use_mini_batch_size": Categorical([False]),
                "dilation": Categorical([1]),
                "activation": Categorical(["sigmoid", "relu"]),
                "loss": Categorical(
                    ["binary_crossentropy", "categorical_crossentropy"]
                ),
                "optimizer": Categorical(
                    [
                        keras.optimizers.Adam(learning_rate=0.00001, clipnorm=1.0),
                        keras.optimizers.SGD(learning_rate=0.00001, clipnorm=1.0),
                    ]
                ),
                "use_bias": Categorical([True, False]),
                "use_rp": Categorical([True, False]),
                "use_att": Categorical([True, False]),
                "use_lstm": Categorical([True, False]),
                "use_cnn": Categorical([True, False]),
                "verbose": Categorical([verbose_param]),
                "random_state": Categorical([random_state_val]),
            }
            self.parameter_space = {
                f"model__{key}": value for key, value in base_params.items()
            }
        else:
            base_params = {
                "filter_sizes": [(256, 256, 128), (128, 128, 64)],
                "kernel_size": [(8, 5, 3), (4, 3, 2)],
                "layers": [(500, 300, 100), (400, 200, 100)],
                "n_epochs": [100],
                "batch_size": [16, 32],
                "dropout": [0.5, 0.3, 0.2],
                "use_mini_batch_size": [False],
                "dilation": [1],
                "activation": ["sigmoid", "relu"],
                "loss": ["binary_crossentropy", "categorical_crossentropy"],
                "optimizer": [
                    keras.optimizers.Adam(learning_rate=0.00001, clipnorm=1.0),
                    keras.optimizers.SGD(learning_rate=0.00001, clipnorm=1.0),
                ],
                "use_bias": [True, False],
                "use_rp": [True, False],
                "use_att": [True, False],
                "use_lstm": [True, False],
                "use_cnn": [True, False],
                "verbose": [verbose_param],
                "random_state": [random_state_val],
            }
            self.parameter_space = {
                f"model__{key}": value for key, value in base_params.items()
            }
