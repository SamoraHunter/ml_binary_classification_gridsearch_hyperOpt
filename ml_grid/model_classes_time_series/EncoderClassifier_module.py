from typing import Any, Dict, List

import numpy as np
import keras
from aeon.classification.deep_learning import EncoderClassifier
from skopt.space import Categorical
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from ml_grid.pipeline.data import pipe


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


class EncoderClassifierWrapper(EncoderClassifier):
    def fit(self, X, y, **kwargs):
        if isinstance(self.kernel_size, tuple):
            self.kernel_size = list(self.kernel_size)
        if isinstance(self.n_filters, tuple):
            self.n_filters = list(self.n_filters)

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


class EncoderClassifier_class:
    """A wrapper for the aeon EncoderClassifier time-series classifier.

    This class provides a consistent interface for the EncoderClassifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon EncoderClassifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: EncoderClassifier
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the EncoderClassifier_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """
        random_state_val = ml_grid_object.global_params.random_state_val

        # Dynamically constrain kernel_size based on the length of the time series.
        # aeon format is (n_samples, n_dims, n_timesteps)
        n_timesteps = ml_grid_object.X_train.shape[2]

        encoder_model = EncoderClassifierWrapper()
        self.algorithm_implementation = Pipeline(
            [("scaler", TimeSeriesStandardScaler()), ("model", encoder_model)]
        )
        self.method_name = "EncoderClassifier"

        if getattr(ml_grid_object.global_params, "test_mode", False):
            self.parameter_space = [
                {
                    "model__n_epochs": [1],
                    "model__n_filters": [(32,)],
                    "model__kernel_size": [(3,)],
                    "model__fc_units": [32],
                    "model__verbose": [0],
                    "model__optimizer": [keras.optimizers.Adam(learning_rate=0.001)],
                }
            ]
            return

        # Common parameters for all configurations
        common_params_bayes = {
            "max_pool_size": Categorical([2, 3]),
            "activation": Categorical(["sigmoid", "relu", "tanh"]),
            "dropout_proba": Categorical([0.0, 0.2, 0.5]),
            "padding": Categorical(["same", "valid"]),
            "strides": Categorical([1, 2]),
            "fc_units": Categorical([128, 256, 512]),
            "save_best_model": [True],
            "save_last_model": [False],
            "random_state": [random_state_val],
            "use_mini_batch_size": Categorical([False]),
            "optimizer": Categorical(
                [
                    keras.optimizers.Adam(learning_rate=0.00001, clipnorm=1.0),
                    keras.optimizers.SGD(learning_rate=0.00001, clipnorm=1.0),
                ]
            ),
        }

        common_params_grid = {
            "max_pool_size": [2, 3],
            "activation": ["sigmoid", "relu", "tanh"],
            "dropout_proba": [0.0, 0.2, 0.5],
            "padding": ["same", "valid"],
            "strides": [1, 2],
            "fc_units": [128, 256, 512],
            "save_best_model": [False],
            "save_last_model": [False],
            "random_state": [random_state_val],
            "use_mini_batch_size": [False],
            "optimizer": [
                keras.optimizers.Adam(learning_rate=0.00001, clipnorm=1.0),
                keras.optimizers.SGD(learning_rate=0.00001, clipnorm=1.0),
            ],
        }

        # Define the full potential parameter space
        full_param_space_bayes = [
            {
                "kernel_size": Categorical([(5,), (11,), (21,)]),
                "n_filters": Categorical([(128,), (256,), (512,)]),
                **common_params_bayes,
            },
            {
                "kernel_size": Categorical([(5, 11), (5, 21), (11, 21)]),
                "n_filters": Categorical([(128, 256), (128, 512), (256, 512)]),
                **common_params_bayes,
            },
            {
                "kernel_size": Categorical([(5, 11, 21)]),
                "n_filters": Categorical([(128, 256, 512)]),
                **common_params_bayes,
            },
        ]

        full_param_space_grid = [
            {
                "kernel_size": [(5,), (11,), (21,)],
                "n_filters": [(128,), (256,), (512,)],
                **common_params_grid,
            },
            {
                "kernel_size": [(5, 11), (5, 21), (11, 21)],
                "n_filters": [(128, 256), (128, 512), (256, 512)],
                **common_params_grid,
            },
            {
                "kernel_size": [(5, 11, 21)],
                "n_filters": [(128, 256, 512)],
                **common_params_grid,
            },
        ]

        # Filter the spaces to only include valid kernel sizes
        filtered_space_bayes = []
        for config in full_param_space_bayes:
            valid_kernels = [
                ks
                for ks in config["kernel_size"].categories
                if all(k <= n_timesteps for k in ks)
            ]
            if valid_kernels:
                new_config = config.copy()
                new_config["kernel_size"] = Categorical(valid_kernels)
                prefixed_config = {
                    f"model__{key}": value for key, value in new_config.items()
                }
                filtered_space_bayes.append(prefixed_config)

        filtered_space_grid = []
        for config in full_param_space_grid:
            valid_kernels = [
                ks for ks in config["kernel_size"] if all(k <= n_timesteps for k in ks)
            ]
            if valid_kernels:
                new_config = config.copy()
                new_config["kernel_size"] = valid_kernels
                prefixed_config = {
                    f"model__{key}": value for key, value in new_config.items()
                }
                filtered_space_grid.append(prefixed_config)

        if ml_grid_object.global_params.bayessearch:
            self.parameter_space = (
                filtered_space_bayes if filtered_space_bayes else [{}]
            )
        else:
            self.parameter_space = filtered_space_grid if filtered_space_grid else [{}]
