from typing import Any, Dict, List

import numpy as np
import keras
from aeon.classification.deep_learning import IndividualInceptionClassifier
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
        X_reshaped = X.transpose(0, 2, 1).reshape(-1, n_dims)
        X_reshaped = np.where(np.isinf(X_reshaped), np.nan, X_reshaped)
        X_imputed = self.imputer.fit_transform(X_reshaped)
        self.scaler.fit(X_imputed)
        self.scaler.scale_ = np.where(
            self.scaler.scale_ < self.epsilon, 1.0, self.scaler.scale_
        )
        return self

    def transform(self, X):
        n_samples, n_dims, n_timesteps = X.shape
        X_reshaped = X.transpose(0, 2, 1).reshape(-1, n_dims)
        X_reshaped = np.where(np.isinf(X_reshaped), np.nan, X_reshaped)
        X_imputed = self.imputer.transform(X_reshaped)
        X_scaled_reshaped = self.scaler.transform(X_imputed)
        X_scaled_reshaped = np.clip(X_scaled_reshaped, -20, 20)
        X_scaled = X_scaled_reshaped.reshape(n_samples, n_timesteps, n_dims).transpose(
            0, 2, 1
        )
        return X_scaled


class IndividualInceptionClassifierWrapper(IndividualInceptionClassifier):
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
        proba = super()._predict_proba(X, **kwargs)
        if np.isnan(proba).any():
            n_classes = len(self.classes_)
            uniform_prob = 1.0 / n_classes
            proba = np.where(np.isnan(proba), uniform_prob, proba)
            row_sums = proba.sum(axis=1)
            row_sums[row_sums == 0] = 1
            proba = proba / row_sums[:, np.newaxis]
        return proba


class IndividualInceptionClassifier_class:
    """A wrapper for the aeon IndividualInceptionClassifier."""

    algorithm_implementation: IndividualInceptionClassifier
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        random_state_val = ml_grid_object.global_params.random_state_val
        param_space = ParamSpace(
            ml_grid_object.local_param_dict.get("param_space_size")
        )
        log_epoch = param_space.param_dict.get("log_epoch")
        if isinstance(log_epoch, list):
            log_epoch = log_epoch[0]

        model = IndividualInceptionClassifierWrapper()
        self.algorithm_implementation = Pipeline(
            [("scaler", TimeSeriesStandardScaler()), ("model", model)]
        )
        self.method_name = "IndividualInceptionClassifier"

        if getattr(ml_grid_object.global_params, "test_mode", False):
            self.parameter_space = [
                {
                    "model__n_epochs": [1],
                    "model__depth": [2],
                    "model__n_filters": [32],
                    "model__use_residual": [False],
                    "model__verbose": [0],
                    "model__optimizer": [keras.optimizers.Adam(learning_rate=0.001)],
                }
            ]
            return

        # Common parameters
        common_params = {
            "depth": Categorical([6]),
            "n_filters": Categorical([32]),
            "n_conv_per_layer": Categorical([3]),
            "kernel_size": Categorical([40]),
            "use_max_pooling": Categorical([True]),
            "max_pool_size": Categorical([3]),
            "activation": Categorical(["relu"]),
            "use_bias": Categorical([True]),
            "use_bottleneck": Categorical([True]),
            "bottleneck_size": Categorical([32]),
            "use_custom_filters": Categorical([False]),
            "batch_size": Categorical([64]),
            "use_mini_batch_size": Categorical([False]),
            "n_epochs": log_epoch,
            "callbacks": Categorical([None]),
            "save_best_model": Categorical([False]),
            "save_last_model": Categorical([False]),
            "random_state": Categorical([random_state_val]),
            "verbose": Categorical([0]),
            "optimizer": Categorical(
                [keras.optimizers.Adam(learning_rate=0.00001, clipnorm=1.0)]
            ),
            "loss": Categorical(["categorical_crossentropy"]),
            "metrics": Categorical([None]),
        }

        # Residual connection constraints
        residual_params = {
            "use_residual": Categorical([True]),
            "padding": Categorical(["same"]),
            "strides": Categorical([1]),
            **common_params,
        }

        self.parameter_space = [
            {f"model__{k}": v for k, v in residual_params.items()},
        ]
