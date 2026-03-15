from typing import Any, Dict, List

import numpy as np
import keras
from aeon.classification.deep_learning import (
    InceptionTimeClassifier,
)
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
        # Assumes input X is in aeon format: (n_instances, n_channels, n_timepoints)
        n_samples, n_channels, n_timepoints = X.shape
        # Reshape to (all_timepoints, n_channels) for scaling
        X_reshaped = X.transpose(0, 2, 1).reshape(-1, n_channels)
        X_reshaped = np.where(np.isinf(X_reshaped), np.nan, X_reshaped)
        X_imputed = self.imputer.fit_transform(X_reshaped)
        self.scaler.fit(X_imputed)
        # Prevent division by zero for constant features
        self.scaler.scale_ = np.where(
            self.scaler.scale_ < self.epsilon, 1.0, self.scaler.scale_
        )
        return self

    def transform(self, X):
        # Assumes input X is in aeon format: (n_instances, n_channels, n_timepoints)
        n_samples, n_channels, n_timepoints = X.shape
        # Reshape to (all_timepoints, n_channels) for scaling
        X_reshaped = X.transpose(0, 2, 1).reshape(-1, n_channels)
        X_reshaped = np.where(np.isinf(X_reshaped), np.nan, X_reshaped)
        X_imputed = self.imputer.transform(X_reshaped)
        X_scaled_reshaped = self.scaler.transform(X_imputed)
        # Clip extreme values after scaling to improve stability for deep learning models
        X_scaled_reshaped = np.clip(X_scaled_reshaped, -20, 20)
        # Reshape back to the standard aeon format: (instances, channels, timepoints)
        X_scaled = X_scaled_reshaped.reshape(
            n_samples, n_timepoints, n_channels
        ).transpose(0, 2, 1)
        return X_scaled


# The wrapper class is kept for pipeline compatibility, but its method overrides
# are removed. The centralized patches in `grid_search_cross_validate_ts.py`
# now handle all state and prediction issues for aeon models, including this one.
class InceptionTimeClassifierWrapper(InceptionTimeClassifier):
    pass


class InceptionTimeClassifier_class:
    """A wrapper for the aeon InceptionTimeClassifier.

    This class provides a consistent interface for the InceptionTimeClassifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon
            InceptionTimeClassifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: InceptionTimeClassifier
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the InceptionTimeClassifier_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """

        random_state_val = ml_grid_object.global_params.random_state_val

        param_space = ParamSpace(
            ml_grid_object.local_param_dict.get("param_space_size")
        )

        log_epoch = param_space.param_dict.get("log_epoch")
        if isinstance(log_epoch, list):
            log_epoch = log_epoch[0]

        inception_model = InceptionTimeClassifierWrapper()
        self.algorithm_implementation = Pipeline(
            [("scaler", TimeSeriesStandardScaler()), ("model", inception_model)]
        )
        self.method_name = "InceptionTimeClassifier"

        if getattr(ml_grid_object.global_params, "test_mode", False):
            self.parameter_space = [
                {
                    "model__n_epochs": [1],
                    "model__n_classifiers": [1],
                    "model__depth": [1],
                    "model__n_filters": [32],
                    "model__use_residual": [False],
                    "model__verbose": [0],
                    "model__optimizer": [keras.optimizers.Adam(learning_rate=0.001)],
                }
            ]
            return

        # Common parameters for all configurations
        common_params_bayes = {
            "n_classifiers": Categorical([3, 5]),
            "depth": Categorical([4, 6]),
            "n_filters": Categorical([32, 64]),
            "n_conv_per_layer": Categorical([3, 4]),
            "kernel_size": Categorical([30, 40]),
            "max_pool_size": Categorical([2, 3]),
            "dilation_rate": Categorical([1, 2]),
            "activation": Categorical(["relu", "tanh"]),
            "use_bias": Categorical([True]),
            "use_bottleneck": Categorical([True, False]),
            "bottleneck_size": Categorical([16, 32]),
            "use_custom_filters": Categorical([False]),
            "batch_size": Categorical([32, 64]),
            "use_mini_batch_size": Categorical([False]),
            "n_epochs": log_epoch,
            "callbacks": Categorical([None]),
            "save_best_model": Categorical([False]),
            "save_last_model": Categorical([False]),
            "random_state": Categorical([random_state_val]),
            "verbose": Categorical([0]),
            "optimizer": Categorical(
                [
                    keras.optimizers.Adam(learning_rate=0.00001, clipnorm=1.0),
                    keras.optimizers.SGD(learning_rate=0.00001, clipnorm=1.0),
                ]
            ),
            "loss": Categorical(["categorical_crossentropy"]),
            "metrics": Categorical([None]),
        }

        common_params_grid = {
            "n_classifiers": [3, 5],
            "depth": [4, 6],
            "n_filters": [32, 64],
            "n_conv_per_layer": [3],
            "kernel_size": [40],
            "max_pool_size": [3],
            "dilation_rate": [1],
            "activation": ["relu"],
            "use_bias": [True],
            "use_bottleneck": [True, False],
            "bottleneck_size": [32],
            "use_custom_filters": [False],  # Already False, but enforce for clarity
            "batch_size": [32],
            "use_mini_batch_size": [False],
            "n_epochs": [100],
            "callbacks": [None],
            "save_best_model": [False],
            "save_last_model": [False],
            "random_state": [random_state_val],
            "verbose": [0],
            "optimizer": [keras.optimizers.Adam(learning_rate=0.00001, clipnorm=1.0)],
            "loss": ["categorical_crossentropy"],
            "metrics": [None],
        }

        if ml_grid_object.global_params.bayessearch:
            # Configuration for models WITH residual connections (requires fixed padding/strides)
            residual_params = {
                "use_residual": Categorical([True]),
                "padding": Categorical(["same"]),
                "strides": Categorical([1]),
                **common_params_bayes,
            }
            # Configuration for models WITHOUT residual connections
            non_residual_params = {
                "use_residual": Categorical([False]),
                "padding": Categorical(["same", "valid"]),
                "strides": Categorical([1, 2]),
                **common_params_bayes,
            }
            self.parameter_space = [
                {f"model__{k}": v for k, v in residual_params.items()},
                {f"model__{k}": v for k, v in non_residual_params.items()},
            ]
        else:
            # Configuration for models WITH residual connections
            residual_params = {
                "use_residual": [True],
                "padding": ["same"],
                "strides": [1],
                **common_params_grid,
            }
            # Configuration for models WITHOUT residual connections
            non_residual_params = {
                "use_residual": [False],
                "padding": ["same", "valid"],
                "strides": [1, 2],
                **common_params_grid,
            }
            self.parameter_space = [
                {f"model__{k}": v for k, v in residual_params.items()},
                {f"model__{k}": v for k, v in non_residual_params.items()},
            ]
