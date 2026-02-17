"""LightGBM Classifier.

This module contains the LightGBMClassifierWrapper, which is a configuration
class for the LightGBMClassifier. It provides parameter spaces for
grid search and Bayesian optimization.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from skopt.space import Categorical, Integer, Real

from ml_grid.model_classes.lightgbm_class import LightGBMClassifier
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters


class LightGBMClassifierWrapper:
    """LightGBMClassifier with support for both Bayesian and non-Bayesian parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        """Initializes the LightGBMClassifierWrapper.

        Args:
            X (Optional[pd.DataFrame]): Feature matrix for training. Defaults to None.
            y (Optional[pd.Series]): Target vector for training. Defaults to None.
            parameter_space_size (Optional[str]): Size of the parameter space for
                optimization. Defaults to None.
        """
        self.X = X
        self.y = y

        global_params = global_parameters

        self.algorithm_implementation = (
            LightGBMClassifier()
        )  # Custom scikit-learn wrapper
        self.method_name = "LightGBMClassifier"

        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        self.parameter_space: Union[Dict[str, Any], List[Dict[str, Any]]]

        if global_params.bayessearch:
            self.parameter_space = {
                "boosting_type": Categorical(("gbdt", "dart", "goss")),
                "num_leaves": Integer(2, 100),
                "learning_rate": Real(1e-5, 1e-1, prior="log-uniform"),
                "n_estimators": Integer(50, 500),
                "objective": Categorical(("binary",)),
                "metric": Categorical(("logloss",)),
                "feature_fraction": Real(0.8, 1.0),
            }
        else:
            self.parameter_space = [
                {
                    "boosting_type": ["gbdt", "dart", "goss"],
                    "num_leaves": list(
                        self.parameter_vector_space.param_dict.get("log_large_long", [])
                    ),
                    "learning_rate": list(
                        self.parameter_vector_space.param_dict.get("log_small", [])
                    ),
                    "n_estimators": list(
                        self.parameter_vector_space.param_dict.get("log_large_long", [])
                    ),
                    "objective": ["binary"],
                    "metric": ["logloss"],
                    "feature_fraction": [0.8, 0.9, 1.0],
                }
            ]


logging.getLogger("ml_grid").debug("Imported LightGBM classifier wrapper class")
