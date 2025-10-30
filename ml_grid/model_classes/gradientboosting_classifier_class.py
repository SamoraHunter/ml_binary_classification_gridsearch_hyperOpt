"""Gradient Boosting Classifier.

This module contains the GradientBoostingClassifier_class, which is a configuration
class for the GradientBoostingClassifier. It provides parameter spaces for
grid search and Bayesian optimization.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from sklearn.ensemble import GradientBoostingClassifier
from skopt.space import Categorical, Real, Integer


class GradientBoostingClassifierClass:
    """GradientBoostingClassifier with support for both Bayesian and non-Bayesian parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        """Initializes the GradientBoostingClassifierClass.

        Args:
            X (Optional[pd.DataFrame]): Feature matrix for training.
                Defaults to None.
            y (Optional[pd.Series]): Target vector for training.
                Defaults to None.
            parameter_space_size (Optional[str]): Size of the parameter space for
                optimization. Defaults to None.

        Raises:
            ValueError: If `parameter_space_size` is not a valid key (though current
                implementation does not explicitly raise this).
        """
        global_params = global_parameters
        self.X: Optional[pd.DataFrame] = X
        self.y: Optional[pd.Series] = y

        # Use the standard GradientBoostingClassifier directly
        self.algorithm_implementation: GradientBoostingClassifier = (
            GradientBoostingClassifier()
        )
        self.method_name: str = "GradientBoostingClassifier"

        # Define the parameter vector space
        self.parameter_vector_space: param_space.ParamSpace = param_space.ParamSpace(
            parameter_space_size
        )
        self.parameter_space: Dict[str, Any]

        if global_params.bayessearch:
            # Define the parameter space for Bayesian optimization
            self.parameter_space = {
                "ccp_alpha": Real(0.0, 1.0, prior="uniform"),
                "criterion": Categorical(["friedman_mse"]),
                "init": Categorical([None]),
                "learning_rate": Real(1e-5, 1e-1, prior="log-uniform"),
                "loss": Categorical(["log_loss", "exponential"]),
                "max_depth": Integer(2, 10),
                "max_features": Categorical(["sqrt", "log2"]),
                "max_leaf_nodes": Integer(10, 100),
                "min_impurity_decrease": Real(0.0, 0.5, prior="uniform"),
                "min_samples_leaf": Integer(1, 20),
                "min_samples_split": Integer(2, 20),
                "min_weight_fraction_leaf": Real(0.0, 0.5, prior="uniform"),
                "n_estimators": Integer(50, 500),
                "n_iter_no_change": Integer(5, 20),
                "subsample": Real(0.1, 1.0, prior="uniform"),
                "tol": Real(1e-5, 1e-1, prior="log-uniform"),
                "validation_fraction": Real(0.1, 0.3, prior="uniform"),
                "verbose": Categorical([0]),
                "warm_start": Categorical([False]),
            }
        else:
            # Define the parameter space for traditional grid search
            self.parameter_space = {
                "ccp_alpha": list(
                    self.parameter_vector_space.param_dict.get("log_small")
                ),
                "criterion": ["friedman_mse"],
                "init": [None],
                "learning_rate": list(
                    self.parameter_vector_space.param_dict.get("log_small")
                ),
                "loss": ["log_loss", "exponential"],
                # "max_depth": list(range(2, 11)),  # Uncomment if needed
                "max_features": ["sqrt", "log2"],
                # "max_leaf_nodes": list(range(10, 1001)),  # Uncomment if needed
                # "min_impurity_decrease": list(self.parameter_vector_space.param_dict.get("log_small")),  # Uncomment if needed
                # "min_samples_leaf": list(range(1, 11)),  # Uncomment if needed
                # "min_samples_split": list(range(2, 21)),  # Uncomment if needed
                # "min_weight_fraction_leaf": np.linspace(0.0, 0.5, 6).tolist(),  # Uncomment if needed
                "n_estimators": list(
                    self.parameter_vector_space.param_dict.get("log_large_long")
                ),
                # "n_iter_no_change": list(range(5, 51)),  # Uncomment if needed
                "subsample": list(
                    np.delete(
                        self.parameter_vector_space.param_dict.get("lin_zero_one"), 0
                    )
                ),
                "tol": list(self.parameter_vector_space.param_dict.get("log_small")),
                "validation_fraction": [0.1],
                "verbose": [0],
                "warm_start": [False],
            }
