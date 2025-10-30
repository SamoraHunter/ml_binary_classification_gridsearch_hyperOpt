"""Logistic Regression Classifier.

This module contains the LogisticRegression_class, which is a configuration
class for the LogisticRegression. It provides parameter spaces for
grid search and Bayesian optimization.
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from ml_grid.util import param_space
from sklearn.linear_model import LogisticRegression
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Real, Integer
import logging

logging.getLogger('ml_grid').debug("Imported logistic regression class")

class LogisticRegressionClass:
    """LogisticRegression with support for both Bayesian and non-Bayesian parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        """Initializes the LogisticRegressionClass.

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

        # Set the base implementation
        self.algorithm_implementation: LogisticRegression = LogisticRegression()
        self.method_name: str = "LogisticRegression"

        self.parameter_vector_space: param_space.ParamSpace = param_space.ParamSpace(
            parameter_space_size
        )

        self.parameter_space: List[Dict[str, Any]]

        # Define parameter space based on search type
        if global_params.bayessearch:
            # Bayesian search parameter definitions as a list of parameter spaces
            self.parameter_space = [
                # ElasticNet penalty: solver must be 'saga'
                {
                    "C": Real(1e-5, 1e-2, prior="log-uniform"),
                    "class_weight": Categorical([None, "balanced"]),
                    "dual": Categorical([False]),
                    "fit_intercept": Categorical([True]),
                    "intercept_scaling": Real(0.1, 10.0, prior="log-uniform"),
                    "l1_ratio": Real(0.0, 1.0, prior="uniform"),  # For elasticnet penalty only
                    "max_iter": Integer(100, 1000),
                    "multi_class": Categorical(["auto", "ovr", "multinomial"]),
                    "n_jobs": Categorical([None, -1]),
                    "penalty": Categorical(["elasticnet"]),
                    "solver": Categorical(["saga"]),
                    "tol": Real(1e-5, 1e-2, prior="log-uniform"),
                    "verbose": Categorical([0]),
                    "warm_start": Categorical([False]),
                },
                # L1 penalty: solver must be 'saga'
                {
                    "C": Real(1e-5, 1e-2, prior="log-uniform"),
                    "class_weight": Categorical([None, "balanced"]),
                    "dual": Categorical([False]),
                    "fit_intercept": Categorical([True]),
                    "intercept_scaling": Real(0.1, 10.0, prior="log-uniform"),
                    "l1_ratio": Categorical([None]),  # No l1_ratio for l1 penalty
                    "max_iter": Integer(100, 1000),
                    "multi_class": Categorical(["auto", "ovr", "multinomial"]),
                    "n_jobs": Categorical([None, -1]),
                    "penalty": Categorical(["l1"]),
                    "solver": Categorical(["saga"]),
                    "tol": Real(1e-5, 1e-2, prior="log-uniform"),
                    "verbose": Categorical([0]),
                    "warm_start": Categorical([False]),
                },
                # L2 penalty: solver can be 'saga', 'newton-cg', or 'lbfgs'
                {
                    "C": Real(1e-5, 1e-2, prior="log-uniform"),
                    "class_weight": Categorical([None, "balanced"]),
                    "dual": Categorical([False]),
                    "fit_intercept": Categorical([True]),
                    "intercept_scaling": Real(0.1, 10.0, prior="log-uniform"),
                    "l1_ratio": Categorical([None]),  # No l1_ratio for l2 penalty
                    "max_iter": Integer(100, 1000),
                    "multi_class": Categorical(["auto", "ovr", "multinomial"]),
                    "n_jobs": Categorical([None, -1]),
                    "penalty": Categorical(["l2"]),
                    "solver": Categorical(["newton-cg", "lbfgs", "saga"]),
                    "tol": Real(1e-5, 1e-2, prior="log-uniform"),
                    "verbose": Categorical([0]),
                    "warm_start": Categorical([False]),
                },
            ]
        else:
            # Grid search parameter definitions as a list of parameter spaces
            self.parameter_space = [
                {
                    "C": self.parameter_vector_space.param_dict.get("log_small"),
                    "class_weight": [None, "balanced"],
                    "dual": [False],
                    "fit_intercept": [True],
                    "intercept_scaling": [1],
                    "l1_ratio": [0.5],  # Only for elasticnet penalty
                    "max_iter": self.parameter_vector_space.param_dict.get("log_large_long"),
                    "multi_class": ["auto"],
                    "n_jobs": [None, -1],
                    "penalty": ["elasticnet"],
                    "solver": ["saga"],
                    "tol": self.parameter_vector_space.param_dict.get("log_small"),
                    "verbose": [0],
                    "warm_start": [False],
                },
                {
                    "C": self.parameter_vector_space.param_dict.get("log_small"),
                    "class_weight": [None, "balanced"],
                    "dual": [False],
                    "fit_intercept": [True],
                    "intercept_scaling": [1],
                    "l1_ratio": [None],  # No l1_ratio for l1 and l2 penalties
                    "max_iter": self.parameter_vector_space.param_dict.get("log_large_long"),
                    "multi_class": ["auto"],
                    "n_jobs": [None, -1],
                    "penalty": ["l1"],
                    "solver": ["saga"],
                    "tol": self.parameter_vector_space.param_dict.get("log_small"),
                    "verbose": [0],
                    "warm_start": [False],
                },
                {
                    "C": self.parameter_vector_space.param_dict.get("log_small"),
                    "class_weight": [None, "balanced"],
                    "dual": [False],
                    "fit_intercept": [True],
                    "intercept_scaling": [1],
                    "l1_ratio": [None],  # No l1_ratio for l2 penalty
                    "max_iter": self.parameter_vector_space.param_dict.get("log_large_long"),
                    "multi_class": ["auto"],
                    "n_jobs": [None, -1],
                    "penalty": ["l2"],
                    "solver": ["newton-cg", "lbfgs", "saga"],  # All solvers work for l2
                    "tol": self.parameter_vector_space.param_dict.get("log_small"),
                    "verbose": [0],
                    "warm_start": [False],
                },
            ]
