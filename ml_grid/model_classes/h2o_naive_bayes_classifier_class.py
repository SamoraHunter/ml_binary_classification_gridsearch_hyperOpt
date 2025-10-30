"""H2O Naive Bayes Classifier.

This module contains the H2O_NaiveBayes_class, which is a configuration
class for the H2ONaiveBayesClassifier. It provides parameter spaces for
grid search and Bayesian optimization.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from skopt.space import Integer, Real

from ml_grid.model_classes.H2ONaiveBayesClassifier import H2ONaiveBayesClassifier
from ml_grid.util.global_params import global_parameters

logger = logging.getLogger(__name__)
logger.debug("Imported h2o_naive_bayes_classifier_class")

# Define parameter spaces outside the class for better organization and reusability.
PARAM_SPACE_GRID: Dict[str, Dict[str, List[Union[int, float]]]] = {
    "xsmall": {
        "laplace": [1],
        "min_sdev": [0.001],
        "eps_sdev": [0.001],
        "seed": [1],
    },
    "small": {
        "laplace": [0, 1, 5],
        "min_sdev": [0.001, 0.1],
        "eps_sdev": [0, 0.001, 0.1],
        "seed": [1, 42],
    },
    "medium": {
        "laplace": [0, 0.25, 0.5, 0.75, 1.0],
        "min_sdev": [0.001, 0.01, 0.1],
        "eps_sdev": [0, 0.01, 0.1],
        "seed": [1, 42, 123],
    },
}

PARAM_SPACE_BAYES: Dict[str, Dict[str, Union[Real, Integer]]] = {
    "xsmall": {
        "laplace": Real(0.5, 2.0, "log-uniform"),
        "min_sdev": Real(0.001, 0.1, "log-uniform"),
        "eps_sdev": Real(1e-4, 0.1, "log-uniform"),
        "seed": Integer(1, 100),
    },
    "small": {
        "laplace": Real(1e-5, 10.0, "log-uniform"),
        "min_sdev": Real(0.001, 1.0, "log-uniform"),
        "eps_sdev": Real(1e-5, 1.0, "log-uniform"),
        "seed": Integer(1, 1000),
    },
    "medium": {
        "laplace": Real(0, 10),
        "min_sdev": Real(1e-5, 1.0, "log-uniform"),
        "eps_sdev": Real(1e-5, 1.0, "log-uniform"),
        "seed": Integer(1, 2000),
    },
}

ParameterSpace = Union[List[Dict[str, Any]], Dict[str, Any]]


class H2O_NaiveBayes_class:
    """A configuration class for the H2ONaiveBayesClassifier.

    This class provides parameter spaces for grid search and Bayesian
    optimization for the H2ONaiveBayesClassifier.

    Attributes:
        X (Optional[pd.DataFrame]): The input features.
        y (Optional[pd.Series]): The target variable.
        algorithm_implementation (H2ONaiveBayesClassifier): An instance of the
            classifier.
        method_name (str): The name of the method.
        parameter_space (ParameterSpace): The
            hyperparameter search space.
    """

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: str = "small",
    ) -> None:
        """Initializes the H2O_NaiveBayes_class.

        Args:
            X (Optional[pd.DataFrame]): The input features. Defaults to None.
            y (Optional[pd.Series]): The target variable. Defaults to None.
            parameter_space_size (str): The size of the parameter space to use
                ('xsmall', 'small', 'medium'). Defaults to 'small'.

        Raises:
            ValueError: If `parameter_space_size` is not a valid key.
        """
        self.X: Optional[pd.DataFrame] = X
        self.y: Optional[pd.Series] = y
        self.algorithm_implementation = H2ONaiveBayesClassifier()
        self.method_name: str = "H2ONaiveBayesClassifier"

        if parameter_space_size not in PARAM_SPACE_GRID:
            raise ValueError(
                f"Invalid parameter_space_size: '{parameter_space_size}'. "
                f"Must be one of {list(PARAM_SPACE_GRID.keys())}"
            )

        if global_parameters.bayessearch:
            self.parameter_space: ParameterSpace = PARAM_SPACE_BAYES[
                parameter_space_size
            ]
        else:
            self.parameter_space: ParameterSpace = [
                PARAM_SPACE_GRID[parameter_space_size]
            ]
