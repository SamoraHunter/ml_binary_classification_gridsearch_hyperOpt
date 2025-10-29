"""Configuration class for the H2O Generalized Linear Model (GLM) Classifier."""

from typing import Any, Dict, List, Optional, Union

import logging
import pandas as pd
from ml_grid.model_classes.H2OGLMClassifier import H2OGLMClassifier
from ml_grid.util.global_params import global_parameters
from skopt.space import Integer, Real

logger = logging.getLogger(__name__)
logger.debug("Imported h2o_glm_classifier_class")

# Define parameter spaces outside the class for better organization and reusability.
PARAM_SPACE_GRID = {
    "xsmall": {
        "alpha": [0.5],
        "lambda_": [1e-4],
        "seed": [1],
    },
    "small": {
        "alpha": [0.0, 0.2, 0.5, 0.8, 1.0],
        "lambda_": [1e-5, 1e-4, 1e-3, 1e-2, 0.1],
        "seed": [1, 42],
    },
    "medium": {
        "alpha": [0.0, 0.25, 0.5, 0.75, 1.0],
        "lambda_": [1e-5, 1e-4, 1e-3, 1e-2],
        "seed": [1, 42, 123],
    },
}

PARAM_SPACE_BAYES = {
    "xsmall": {
        "alpha": Real(0.4, 0.6),
        "lambda_": Real(1e-5, 1e-3, "log-uniform"),
        "seed": Integer(1, 100),
    },
    "small": {
        "alpha": Real(0.0, 1.0),  # Mix between L1 and L2
        "lambda_": Real(1e-8, 1.0, "log-uniform"),  # Regularization strength
        "seed": Integer(1, 1000),
    },
    "medium": {
        "alpha": Real(0.0, 1.0),
        "lambda_": Real(1e-8, 1.0, "log-uniform"),
        "seed": Integer(1, 2000),
    },
}


class H2OGLMConfig:
    """Configuration class for H2OGLMClassifier.

    This class provides parameter spaces for grid search and Bayesian
    optimization. The H2OGLMClassifier is instantiated with `family='binomial'`
    for binary classification tasks.

    Attributes:
        X (Optional[pd.DataFrame]): The input features.
        y (Optional[pd.Series]): The target variable.
        algorithm_implementation (H2OGLMClassifier): An instance of the
            classifier.
        method_name (str): The name of the method, "H2OGLMClassifier".
        parameter_space (Union[List[Dict[str, Any]], Dict[str, Any]]): The
            hyperparameter search space.
    """

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: str = "small",
    ) -> None:
        """Initializes the H2OGLMConfig.

        Args:
            X (Optional[pd.DataFrame]): The input features.
            y (Optional[pd.Series]): The target variable.
            parameter_space_size (str): The size of the parameter space to use
                ('xsmall', 'small', 'medium'). Defaults to 'small'.

        Raises:
            ValueError: If `parameter_space_size` is not a valid key.
        """
        self.X: Optional[pd.DataFrame] = X
        self.y: Optional[pd.Series] = y
        # For binary classification, it's crucial to set family='binomial'
        self.algorithm_implementation = H2OGLMClassifier(family="binomial")
        self.method_name: str = "H2OGLMClassifier"

        if parameter_space_size not in PARAM_SPACE_GRID:
            raise ValueError(
                f"Invalid parameter_space_size: '{parameter_space_size}'. "
                f"Must be one of {list(PARAM_SPACE_GRID.keys())}"
            )

        if global_parameters.bayessearch:
            self.parameter_space: Dict[str, Any] = PARAM_SPACE_BAYES[
                parameter_space_size
            ]
        else:
            self.parameter_space: List[Dict[str, Any]] = [
                PARAM_SPACE_GRID[parameter_space_size]
            ]
