from typing import Optional
import pandas as pd
from ml_grid.model_classes.H2OGLMClassifier import H2OGLMClassifier
from ml_grid.util.global_params import global_parameters
from skopt.space import Real, Integer
import logging

logging.getLogger('ml_grid').debug("Imported h2o_glm_classifier_class")

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
    }
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
    }
}

class H2O_GLM_class:
    """H2OGLMClassifier with support for Bayesian and grid search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: str = 'small',
    ):
        self.X = X
        self.y = y
        # For binary classification, it's crucial to set family='binomial'
        self.algorithm_implementation = H2OGLMClassifier(family='binomial')
        self.method_name = "H2OGLMClassifier"

        if parameter_space_size not in PARAM_SPACE_GRID:
            raise ValueError(f"Invalid parameter_space_size: '{parameter_space_size}'. Must be one of {list(PARAM_SPACE_GRID.keys())}")

        if global_parameters.bayessearch:
            # For Bayesian search, the parameter space is a single dictionary
            self.parameter_space = PARAM_SPACE_BAYES[parameter_space_size]
        else:
            # For Grid search, the parameter space is a list of dictionaries
            self.parameter_space = [PARAM_SPACE_GRID[parameter_space_size]]