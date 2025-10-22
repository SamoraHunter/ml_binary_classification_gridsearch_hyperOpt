from typing import Optional
import pandas as pd
from ml_grid.model_classes.H2ODeepLearningClassifier import H2ODeepLearningClassifier
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Real, Integer
import logging

logging.getLogger('ml_grid').debug("Imported h2o_deeplearning_classifier_class")

# Define parameter spaces outside the class for better organization and reusability.
# This also makes it easier to manage different parameter space sizes (e.g., 'small', 'medium', 'large').
PARAM_SPACE_GRID = {
    "xsmall": {
        "epochs": [5],
        "hidden_config": ['small'],
        "activation": ['Rectifier'],
        "l1": [0],
        "l2": [0],
        "seed": [1],
    },
    "small": {
        "epochs": [5, 10],
        "hidden_config": ['small', 'medium'],
        "activation": ['Rectifier', 'Tanh'],
        "l1": [0, 1e-4],
        "l2": [0, 1e-4],
        "seed": [1, 42],
    }
}

PARAM_SPACE_BAYES = {
    "xsmall": {
        "epochs": Integer(5, 10),
        "hidden_config": Categorical(['small']),
        "activation": Categorical(['Rectifier']),
        "l1": Real(1e-5, 1e-4, "log-uniform"),
        "l2": Real(1e-5, 1e-4, "log-uniform"),
        "seed": Integer(1, 100),
    },
    "small": {
        "epochs": Integer(5, 20),
        "hidden_config": Categorical(['small', 'medium', 'large']),
        "activation": Categorical(['Rectifier', 'Tanh']),
        "l1": Real(1e-5, 1e-3, "log-uniform"),
        "l2": Real(1e-5, 1e-3, "log-uniform"),
        "seed": Integer(1, 1000),
    }
}

class H2O_DeepLearning_class:
    """H2ODeepLearningClassifier with support for Bayesian and grid search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: str = 'small',
    ):
        self.X = X
        self.y = y
        self.algorithm_implementation = H2ODeepLearningClassifier()
        self.method_name = "H2ODeepLearningClassifier"

        # Ensure parameter_space_size is a valid key
        if parameter_space_size not in PARAM_SPACE_GRID:
            raise ValueError(f"Invalid parameter_space_size: '{parameter_space_size}'. Must be one of {list(PARAM_SPACE_GRID.keys())}")

        if global_parameters.bayessearch:
            # For Bayesian search, the parameter space is a single dictionary
            self.parameter_space = PARAM_SPACE_BAYES[parameter_space_size]
        else:
            # For Grid search, the parameter space is a list of dictionaries
            self.parameter_space = [PARAM_SPACE_GRID[parameter_space_size]]

        # Note: The 'hidden_config' string is decoded into a layer list
        # inside the H2ODeepLearningClassifier wrapper during the fit method.