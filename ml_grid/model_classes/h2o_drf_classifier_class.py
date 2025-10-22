from typing import Optional
import pandas as pd
from ml_grid.model_classes.H2ODRFClassifier import H2ODRFClassifier
from ml_grid.util.global_params import global_parameters
from skopt.space import Real, Integer
import logging

logging.getLogger('ml_grid').debug("Imported h2o_drf_classifier_class")

# Define parameter spaces outside the class for better organization and reusability.
PARAM_SPACE_GRID = {
    "xsmall": {
        "ntrees": [50],
        "max_depth": [10],
        "sample_rate": [0.8],
        "col_sample_rate_per_tree": [0.8],
        "seed": [1],
    },
    "small": {
        "ntrees": [50, 100, 200],
        "max_depth": [10, 20, 30],
        "sample_rate": [0.632, 0.8, 1.0],  # 0.632 is the default
        "col_sample_rate_per_tree": [0.8, 1.0],
        "seed": [1, 42],
    },
    "medium": {
        "ntrees": [50, 100, 200, 300],
        "max_depth": [10, 20, 30],
        "min_rows": [1, 5, 10],
        "nbins": [20, 50, 100],
        "sample_rate": [0.6, 0.8, 1.0],
        "col_sample_rate_per_tree": [0.6, 0.8, 1.0],
        "seed": [1, 42, 123],
    }
}

PARAM_SPACE_BAYES = {
    "xsmall": {
        "ntrees": Integer(50, 100),
        "max_depth": Integer(5, 10),
        "sample_rate": Real(0.7, 0.9),
        "col_sample_rate_per_tree": Real(0.7, 0.9),
        "seed": Integer(1, 100),
    },
    "small": {
        "ntrees": Integer(50, 500),
        "max_depth": Integer(5, 30),
        "sample_rate": Real(0.5, 1.0),
        "col_sample_rate_per_tree": Real(0.5, 1.0),
        "seed": Integer(1, 1000),
    },
    "medium": {
        "ntrees": Integer(50, 1000),
        "max_depth": Integer(10, 50),
        "min_rows": Integer(1, 20),
        "nbins": Integer(20, 200),
        "sample_rate": Real(0.5, 1.0),
        "col_sample_rate_per_tree": Real(0.5, 1.0),
        "seed": Integer(1, 2000),
    }
}

class H2O_DRF_class:
    """H2ODRFClassifier with support for Bayesian and grid search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: str = 'small',
    ):
        self.X = X
        self.y = y
        self.algorithm_implementation = H2ODRFClassifier()
        self.method_name = "H2ODRFClassifier"

        if parameter_space_size not in PARAM_SPACE_GRID:
            raise ValueError(f"Invalid parameter_space_size: '{parameter_space_size}'. Must be one of {list(PARAM_SPACE_GRID.keys())}")

        if global_parameters.bayessearch:
            # For Bayesian search, the parameter space is a single dictionary
            self.parameter_space = PARAM_SPACE_BAYES[parameter_space_size]
        else:
            # For Grid search, the parameter space is a list of dictionaries
            self.parameter_space = [PARAM_SPACE_GRID[parameter_space_size]]