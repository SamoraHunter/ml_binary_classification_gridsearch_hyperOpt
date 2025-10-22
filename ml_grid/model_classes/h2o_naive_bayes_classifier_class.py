from typing import Optional
import pandas as pd
from ml_grid.model_classes.H2ONaiveBayesClassifier import H2ONaiveBayesClassifier
from ml_grid.util.global_params import global_parameters
from skopt.space import Real, Integer
import logging

logging.getLogger('ml_grid').debug("Imported h2o_naive_bayes_classifier_class")

# Define parameter spaces outside the class for better organization and reusability.
PARAM_SPACE_GRID = {
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
    }
}

PARAM_SPACE_BAYES = {
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
    }
}

class H2O_NaiveBayes_class:
    """H2ONaiveBayesClassifier with support for Bayesian and grid search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: str = 'small',
    ):
        self.X = X
        self.y = y
        self.algorithm_implementation = H2ONaiveBayesClassifier()
        self.method_name = "H2ONaiveBayesClassifier"

        if parameter_space_size not in PARAM_SPACE_GRID:
            raise ValueError(f"Invalid parameter_space_size: '{parameter_space_size}'. Must be one of {list(PARAM_SPACE_GRID.keys())}")

        if global_parameters.bayessearch:
            # For Bayesian search, the parameter space is a single dictionary
            self.parameter_space = PARAM_SPACE_BAYES[parameter_space_size]
        else:
            # For Grid search, the parameter space is a list of dictionaries
            self.parameter_space = [PARAM_SPACE_GRID[parameter_space_size]]