from typing import Optional
import pandas as pd
from ml_grid.model_classes.H2ORuleFitClassifier import H2ORuleFitClassifier
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Integer
import logging

logging.getLogger('ml_grid').debug("Imported h2o_rulefit_classifier_class")

# Define parameter spaces outside the class for better organization and reusability.
PARAM_SPACE_GRID = {
    "xsmall": {
        "min_rule_length": [1],
        "max_rule_length": [5],
        "model_type": ["rules_and_linear"],
        "rule_generation_ntrees": [20],
        "seed": [1],
    },
    "small": {
        "min_rule_length": [1, 2],
        "max_rule_length": [3, 5, 10],
        "model_type": ["rules_and_linear", "rules"],
        "seed": [1, 42],
    },
    "medium": {
        "min_rule_length": [1, 2, 3],
        "max_rule_length": [3, 5, 10, 15],
        "model_type": ["rules_and_linear", "rules", "linear"],
        "rule_generation_ntrees": [50, 100],
        "seed": [1, 42, 123],
    }
}

PARAM_SPACE_BAYES = {
    "xsmall": {
        "min_rule_length": Integer(1, 2),
        "max_rule_length": Integer(3, 5),
        "model_type": Categorical(["rules_and_linear"]),
        "rule_generation_ntrees": Integer(20, 50),
        "seed": Integer(1, 100),
    },
    "small": {
        "min_rule_length": Integer(1, 5),
        "max_rule_length": Integer(2, 10),
        "model_type": Categorical(["rules_and_linear", "rules", "linear"]),
        "seed": Integer(1, 1000),
    },
    "medium": {
        "min_rule_length": Integer(1, 5),
        "max_rule_length": Integer(3, 20),
        "model_type": Categorical(["rules_and_linear", "rules", "linear"]),
        "rule_generation_ntrees": Integer(20, 200),
        "seed": Integer(1, 2000),
    }
}

class H2O_RuleFit_class:
    """H2ORuleFitClassifier with support for Bayesian and grid search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: str = 'small',
    ):
        self.X = X
        self.y = y
        self.algorithm_implementation = H2ORuleFitClassifier()
        self.method_name = "H2ORuleFitClassifier"

        if parameter_space_size not in PARAM_SPACE_GRID:
            raise ValueError(f"Invalid parameter_space_size: '{parameter_space_size}'. Must be one of {list(PARAM_SPACE_GRID.keys())}")

        if global_parameters.bayessearch:
            self.parameter_space = PARAM_SPACE_BAYES[parameter_space_size]
        else:
            self.parameter_space = [PARAM_SPACE_GRID[parameter_space_size]]