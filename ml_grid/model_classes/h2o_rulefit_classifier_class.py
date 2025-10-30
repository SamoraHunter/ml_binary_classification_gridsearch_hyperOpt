"""H2O RuleFit Classifier.

This module contains the H2ORuleFitClass, which is a configuration
the H2ORuleFitClassifier. It provides parameter spaces for grid search and
Bayesian optimization.
"""

import logging
from typing import Dict, List, Optional, Union

import pandas as pd # type: ignore
from skopt.space import Categorical, Integer

from ml_grid.model_classes.H2ORuleFitClassifier import H2ORuleFitClassifier
from ml_grid.util.global_params import global_parameters

logging.getLogger("ml_grid").debug("Imported h2o_rulefit_classifier_class")


# Define parameter spaces outside the class for better organization and reusability.
PARAM_SPACE_GRID: Dict[str, Dict[str, List[Union[int, str]]]] = {
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
    },
}

PARAM_SPACE_BAYES: Dict[str, Dict[str, Union[Integer, Categorical]]] = {
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
    },
}


class H2ORuleFitClass:
    """A wrapper for the H2ORuleFitClassifier.

    This class provides a consistent interface for using the H2ORuleFitClassifier
    within the ml_grid framework, including support for both grid search and
    Bayesian optimization of hyperparameters.

    Attributes:
        X (Optional[pd.DataFrame]): The input features.
        y (Optional[pd.Series]): The target variable.
        algorithm_implementation (H2ORuleFitClassifier): An instance of the classifier.
        method_name (str): The name of the method.
        parameter_space (Union[List[Dict[str, Any]], Dict[str, Any]]): The hyperparameter search space.
    """

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: str = "small",
    ) -> None:
        """Initializes the H2ORuleFitClass.

        Args:
            X: The input features (optional).
            y: The target variable (optional).
            parameter_space_size: The size of the hyperparameter space to use
                ('xsmall', 'small', or 'medium').
        """
        self.X: Optional[pd.DataFrame] = X
        self.y: Optional[pd.Series] = y
        self.algorithm_implementation: H2ORuleFitClassifier = H2ORuleFitClassifier()
        self.method_name: str = "H2ORuleFitClassifier"
        self.parameter_space: Union[List[Dict[str, Any]], Dict[str, Any]]

        if parameter_space_size not in PARAM_SPACE_GRID:
            raise ValueError(
                f"Invalid parameter_space_size: '{parameter_space_size}'. "
                f"Must be one of {list(PARAM_SPACE_GRID.keys())}"
            )
        if global_parameters.bayessearch:
            self.parameter_space = PARAM_SPACE_BAYES[
                parameter_space_size
            ]
        else:
            self.parameter_space = [
                PARAM_SPACE_GRID[parameter_space_size]
            ]
