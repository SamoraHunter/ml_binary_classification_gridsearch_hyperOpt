"""Configuration class for the H2O Deep Learning Classifier."""

from typing import Any, Dict, List, Optional, Union

import logging
import pandas as pd
from ml_grid.model_classes.H2ODeepLearningClassifier import H2ODeepLearningClassifier
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Integer, Real

logger = logging.getLogger(__name__)
logger.debug("Imported h2o_deeplearning_classifier_class")

# Define parameter spaces outside the class for better organization and reusability.
# This also makes it easier to manage different parameter space sizes.
PARAM_SPACE_GRID = {
    "xsmall": {
        "epochs": [5],
        "hidden_config": ["small"],
        "activation": ["Rectifier"],
        "l1": [0],
        "l2": [0],
        "seed": [1],
    },
    "small": {
        "epochs": [5, 10],
        "hidden_config": ["small", "medium"],
        "activation": ["Rectifier", "Tanh"],
        "l1": [0, 1e-4],
        "l2": [0, 1e-4],
        "seed": [1, 42],
    },
    "medium": {
        "epochs": [10, 50, 100],
        "hidden_config": ["small", "medium", "large"],
        "activation": ["Rectifier", "Tanh", "Maxout"],
        "l1": [0, 1e-4, 1e-3],
        "l2": [0, 1e-4, 1e-3],
        "seed": [1, 42, 123],
    },
}

PARAM_SPACE_BAYES = {
    "xsmall": {
        "epochs": Integer(5, 10),
        "hidden_config": Categorical(["small"]),
        "activation": Categorical(["Rectifier"]),
        "l1": Real(1e-5, 1e-4, "log-uniform"),
        "l2": Real(1e-5, 1e-4, "log-uniform"),
        "seed": Integer(1, 100),
    },
    "small": {
        "epochs": Integer(5, 20),
        "hidden_config": Categorical(["small", "medium", "large"]),
        "activation": Categorical(["Rectifier", "Tanh"]),
        "l1": Real(1e-5, 1e-3, "log-uniform"),
        "l2": Real(1e-5, 1e-3, "log-uniform"),
        "seed": Integer(1, 1000),
    },
    "medium": {
        "epochs": Integer(10, 200),
        "hidden_config": Categorical(["small", "medium", "large"]),
        "activation": Categorical(["Rectifier", "Tanh", "Maxout"]),
        "l1": Real(1e-6, 1e-2, "log-uniform"),
        "l2": Real(1e-6, 1e-2, "log-uniform"),
        "seed": Integer(1, 2000),
    },
}


class H2ODeepLearningConfig:
    """Configuration class for H2ODeepLearningClassifier.

    Provides parameter spaces for grid search and Bayesian optimization.

    Attributes:
        X (Optional[pd.DataFrame]): The input features.
        y (Optional[pd.Series]): The target variable.
        algorithm_implementation (H2ODeepLearningClassifier): An instance of the
            classifier.
        method_name (str): The name of the method, "H2ODeepLearningClassifier".
        parameter_space (Union[List[Dict[str, Any]], Dict[str, Any]]): The
            hyperparameter search space.
    """

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: str = "small",
    ) -> None:
        """Initializes the H2ODeepLearningConfig.

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
        self.algorithm_implementation = H2ODeepLearningClassifier()
        self.method_name: str = "H2ODeepLearningClassifier"

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

        # Note: The 'hidden_config' string is decoded into a layer list
        # inside the H2ODeepLearningClassifier wrapper during the fit method.
