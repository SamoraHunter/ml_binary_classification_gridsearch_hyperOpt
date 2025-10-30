"""H2O Stacked Ensemble Classifier.

This module contains the H2O_StackedEnsemble_class, which is a configuration
class for the H2OStackedEnsembleClassifier. It provides parameter spaces for
grid search and Bayesian optimization.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
from ml_grid.model_classes.H2OStackedEnsembleClassifier import H2OStackedEnsembleClassifier
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Integer
import logging
# --- FIX: Import our custom wrapper classes to use as base models ---
from .h2o_gbm_classifier_class import H2O_GBM_class
from .h2o_drf_classifier_class import H2ODRFClass as H2O_DRF_class
from .h2o_glm_classifier_class import H2O_GLM_class


logging.getLogger('ml_grid').debug("Imported h2o_stackedensemble_classifier_class")

# Define parameter spaces for the metalearner outside the class.
PARAM_SPACE_GRID = {
    "xsmall": {
        "metalearner_algorithm": ["AUTO"],
        "seed": [1],
    },
    "small": {
        "metalearner_algorithm": ["AUTO", "glm"],
        "seed": [1, 42],
    },
    "medium": {
        "metalearner_algorithm": ["AUTO", "glm", "drf"],
        "seed": [1, 42, 123],
    }
}

PARAM_SPACE_BAYES: Dict[str, Dict[str, Union[Categorical, Integer]]] = {
    "xsmall": {
        "metalearner_algorithm": Categorical(["AUTO", "glm"]),
        "seed": Integer(1, 100),
    },
    "small": {
        "metalearner_algorithm": Categorical(["AUTO", "glm", "drf", "gbm"]),
        "seed": Integer(1, 1000),
    },
    "medium": {
        "metalearner_algorithm": Categorical(["AUTO", "glm", "drf", "gbm"]),
        "seed": Integer(1, 2000),
    }
}


class H2O_StackedEnsemble_class:
    """A configuration class for the H2OStackedEnsembleClassifier.

    This class provides parameter spaces for grid search and Bayesian
    optimization for the H2OStackedEnsembleClassifier.

    Attributes:
        X (Optional[pd.DataFrame]): The input features.
        y (Optional[pd.Series]): The target variable.
        algorithm_implementation (H2OStackedEnsembleClassifier): An instance of the
            classifier.
        method_name (str): The name of the method.
        parameter_space (Union[List[Dict[str, Any]], Dict[str, Any]]): The
            hyperparameter search space.
    """

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: str = 'small',
    ) -> None:
        """Initializes the H2O_StackedEnsemble_class.

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

        # --- FIX: Instantiate our custom classes as base models ---
        # This ensures the parameter_space_size is correctly passed down.
        # The H2OStackedEnsembleClassifier wrapper expects a list of *wrapper* instances.
        base_models = [
            H2O_GBM_class(parameter_space_size=parameter_space_size),
            H2O_DRF_class(parameter_space_size=parameter_space_size),
            H2O_GLM_class(parameter_space_size=parameter_space_size),
        ]

        # A stacked ensemble requires a list of base models to train.
        self.algorithm_implementation: H2OStackedEnsembleClassifier = H2OStackedEnsembleClassifier(base_models=base_models)
        self.method_name: str = "H2OStackedEnsembleClassifier"

        if parameter_space_size not in PARAM_SPACE_GRID:
            raise ValueError(f"Invalid parameter_space_size: '{parameter_space_size}'. Must be one of {list(PARAM_SPACE_GRID.keys())}")

        if global_parameters.bayessearch:
            # For Bayesian search, the parameter space is a single dictionary
            self.parameter_space: Dict[str, Any] = PARAM_SPACE_BAYES[parameter_space_size]
        else:
            # For Grid search, the parameter space is a list of dictionaries
            self.parameter_space: List[Dict[str, Any]] = [PARAM_SPACE_GRID[parameter_space_size]]