from typing import Optional
import pandas as pd
from ml_grid.model_classes.H2OStackedEnsembleClassifier import H2OStackedEnsembleClassifier
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Integer
import logging
# --- FIX: Import our custom wrapper classes to use as base models ---
from .h2o_gbm_classifier_class import H2O_GBM_class
from .h2o_drf_classifier_class import H2O_DRF_class
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
    }
}

PARAM_SPACE_BAYES = {
    "xsmall": {
        "metalearner_algorithm": Categorical(["AUTO", "glm"]),
        "seed": Integer(1, 100),
    },
    "small": {
        "metalearner_algorithm": Categorical(["AUTO", "glm", "drf", "gbm"]),
        "seed": Integer(1, 1000),
    }
}

class H2O_StackedEnsemble_class:
    """H2OStackedEnsembleClassifier with support for Bayesian and grid search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: str = 'small',
    ):
        self.X = X
        self.y = y

        # --- FIX: Instantiate our custom classes as base models ---
        # This ensures the parameter_space_size is correctly passed down.
        # We access the underlying H2O estimator via the .algorithm_implementation attribute.
        base_models = [
            H2O_GBM_class(parameter_space_size=parameter_space_size).algorithm_implementation,
            H2O_DRF_class(parameter_space_size=parameter_space_size).algorithm_implementation,
            H2O_GLM_class(parameter_space_size=parameter_space_size).algorithm_implementation,
        ]

        # A stacked ensemble requires a list of base models to train.
        self.algorithm_implementation = H2OStackedEnsembleClassifier(base_models=base_models)
        self.method_name = "H2OStackedEnsembleClassifier"

        if parameter_space_size not in PARAM_SPACE_GRID:
            raise ValueError(f"Invalid parameter_space_size: '{parameter_space_size}'. Must be one of {list(PARAM_SPACE_GRID.keys())}")

        if global_parameters.bayessearch:
            # For Bayesian search, the parameter space is a single dictionary
            self.parameter_space = PARAM_SPACE_BAYES[parameter_space_size]
        else:
            # For Grid search, the parameter space is a list of dictionaries
            self.parameter_space = [PARAM_SPACE_GRID[parameter_space_size]]