from typing import Optional
import pandas as pd
from ml_grid.model_classes.H2OStackedEnsembleClassifier import H2OStackedEnsembleClassifier
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Integer
import logging

logging.getLogger('ml_grid').debug("Imported h2o_stackedensemble_classifier_class")

class H2O_StackedEnsemble_class:
    """H2OStackedEnsembleClassifier with support for Bayesian and grid search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        self.X = X
        self.y = y
        self.algorithm_implementation = H2OStackedEnsembleClassifier()
        self.method_name = "H2OStackedEnsembleClassifier"
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        # The parameter space is minimal as the base models are fixed.
        # We can expose metalearner params here in the future.
        if global_parameters.bayessearch:
            self.parameter_space = [
                {
                    "metalearner_algorithm": Categorical(["AUTO", "glm"]),
                    "seed": Integer(1, 1000),
                }
            ]
        else:
            self.parameter_space = [
                {
                    "metalearner_algorithm": ["AUTO", "glm"],
                    "seed": [1, 42],
                }
            ]