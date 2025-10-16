from typing import Optional
import pandas as pd
from ml_grid.model_classes.H2OGLMClassifier import H2OGLMClassifier
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Real, Integer
import logging

logging.getLogger('ml_grid').debug("Imported h2o_glm_classifier_class")

class H2O_GLM_class:
    """H2OGLMClassifier with support for Bayesian and grid search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        self.X = X
        self.y = y
        self.algorithm_implementation = H2OGLMClassifier()
        self.method_name = "H2OGLMClassifier"
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        if global_parameters.bayessearch:
            self.parameter_space = [
                {
                    "alpha": Real(0.0, 1.0),  # Mix between L1 and L2
                    "lambda_": Real(1e-8, 1.0, "log-uniform"),  # Regularization strength
                    "seed": Integer(1, 1000),
                }
            ]
        else:
            self.parameter_space = [
                {
                    "alpha": [0.0, 0.2, 0.5, 0.8, 1.0],
                    "lambda_": [1e-5, 1e-4, 1e-3, 1e-2, 0.1],
                    "seed": [1, 42],
                }
            ]