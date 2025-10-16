from typing import Optional
import pandas as pd
from ml_grid.model_classes.H2ODRFClassifier import H2ODRFClassifier
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Real, Integer
import logging

logging.getLogger('ml_grid').debug("Imported h2o_drf_classifier_class")

class H2O_DRF_class:
    """H2ODRFClassifier with support for Bayesian and grid search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        self.X = X
        self.y = y
        self.algorithm_implementation = H2ODRFClassifier()
        self.method_name = "H2ODRFClassifier"
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        if global_parameters.bayessearch:
            self.parameter_space = [
                {
                    "ntrees": Integer(50, 500),
                    "max_depth": Integer(5, 30),
                    "sample_rate": Real(0.5, 1.0),
                    "col_sample_rate_per_tree": Real(0.5, 1.0),
                    "seed": Integer(1, 1000),
                }
            ]
        else:
            self.parameter_space = [
                {
                    "ntrees": [50, 100, 200],
                    "max_depth": [10, 20, 30],
                    "sample_rate": [0.632, 0.8, 1.0], # 0.632 is the default
                    "col_sample_rate_per_tree": [0.8, 1.0],
                    "seed": [1, 42],
                }
            ]