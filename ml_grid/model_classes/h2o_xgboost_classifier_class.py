from typing import Optional
import pandas as pd
from ml_grid.model_classes.H2OXGBoostClassifier import H2OXGBoostClassifier
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Real, Integer
import logging

logging.getLogger('ml_grid').debug("Imported h2o_xgboost_classifier_class")

class H2O_XGBoost_class:
    """H2OXGBoostClassifier with support for Bayesian and grid search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        self.X = X
        self.y = y
        self.algorithm_implementation = H2OXGBoostClassifier()
        self.method_name = "H2OXGBoostClassifier"
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        if global_parameters.bayessearch:
            self.parameter_space = [
                {
                    "ntrees": Integer(50, 500),
                    "max_depth": Integer(3, 10),
                    "learn_rate": Real(0.01, 0.3, "log-uniform"),
                    "sample_rate": Real(0.5, 1.0),
                    "col_sample_rate_bytree": Real(0.5, 1.0),
                    "seed": Integer(1, 1000),
                }
            ]
        else:
            self.parameter_space = [
                {
                    "ntrees": [50, 100, 200],
                    "max_depth": [3, 5, 8],
                    "learn_rate": [0.01, 0.1, 0.2],
                    "sample_rate": [0.8, 1.0],
                    "col_sample_rate_bytree": [0.8, 1.0],
                    "seed": [1, 42],
                }
            ]