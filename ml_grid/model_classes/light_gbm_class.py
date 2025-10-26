from typing import Optional

import pandas as pd
from ml_grid.util import param_space
from ml_grid.model_classes.lightgbm_class import LightGBMClassifier
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Real, Integer
import logging


class LightGBMClassifierWrapper:
    """LightGBMClassifier with support for both Bayesian and non-Bayesian parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        """Initializes the LightGBMClassifierWrapper.

        Args:
            X (Optional[pd.DataFrame]): Feature matrix for training. Defaults to None.
            y (Optional[pd.Series]): Target vector for training. Defaults to None.
            parameter_space_size (Optional[str]): Size of the parameter space for
                optimization. Defaults to None.
        """
        self.X = X
        self.y = y

        global_params = global_parameters

        self.algorithm_implementation = (
            LightGBMClassifier()
        )  # lgb.LGBMClassifier() #custom skelarn wrapper
        self.method_name = "LightGBMClassifier"

        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        if global_params.bayessearch:
            self.parameter_space = {
                "boosting_type": Categorical(("gbdt", "dart", "goss")),
                "num_leaves": Integer(2, 100),
                "learning_rate": Real(1e-5, 1e-1, prior="log-uniform"),
                "n_estimators": Integer(50, 500),
                "objective": Categorical(("binary",)),
                "num_class": Categorical((1,)),
                "metric": Categorical(("logloss",)),
                "feature_fraction": Categorical((0.8, 0.9, 1.0)),
                "early_stopping_rounds": Categorical((None, 10, 20)),
            }
        else:
            self.parameter_space = {
                "boosting_type": ("gbdt", "dart", "goss"),
                "num_leaves": list(
                    self.parameter_vector_space.param_dict.get("log_large_long", [])
                ),
                "learning_rate": list(
                    self.parameter_vector_space.param_dict.get("log_small", [])
                ),
                "n_estimators": list(
                    self.parameter_vector_space.param_dict.get("log_large_long", [])
                ),
                "objective": ("binary",),
                "num_class": (1,),
                "metric": ("logloss",),
                "feature_fraction": (0.8, 0.9, 1.0),
                "early_stopping_rounds": (None, 10, 20),
            }



logging.getLogger('ml_grid').debug("Imported LightGBM classifier wrapper class")
# light_gbm_class LightGBMClassifierWrapper
