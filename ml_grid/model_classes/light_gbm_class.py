from typing import Optional

import pandas as pd
import lightgbm as lgb
from ml_grid.util import param_space

# from sklearn.base import BaseEstimator, ClassifierMixin

from ml_grid.model_classes.lightgbm_class import LightGBMClassifier
from ml_grid.util.global_params import global_parameters


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

        from skopt.space import Categorical

        if global_params.bayessearch:
            self.parameter_space = {
                "boosting_type": Categorical(("gbdt", "dart", "goss")),
                "num_leaves": Categorical(
            tuple(
                value for value in range(
                    int(self.parameter_vector_space.param_dict["log_large_long"].low),
                    int(self.parameter_vector_space.param_dict["log_large_long"].high) + 1
                ) if value > 1
            )
        ),
                "learning_rate": self.parameter_vector_space.param_dict.get("log_small"),
                "n_estimators": self.parameter_vector_space.param_dict.get("log_large_long"),
                "objective": Categorical(("binary",)),
                "num_class": Categorical((1,)),
                "metric": Categorical(("logloss",)),
                "feature_fraction": Categorical((0.8, 0.9, 1.0)),
                "early_stopping_rounds": Categorical((None, 10, 20)),
                # Uncomment and adapt these lines if needed:
                # "verbosity": Categorical((-1,)),
                # "error_score": Categorical(("raise",)),
                # "verbose_eval": Categorical((-1,)),
                # "verbose": Categorical((-1,)),
            }
        else:
            self.parameter_space = {
                "boosting_type": ("gbdt", "dart", "goss"),
                "num_leaves": tuple(self.parameter_vector_space.param_dict.get("log_large_long", [])),
                "learning_rate": tuple(self.parameter_vector_space.param_dict.get("log_small", [])),
                "n_estimators": tuple(self.parameter_vector_space.param_dict.get("log_large_long", [])),
                "objective": ("binary",),
                "num_class": (1,),
                "metric": ("logloss",),
                "feature_fraction": (0.8, 0.9, 1.0),
                "early_stopping_rounds": (None, 10, 20),
                # Uncomment and adapt these lines if needed:
                # "verbosity": (-1,),
                # "error_score": ("raise",),
                # "verbose_eval": (-1,),
                # "verbose": (-1,),
            }




print("Imported LightGBM classifier wrapper class")
# light_gbm_class LightGBMClassifierWrapper
