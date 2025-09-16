"""Defines the QuadraticDiscriminantAnalysis model class."""

from typing import Optional

import pandas as pd
from ml_grid.util import param_space
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical

print("Imported QuadraticDiscriminantAnalysis class")


class quadratic_discriminant_analysis_class:
    """QuadraticDiscriminantAnalysis with support for hyperparameter tuning."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        """Initializes the quadratic_discriminant_analysis_class.

        Args:
            X (Optional[pd.DataFrame]): Feature matrix for training.
                Defaults to None.
            y (Optional[pd.Series]): Target vector for training.
                Defaults to None.
            parameter_space_size (Optional[str]): Size of the parameter space for
                optimization. Defaults to None.
        """
        global_params = global_parameters
        self.X = X
        self.y = y

        self.algorithm_implementation = QuadraticDiscriminantAnalysis()
        self.method_name = "QuadraticDiscriminantAnalysis"

        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        if global_params.bayessearch:
            self.parameter_space = {
                "priors": Categorical([None]),  # Categorical: single option, None
                "reg_param": self.parameter_vector_space.param_dict.get("log_small"),  # Log-uniform between 1e-5 and 1e-2
                "store_covariance": Categorical([False]),  # Categorical: single option, False
                "tol": self.parameter_vector_space.param_dict.get("log_small"),  # Log-uniform between 1e-5 and 1e-2
            }

        else:
            self.parameter_space = {
                "priors": [None],
                "reg_param": self.parameter_vector_space.param_dict.get("log_small"),
                "store_covariance": [False],
                "tol": self.parameter_vector_space.param_dict.get("log_small"),
            }

        return None
