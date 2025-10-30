"""Defines the QuadraticDiscriminantAnalysis model class."""

import logging
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from skopt.space import Categorical, Real

from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters

logging.getLogger("ml_grid").debug("Imported QuadraticDiscriminantAnalysis class")


class QuadraticDiscriminantAnalysisClass:
    """QuadraticDiscriminantAnalysis with support for hyperparameter tuning."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        """Initializes the QuadraticDiscriminantAnalysisClass.

        Args:
            X (Optional[pd.DataFrame]): Feature matrix for training.
                Defaults to None.
            y (Optional[pd.Series]): Target vector for training.
                Defaults to None.
            parameter_space_size (Optional[str]): Size of the parameter space for
                optimization. Defaults to None.

        Raises:
            ValueError: If `parameter_space_size` is not a valid key (though current
                implementation does not explicitly raise this).
        """
        global_params = global_parameters
        self.X: Optional[pd.DataFrame] = X
        self.y: Optional[pd.Series] = y

        self.algorithm_implementation: QuadraticDiscriminantAnalysis = (
            QuadraticDiscriminantAnalysis()
        )
        self.method_name: str = "QuadraticDiscriminantAnalysis"

        self.parameter_vector_space: param_space.ParamSpace = param_space.ParamSpace(
            parameter_space_size
        )
        self.parameter_space: Dict[str, Any]

        if global_params.bayessearch:
            self.parameter_space = {
                "priors": Categorical([None]),  # Categorical: single option, None
                "reg_param": Real(1e-5, 1e-2, prior="log-uniform"),
                "store_covariance": Categorical(
                    [False]
                ),  # Categorical: single option, False
                "tol": Real(1e-5, 1e-2, prior="log-uniform"),
            }

        else:
            self.parameter_space = {
                "priors": [None],
                "reg_param": self.parameter_vector_space.param_dict.get("log_small"),
                "store_covariance": [False],
                "tol": self.parameter_vector_space.param_dict.get("log_small"),
            }
