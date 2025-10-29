"""Defines the GaussianNB model class.."""

from typing import Any, Dict, List, Optional

import logging
import numpy as np
import pandas as pd
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from sklearn.naive_bayes import GaussianNB
from skopt.space import Categorical, Real

logging.getLogger('ml_grid').debug("Imported gaussiannb class")


class GaussianNBWrapper(GaussianNB):
    """A wrapper for GaussianNB to handle integer-mapped priors for Bayesian search.

    This wrapper allows for the use of integer-based categorical variables for the
    'priors' parameter in Bayesian optimization, which is then mapped to the
    actual list of prior probabilities.
    """

    def set_params(self, **params: Any) -> "GaussianNBWrapper":
        """Sets the parameters of the estimator.

        This method intercepts the 'priors' parameter if it's an integer index
        and maps it to the corresponding list of prior probabilities before
        passing it to the parent's set_params method.

        Args:
            **params: Estimator parameters.

        Returns:
            The instance with updated parameters.
        """
        prior_mapping: Dict[int, Optional[List[float]]] = {
            0: None,  # Default priors (based on the class distribution in the dataset)
            1: [0.5, 0.5],  # Equal probabilities
            2: [0.6, 0.4],  # Slight imbalance favoring class 0
            3: [0.4, 0.6],  # Slight imbalance favoring class 1
            4: [0.7, 0.3],  # Moderate imbalance favoring class 0
            5: [0.3, 0.7],  # Moderate imbalance favoring class 1
            6: [0.8, 0.2],  # Strong imbalance favoring class 0
            7: [0.2, 0.8],  # Strong imbalance favoring class 1
            8: [0.9, 0.1],  # Extreme imbalance favoring class 0
            9: [0.1, 0.9],  # Extreme imbalance favoring class 1
        }

        if "priors" in params:
            priors_idx = params.pop("priors")
            params["priors"] = prior_mapping[priors_idx]
        return super().set_params(**params)


class GaussianNBClassifierClass:
    """A GaussianNB classifier with support for hyperparameter tuning.

    This class defines the parameter space for the GaussianNB classifier and
    initializes it with either a standard implementation or a wrapper for

    Bayesian search.
    """

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ) -> None:
        """Initializes the GaussianNBClassifierClass.

        Args:
            X (Optional[pd.DataFrame]): The input features. Defaults to None.
            y (Optional[pd.Series]): The target variable. Defaults to None.
            parameter_space_size (Optional[str]): The size of the parameter
              space. Defaults to None.
        """
        self.X = X
        self.y = y

        if not global_parameters.bayessearch:
            self.algorithm_implementation = GaussianNB()
        else:
            self.algorithm_implementation = (
                GaussianNBWrapper()
            )  # Wrapper necessary for passing priors to bayescv

        self.method_name = "GaussianNB"

        if global_parameters.bayessearch:
            # For BayesSearchCV, use distributions from skopt.space
            self.parameter_space = {
                'var_smoothing': Real(1e-9, 1e-2, prior='log-uniform'),
                'priors': Categorical([0, 1, 2])  # Integer mapping
            }

            # Log parameter space for verification
            logging.getLogger('ml_grid').debug(f"Parameter Space: {self.parameter_space}")

        else:
            # For traditional grid search, use lists
            self.parameter_space = {
                "priors": [
                    None,
                    [0.1, 0.9],
                    [0.9, 0.1],
                    [0.7, 0.3],
                    [0.3, 0.7],
                    [0.5, 0.5],
                    [0.6, 0.4],
                    [0.4, 0.6],
                ],  # Enumerates possible values as a list
                "var_smoothing": list(param_space.ParamSpace(parameter_space_size).param_dict.get("log_small")) + [1e-9],
            }
