"""CatBoost Classifier.

This module contains the CatBoostClassifierClass, which is a configuration
class for the CatBoostClassifier. It provides parameter spaces for
grid search and Bayesian optimization.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from skopt.space import Categorical, Real, Integer
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
import logging


class CatBoostClassifierClass:
    """A class for the CatBoost Classifier.

    This class encapsulates the CatBoostClassifier, providing a flexible way to
    define parameter spaces for hyperparameter tuning. It supports both Bayesian
    optimization using `skopt` and traditional grid/random search.
    """

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ) -> None:
        """Initializes the CatBoostClassifierClass.

        Args:
            X (Optional[pd.DataFrame]): The input features. Defaults to None.
            y (Optional[pd.Series]): The target variable. Defaults to None.
            parameter_space_size (Optional[str]): The size of the parameter
              space. Defaults to None.

        Raises:
            ValueError: If `parameter_space_size` is not a valid key (though current
                implementation does not explicitly raise this).
        """
        self.X: Optional[pd.DataFrame] = X
        self.y: Optional[pd.Series] = y

        # Use CatBoostClassifier directly
        self.algorithm_implementation: CatBoostClassifier = CatBoostClassifier()
        self.method_name: str = "CatBoostClassifier"

        self.parameter_space: Union[List[Dict[str, Any]], Dict[str, Any]]
        # Define parameter space for Bayesian search or traditional grid search
        if global_parameters.bayessearch:
            self.parameter_space = {
                "iterations": Integer(100, 1000),
                "learning_rate": Real(0.01, 0.3, prior="uniform"),
                "depth": Integer(4, 10),
                "l2_leaf_reg": Real(1e-5, 1, prior="log-uniform"),
                "random_strength": Real(1e-5, 1, prior="log-uniform"),
                "rsm": Real(0.8, 1.0, prior="uniform"),
                "loss_function": Categorical(["Logloss", "CrossEntropy"]),
                "eval_metric": Categorical(["Accuracy", "AUC"]),
                "bootstrap_type": Categorical(["Bernoulli", "MVS"]),
                "subsample": Real(0.8, 1.0, prior="uniform"),
                "max_bin": Integer(32, 128),
                "grow_policy": Categorical(["SymmetricTree", "Depthwise", "Lossguide"]),
                "min_data_in_leaf": Integer(1, 7),
                "one_hot_max_size": Integer(2, 10),
                "leaf_estimation_method": Categorical(["Newton", "Gradient"]),
                "fold_permutation_block": Integer(1, 5),
                "od_pval": Real(1e-9, 0.1, prior="log-uniform"),
                "od_wait": Integer(10, 30),
                "verbose": Categorical([0]),
                "allow_const_label": Categorical([True]),
            }
            logging.getLogger("ml_grid").debug(
                f"Bayesian Parameter Space for CatBoost: {self.parameter_space}"
            )
        else:
            # Grid search parameter space must be a list of dicts
            self.parameter_space = [
                {
                    "iterations": [100, 200, 500, 1000],
                    "learning_rate": [0.01, 0.05, 0.1, 0.3],
                    "depth": [4, 6, 8, 10],
                    "l2_leaf_reg": [1e-5, 1e-3, 0.1, 1],
                    "random_strength": [1e-5, 1e-3, 0.1, 1],
                    "rsm": [0.8, 1],
                    "loss_function": ["Logloss", "CrossEntropy"],
                    "eval_metric": ["Accuracy", "AUC"],
                    "bootstrap_type": ["Bernoulli", "MVS"],
                    "subsample": [0.8, 1],
                    "max_bin": [32, 64, 128],
                    "grow_policy": ["SymmetricTree", "Depthwise", "Lossguide"],
                    "min_data_in_leaf": [1, 3, 5, 7],
                    "one_hot_max_size": [2, 5, 10],
                    "leaf_estimation_method": ["Newton", "Gradient"],
                    "fold_permutation_block": [1, 3, 5],
                    "od_pval": [1e-9, 1e-7, 1e-5, 1e-3],
                    "od_wait": [10, 20, 30],
                    "verbose": [0],
                    "allow_const_label": [True],
                }
            ]
            logging.getLogger("ml_grid").debug(
                f"Traditional Parameter Space for CatBoost: {self.parameter_space}"
            )
