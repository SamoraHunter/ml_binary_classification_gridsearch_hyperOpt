"""AdaBoost Classifier.

This module contains the adaboost_class, which is a configuration
class for the AdaBoostClassifier. It provides parameter spaces for
grid search and Bayesian optimization.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skopt.space import Categorical, Integer, Real

from ml_grid.util.global_params import global_parameters


class AdaBoostClassifierClass:
    """A class for AdaBoostClassifier that handles both Bayesian and grid search.

    This class encapsulates the AdaBoostClassifier, providing a flexible way to
    define parameter spaces for hyperparameter tuning. It supports both Bayesian
    optimization using `skopt` and traditional grid/random search.
    """

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        """Initializes the AdaBoostClassifierClass.

        Args:
            X: The input features, not used in this class.
            y: The target variable, not used in this class.
            parameter_space_size: The size of the parameter space, not used.

        Raises:
            ValueError: If `parameter_space_size` is not a valid key (though current
                implementation does not explicitly raise this).
        """
        self.X: Optional[pd.DataFrame] = X
        self.y: Optional[pd.Series] = y

        self.algorithm_implementation: AdaBoostClassifier = AdaBoostClassifier()
        self.method_name: str = "AdaBoostClassifier"
        self.parameter_space: List[Dict[str, Any]]
        is_bayes_search = global_parameters.bayessearch

        if getattr(global_parameters, "test_mode", False):
            self.parameter_space = [
                {
                    "n_estimators": [2],
                }
            ]
        elif is_bayes_search:
            # For BayesSearchCV, define the search space using skopt.space objects.
            # We define separate spaces for each estimator type.
            self.parameter_space = [
                {
                    "estimator": Categorical([DecisionTreeClassifier(random_state=1)]),
                    "estimator__max_depth": Integer(1, 5),
                    "n_estimators": Integer(50, 500),
                    "learning_rate": Real(0.01, 1.0, prior="log-uniform"),
                    "algorithm": Categorical(["SAMME"]),
                },
                {
                    "estimator": Categorical([SVC(random_state=1, probability=True)]),
                    "estimator__C": Real(0.1, 10, prior="log-uniform"),
                    "estimator__kernel": Categorical(["rbf", "poly"]),
                    "n_estimators": Integer(
                        50, 200
                    ),  # SVC is slower, so fewer estimators
                    "learning_rate": Real(0.01, 1.0, prior="log-uniform"),
                    "algorithm": Categorical(["SAMME"]),
                },
            ]
        else:
            # For Grid/Random search, define parameter spaces for each valid combination
            self.parameter_space = [
                {
                    # SAMME with DecisionTree
                    "estimator": [
                        DecisionTreeClassifier(max_depth=1, random_state=1),
                        DecisionTreeClassifier(max_depth=2, random_state=1),
                        DecisionTreeClassifier(max_depth=3, random_state=1),
                    ],
                    "estimator__max_depth": [1, 2, 3],
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0],
                    "algorithm": ["SAMME"],
                },
                {
                    # SAMME with SVC (requires probability=True)
                    "estimator": [SVC(random_state=1, probability=True)],
                    "estimator__C": [0.1, 1, 10],
                    "estimator__kernel": ["rbf", "poly"],
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0],
                    "algorithm": ["SAMME"],
                },
            ]
