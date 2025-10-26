from typing import Optional

import pandas as pd
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Integer, Categorical, Real
import logging

logging.getLogger('ml_grid').debug("Imported KNeighborsClassifier class")

class knn_classifiers_class:
    """KNeighborsClassifier with support for both Bayesian and non-Bayesian parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        """Initializes the knn_classifiers_class.

        Args:
            X (Optional[pd.DataFrame]): Feature matrix for training.
                Defaults to None.
            y (Optional[pd.Series]): Target vector for training.
                Defaults to None.
            parameter_space_size (Optional[str]): Size of the parameter space for
                optimization. Defaults to None.
        """
        knn_n_jobs = global_parameters.knn_n_jobs  # Get the number of jobs from global parameters

        self.X = X
        self.y = y

        # Initialize KNeighborsClassifier
        self.algorithm_implementation = KNeighborsClassifier()
        self.method_name = "KNeighborsClassifier"

        # Define the parameter vector space
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        if global_parameters.bayessearch:
            # Bayesian Optimization: Use skopt's Real, Integer, and Categorical for continuous, integer, and categorical parameters
            self.parameter_space = {
                "algorithm": Categorical(["auto", "ball_tree", "kd_tree", "brute"]),
                "leaf_size": Integer(10, 100),  # Integer range for leaf_size
                "metric": Categorical(["minkowski"]),  # Categorical choice for metric
                "metric_params": Categorical([None]),  # No parameter for the metric
                "n_jobs": Categorical([knn_n_jobs]),  # Set the number of jobs to the global param
                "n_neighbors": Integer(1, self.X.shape[0] - 1),  # Integer range for n_neighbors
                "p": Integer(1, 5),  # Integer range for p (distance metric parameter)
                "weights": Categorical(["uniform", "distance"]),  # Categorical choice for weights
            }
        else:
            # Traditional Grid Search: Define parameter space using lists for traditional grid search
            self.parameter_space = {
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "leaf_size": list(self.parameter_vector_space.param_dict.get("log_large_long")),
                "metric": ["minkowski"],
                "metric_params": [None],
                "n_jobs": [knn_n_jobs],
                "n_neighbors": list(self.parameter_vector_space.param_dict.get("log_med")),
                "p": list(self.parameter_vector_space.param_dict.get("log_med")),
                "weights": ["uniform", "distance"],
            }

        return None
