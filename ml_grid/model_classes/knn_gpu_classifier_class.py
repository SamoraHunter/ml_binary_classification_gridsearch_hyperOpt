from typing import Optional

import pandas as pd
from ml_grid.model_classes.knn_wrapper_class import KNNWrapper
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Integer

print("Imported knn__gpu class")

class knn__gpu_wrapper_class:
    """KNN with GPU support, including Bayesian and non-Bayesian parameter space."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        """Initializes the knn__gpu_wrapper_class.

        Args:
            X (Optional[pd.DataFrame]): Feature matrix for training.
                Defaults to None.
            y (Optional[pd.Series]): Target vector for training.
                Defaults to None.
            parameter_space_size (Optional[str]): Size of the parameter space for
                optimization. Defaults to None.
        """
        self.X = X
        self.y = y

        # Initialize KNNWrapper for GPU support
        self.algorithm_implementation = KNNWrapper()
        self.method_name = "knn__gpu"

        # Define the parameter vector space
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        knn_n_jobs = global_parameters.knn_n_jobs  # Get the number of jobs from global parameters

        if global_parameters.bayessearch:
            # Bayesian Optimization: Use skopt's Integer and Categorical for the parameter space
            self.parameter_space = {
                "algorithm": Categorical(["auto", "ball_tree", "kd_tree", "brute"]),
                "metric": Categorical(["minkowski"]),
                "metric_params": [None],
                "n_jobs": Categorical([knn_n_jobs]),  # Number of jobs from global parameters
                "n_neighbors": Integer(1, self.X.shape[0] - 1),  # Integer range for n_neighbors
                "p": Integer(1, 5),  # Integer range for p
                "device": Categorical(["gpu"]),  # Device set to GPU
                "mode": Categorical(["arrays", "hdf5"]),  # Categorical choice for mode
                "scoring": Categorical(["accuracy"]),  # Categorical choice for scoring metric
            }
        else:
            # Traditional Grid Search: Define parameter space using lists for grid search
            self.parameter_space = {
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "metric": ["minkowski"],
                "metric_params": [None],
                "n_jobs": [knn_n_jobs],  # Number of jobs from global parameters
                "n_neighbors": list(self.parameter_vector_space.param_dict.get("log_med")),
                "p": list(self.parameter_vector_space.param_dict.get("log_med")),
                "device": ["gpu"],  # Device set to GPU
                "mode": ["arrays", "hdf5"],  # Mode options
                "scoring": ["accuracy"],  # Scoring metric set to accuracy
            }

        return None
