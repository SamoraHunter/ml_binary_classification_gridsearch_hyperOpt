from typing import Any, Dict, List

from aeon.classification.convolution_based import RocketClassifier
from skopt.space import Categorical

from ml_grid.pipeline.data import pipe


class RocketClassifier_class:
    """A wrapper for the aeon RocketClassifier time-series classifier.

    This class provides a consistent interface for the RocketClassifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon RocketClassifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: RocketClassifier
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the RocketClassifier_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """

        random_state_val = ml_grid_object.global_params.random_state_val

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation = RocketClassifier()
        self.method_name = "RocketClassifier"

        gp = ml_grid_object.global_params
        test_mode = getattr(gp, "test_mode", False)
        if not test_mode and hasattr(gp, "__dict__"):
            test_mode = gp.__dict__.get("test_mode", False)

        if test_mode:
            self.parameter_space = {
                "n_kernels": [100],
                "random_state": [random_state_val],
                "estimator": [None],
                "n_jobs": [1],
            }
        elif ml_grid_object.global_params.bayessearch:
            self.parameter_space = {
                "n_kernels": Categorical([5000, 10000, 15000]),
                "random_state": [random_state_val],
                "estimator": [None],
                "n_jobs": [n_jobs_model_val],
            }
        else:
            self.parameter_space = {
                "n_kernels": [5000, 10000, 15000],
                "random_state": [random_state_val],
                "estimator": [None],
                "n_jobs": [n_jobs_model_val],
            }
