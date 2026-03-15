from typing import Any, Dict, List

from aeon.classification.convolution_based import Arsenal
from skopt.space import Categorical, Integer, Real

from ml_grid.pipeline.data import pipe


class Arsenal_class:
    """A wrapper for the aeon Arsenal time-series classifier.

    This class provides a consistent interface for the Arsenal classifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon Arsenal classifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: Arsenal
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the Arsenal_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """
        time_limit_param = ml_grid_object.global_params.time_limit_param

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        random_state_val = ml_grid_object.global_params.random_state_val

        self.algorithm_implementation = Arsenal()

        self.method_name = "Arsenal"

        gp = ml_grid_object.global_params
        test_mode = getattr(gp, "test_mode", False)
        if not test_mode and hasattr(gp, "__dict__"):
            test_mode = gp.__dict__.get("test_mode", False)

        if test_mode:
            self.parameter_space = {
                "n_kernels": [100],
                "n_estimators": [2],
                "rocket_transform": ["rocket"],
                "max_dilations_per_kernel": [16],
                "n_features_per_kernel": [3],
                "time_limit_in_minutes": [0.05],
                "contract_max_n_estimators": [5],
                "n_jobs": [1],
                "random_state": [random_state_val],
            }
        elif ml_grid_object.global_params.bayessearch:
            tl_param = time_limit_param[0]
            if not isinstance(tl_param, (Categorical, Integer, Real)):
                tl_param = Categorical([tl_param])

            self.parameter_space: Dict[str, List[Any]] = {
                "n_kernels": Categorical(
                    [1000, 2000, 3000]
                ),  # Number of kernels for each ROCKET transform.
                "n_estimators": Categorical(
                    [3, 5, 6]
                ),  # Number of estimators to build for the ensemble.
                "rocket_transform": Categorical(
                    ["rocket", "minirocket"]
                ),  # The type of Rocket transformer to use.
                "max_dilations_per_kernel": Categorical(
                    [16, 32, 64]
                ),  # MiniRocket and MultiRocket only.
                "n_features_per_kernel": Categorical(
                    [3, 4, 5]
                ),  # MultiRocket only. The number of features per kernel.
                "time_limit_in_minutes": tl_param,  # Time contract to limit build time in minutes
                "contract_max_n_estimators": Categorical(
                    [50, 100, 150]
                ),  # Max number of estimators when time_limit_in_minutes is set.
                "n_jobs": [
                    n_jobs_model_val
                ],  # The number of jobs to run in parallel for both fit and predict.
                "random_state": [
                    random_state_val
                ],  # Seed for random number generation.
            }
        else:
            self.parameter_space: Dict[str, List[Any]] = {
                "n_kernels": [
                    1000,
                    2000,
                    3000,
                ],  # Number of kernels for each ROCKET transform.
                "n_estimators": [
                    3,
                    5,
                    6,
                ],  # Number of estimators to build for the ensemble.
                "rocket_transform": [
                    "rocket",
                    "minirocket",
                ],  # The type of Rocket transformer to use.
                "max_dilations_per_kernel": [
                    16,
                    32,
                    64,
                ],  # MiniRocket and MultiRocket only.
                "n_features_per_kernel": [
                    3,
                    4,
                    5,
                ],  # MultiRocket only. The number of features per kernel.
                "time_limit_in_minutes": time_limit_param,
                "contract_max_n_estimators": [
                    50,
                    100,
                    150,
                ],  # Max number of estimators when time_limit_in_minutes is set.
                "n_jobs": [n_jobs_model_val],
                "random_state": [random_state_val],
            }
