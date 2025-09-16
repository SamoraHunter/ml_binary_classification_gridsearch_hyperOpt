from typing import Any, Dict, List

from aeon.classification.convolution_based import Arsenal
from ml_grid.pipeline.data import pipe


class Arsenal_class:
    """A wrapper for the aeon Arsenal time-series classifier."""

    def __init__(self, ml_grid_object: pipe):
        """Initializes the Arsenal_class.

        Args:
            ml_grid_object (pipe): The main data pipeline object, which contains
                data and global parameters.
        """
        time_limit_param = ml_grid_object.global_params.time_limit_param

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        random_state_val = ml_grid_object.global_params.random_state_val

        self.algorithm_implementation: Arsenal = Arsenal()

        self.method_name: str = "Arsenal"

        self.parameter_space: Dict[str, List[Any]] = {
            "num_kernels": [
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
            ],  # The type of Rocket transformer to use. #, "multirocket" # broken
            # Valid inputs = ["rocket", "minirocket", "multirocket"].
            "max_dilations_per_kernel": [
                16,
                32,
                64,
            ],  # MiniRocket and MultiRocket only. The maximum number of dilations per kernel.
            "n_features_per_kernel": [
                3,
                4,
                5,
            ],  # MultiRocket only. The number of features per kernel.
            "time_limit_in_minutes": time_limit_param,  # Time contract to limit build time in minutes, overriding n_estimators. Default of 0 means n_estimators is used.
            "contract_max_n_estimators": [
                50,
                100,
                150,
            ],  # Max number of estimators when time_limit_in_minutes is set.
            #'save_transformed_data': [True, False],  # Save the data transformed in fit for use in _get_train_probs.
            "n_jobs": [
                n_jobs_model_val
            ],  # The number of jobs to run in parallel for both fit and predict. -1 means using all processors.
            "random_state": [random_state_val],  # Seed for random number generation.
        }
