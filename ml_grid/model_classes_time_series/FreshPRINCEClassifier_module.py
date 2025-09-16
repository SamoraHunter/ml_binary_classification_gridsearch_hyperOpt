from typing import Any, Dict, List

from aeon.classification.feature_based._fresh_prince import FreshPRINCEClassifier
from ml_grid.pipeline.data import pipe


class FreshPRINCEClassifier_class:
    """A wrapper for the aeon FreshPRINCEClassifier time-series classifier."""

    def __init__(self, ml_grid_object: pipe):
        """Initializes the FreshPRINCEClassifier_class.

        Args:
            ml_grid_object (pipe): The main data pipeline object, which contains
                data and global parameters.
        """
        random_state_val = ml_grid_object.global_params.random_state_val

        verbose_param = ml_grid_object.verbose

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation: FreshPRINCEClassifier = FreshPRINCEClassifier()

        self.method_name: str = "FreshPRINCEClassifier"

        self.parameter_space: Dict[str, List[Any]] = {
            "default_fc_parameters": [
                "minimal",
                "efficient",
                "comprehensive",
            ],  # Set of TSFresh features to be extracted
            "n_estimators": [
                100,
                200,
                300,
            ],  # Number of estimators for the RotationForestClassifier ensemble
            "save_transformed_data": [False],  # Whether to save the transformed data
            "verbose": [
                verbose_param
            ],  # Level of output printed to the console (for information only)
            "n_jobs": [n_jobs_model_val],  # Number of jobs for parallel processing
            "chunksize": [
                None,
                100,
                200,
            ],  # Number of series processed in each parallel TSFresh job
            "random_state": [random_state_val],  # Seed for random, integer
        }
