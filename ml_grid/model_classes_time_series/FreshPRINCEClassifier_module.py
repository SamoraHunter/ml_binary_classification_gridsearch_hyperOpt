from typing import Any, Dict, List

from aeon.classification.feature_based._fresh_prince import FreshPRINCEClassifier
from ml_grid.pipeline.data import pipe


class FreshPRINCEClassifier_class:
    """A wrapper for the aeon FreshPRINCEClassifier time-series classifier.

    This class provides a consistent interface for the FreshPRINCEClassifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon FreshPRINCEClassifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: FreshPRINCEClassifier
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the FreshPRINCEClassifier_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """
        random_state_val = ml_grid_object.global_params.random_state_val

        verbose_param = ml_grid_object.verbose

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation = FreshPRINCEClassifier()
        self.method_name = "FreshPRINCEClassifier"

        self.parameter_space = {
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
