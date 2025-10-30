from typing import Any, Dict, List

from aeon.classification.feature_based import TSFreshClassifier
from sklearn.ensemble import RandomForestClassifier

from ml_grid.pipeline.data import pipe


class TSFreshClassifier_class:
    """A wrapper for the aeon TSFreshClassifier time-series classifier.

    This class provides a consistent interface for the TSFreshClassifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon TSFreshClassifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: TSFreshClassifier
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the TSFreshClassifier_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """

        verbose_param = ml_grid_object.verbose
        random_state_val = ml_grid_object.global_params.random_state_val
        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation = TSFreshClassifier()
        self.method_name = "TSFreshClassifier"
        self.parameter_space = {
            "default_fc_parameters": ["minimal", "efficient", "comprehensive"],
            "relevant_feature_extractor": [True, False],
            "estimator": [None, RandomForestClassifier(n_estimators=200)],
            "verbose": [verbose_param],
            "n_jobs": [n_jobs_model_val],
            "chunksize": [None, 10, 100],
            "random_state": [random_state_val],
        }
