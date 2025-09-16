from typing import Any, Dict, List

from aeon.classification.feature_based import TSFreshClassifier
from sklearn.ensemble import RandomForestClassifier
from ml_grid.pipeline.data import pipe


class TSFreshClassifier_class:
    """A wrapper for the aeon TSFreshClassifier time-series classifier."""

    def __init__(self, ml_grid_object: pipe):
        """Initializes the TSFreshClassifier_class.

        Args:
            ml_grid_object (pipe): The main data pipeline object, which contains
                data and global parameters.
        """

        verbose_param = ml_grid_object.verbose
        random_state_val = ml_grid_object.global_params.random_state_val
        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation: TSFreshClassifier = TSFreshClassifier()

        self.method_name: str = "TSFreshClassifier"

        self.parameter_space: Dict[str, List[Any]] = {
            "default_fc_parameters": ["minimal", "efficient", "comprehensive"],
            "relevant_feature_extractor": [True, False],
            "estimator": [None, RandomForestClassifier(n_estimators=200)],
            "verbose": [verbose_param],
            "n_jobs": [n_jobs_model_val],
            "chunksize": [None, 10, 100],
            "random_state": [random_state_val],
        }
