from typing import Any, Dict, List

from aeon.classification.feature_based import SummaryClassifier
from sklearn.ensemble import RandomForestClassifier
from ml_grid.pipeline.data import pipe


class SummaryClassifier_class:
    """A wrapper for the aeon SummaryClassifier time-series classifier.

    This class provides a consistent interface for the SummaryClassifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon SummaryClassifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: SummaryClassifier
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the SummaryClassifier_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val
        random_state_val = ml_grid_object.global_params.random_state_val

        self.algorithm_implementation = SummaryClassifier()
        self.method_name = "SummaryClassifier"
        self.parameter_space = {
            "summary_functions": [
                "mean",
                "std",
                "min",
                "max",
                "median",
                "sum",
                "skew",
                "kurt",
                "var",
                "mad",
                "sem",
                "nunique",
                "count",
            ],
            "summary_quantiles": [None, [0.25, 0.5, 0.75]],
            "estimator": [None, RandomForestClassifier(n_estimators=200)],
            "n_jobs": [n_jobs_model_val],
            "random_state": [random_state_val],
        }
