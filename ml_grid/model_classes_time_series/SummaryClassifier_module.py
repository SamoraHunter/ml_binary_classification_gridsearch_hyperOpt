from typing import Any, Dict, List

from aeon.classification.feature_based import SummaryClassifier
from sklearn.ensemble import RandomForestClassifier
from skopt.space import Categorical

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

        if getattr(ml_grid_object.global_params, "test_mode", False):
            self.parameter_space = {
                "summary_stats": [("mean", "std")],
                "estimator": [RandomForestClassifier(n_estimators=10)],
                "n_jobs": [1],
            }
            return

        if ml_grid_object.global_params.bayessearch:
            self.parameter_space = {
                "summary_stats": Categorical(
                    [
                        ("mean", "std", "min", "max"),
                        ("mean", "std", "skew", "kurt", "median"),
                        (
                            "mean",
                            "std",
                            "min",
                            "max",
                            "skew",
                            "kurt",
                            "median",
                            "sum",
                        ),
                    ]
                ),
                "estimator": Categorical(
                    [None, RandomForestClassifier(n_estimators=200)]
                ),
                "n_jobs": [n_jobs_model_val],
                "random_state": [random_state_val],
            }
        else:
            self.parameter_space = {
                # The summary_stats parameter expects a list of strings. To make this
                # searchable with skopt, we provide a list of tuples. Tuples are
                # hashable and can be used as categories in a Categorical space.
                "summary_stats": [
                    ("mean", "std", "min", "max"),
                    ("mean", "std", "skew", "kurt", "median"),
                    ("mean", "std", "min", "max", "skew", "kurt", "median", "sum"),
                ],
                "estimator": [None, RandomForestClassifier(n_estimators=200)],
                "n_jobs": [n_jobs_model_val],
                "random_state": [random_state_val],
            }
