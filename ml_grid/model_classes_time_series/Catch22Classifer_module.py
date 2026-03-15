from typing import Any, Dict, List

from aeon.classification.feature_based import Catch22Classifier
from sklearn.ensemble import RandomForestClassifier
from skopt.space import Categorical
from sklearn.tree import DecisionTreeClassifier

from ml_grid.pipeline.data import pipe


class Catch22Classifier_class:
    """A wrapper for the aeon Catch22Classifier time-series classifier.

    This class provides a consistent interface for the Catch22Classifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon Catch22Classifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: Catch22Classifier
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the Catch22Classifier_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """

        random_state_val = ml_grid_object.global_params.random_state_val
        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation: Catch22Classifier = Catch22Classifier()
        self.method_name: str = "Catch22Classifier"
        self.parameter_space: Dict[str, List[Any]]

        if getattr(ml_grid_object.global_params, "test_mode", False):
            self.parameter_space = {
                "features": ["all"],
                "estimator": [DecisionTreeClassifier(max_depth=2)],
                "n_jobs": [1],
            }
            return

        if ml_grid_object.global_params.bayessearch:
            self.parameter_space = {
                "features": Categorical(
                    ["all", ("DN_HistogramMode_5", "DN_HistogramMode_10")]
                ),
                "catch24": Categorical([True, False]),
                "outlier_norm": Categorical([True, False]),
                "replace_nans": Categorical([True, False]),
                "use_pycatch22": Categorical([True, False]),
                "estimator": Categorical(
                    [
                        RandomForestClassifier(n_estimators=200),
                        DecisionTreeClassifier(),
                    ]
                ),
                "random_state": [random_state_val],
                "n_jobs": [n_jobs_model_val],
            }
        else:
            self.parameter_space = {
                "features": ["all", ("DN_HistogramMode_5", "DN_HistogramMode_10")],
                "catch24": [True, False],
                "outlier_norm": [True, False],
                "replace_nans": [True, False],
                "use_pycatch22": [True, False],
                "estimator": [
                    RandomForestClassifier(n_estimators=200),
                    DecisionTreeClassifier(),
                ],
                "random_state": [random_state_val],
                "n_jobs": [n_jobs_model_val],
            }
