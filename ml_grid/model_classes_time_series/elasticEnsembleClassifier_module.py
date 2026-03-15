from typing import Any, Dict, List

from aeon.classification.distance_based import ElasticEnsemble
from skopt.space import Categorical, Real

from ml_grid.pipeline.data import pipe


class ElasticEnsemble_class:
    """A wrapper for the aeon ElasticEnsemble time-series classifier.

    This class provides a consistent interface for the ElasticEnsemble
    classifier, including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon ElasticEnsemble
            classifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: ElasticEnsemble
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the ElasticEnsemble_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation = ElasticEnsemble()
        self.method_name = "ElasticEnsemble"

        if getattr(ml_grid_object.global_params, "test_mode", False):
            self.parameter_space = {
                "proportion_of_param_options": [0.1],
                "proportion_train_in_param_finding": [0.1],
                "proportion_train_for_test": [0.1],
                "n_jobs": [1],
            }
            return

        if ml_grid_object.global_params.bayessearch:
            self.parameter_space = {
                "proportion_of_param_options": Real(0.6, 0.8),
                "proportion_train_in_param_finding": Real(0.6, 0.8),
                "proportion_train_for_test": Real(0.6, 0.8),
                "n_jobs": [n_jobs_model_val],
                "majority_vote": Categorical([False, True]),
            }
        else:
            self.parameter_space = {
                "proportion_of_param_options": [0.8, 0.7, 0.6],
                "proportion_train_in_param_finding": [0.8, 0.7, 0.6],
                "proportion_train_for_test": [0.8, 0.7, 0.6],
                "n_jobs": [n_jobs_model_val],
                "majority_vote": [False, True],
            }
