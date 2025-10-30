from typing import Any, Dict, List

from aeon.classification.hybrid._hivecote_v1 import HIVECOTEV1
from ml_grid.pipeline.data import pipe


class HIVECOTEV1_class:
    """A wrapper for the aeon HIVECOTEV1 time-series classifier.

    This class provides a consistent interface for the HIVECOTEV1 classifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon HIVECOTEV1 classifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: HIVECOTEV1
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the HIVECOTEV1_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """
        random_state_val = ml_grid_object.global_params.random_state_val

        verbose_param = ml_grid_object.verbose

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation = HIVECOTEV1()
        self.method_name = "HIVECOTEV1"

        self.parameter_space = {
            "stc_params": [None],
            "tsf_params": [None],
            "rise_params": [None],
            "cboss_params": [None],
            "verbose": [verbose_param],
            "n_jobs": [n_jobs_model_val],
            "random_state": [random_state_val],
        }
