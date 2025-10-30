from typing import Any, Dict, List

from aeon.classification.dictionary_based._tde import IndividualTDE
from ml_grid.pipeline.data import pipe


class IndividualTDE_class:
    """A wrapper for the aeon IndividualTDE time-series classifier.

    This class provides a consistent interface for the IndividualTDE classifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon IndividualTDE
            classifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: IndividualTDE
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the IndividualTDE_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """

        random_state_val = ml_grid_object.global_params.random_state_val

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation: IndividualTDE = IndividualTDE()
        self.method_name: str = "IndividualTDE"

        self.parameter_space = {
            "window_size": [5, 10, 15],
            "word_length": [4, 8, 12],
            "norm": [True, False],
            "levels": [1, 2, 3],
            "igb": [True, False],
            "alphabet_size": [3, 4, 5],
            "bigrams": [True, False],
            "dim_threshold": [0.8, 0.85, 0.9],
            "max_dims": [15, 20, 25],
            "typed_dict": [True, False],
            "n_jobs": [n_jobs_model_val],
            "random_state": [random_state_val],
        }
