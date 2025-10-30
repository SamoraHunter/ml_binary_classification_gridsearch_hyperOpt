from typing import Any, Dict, List

from aeon.classification.distance_based import ShapeDTW
from ml_grid.pipeline.data import pipe


class ShapeDTW_class:
    """A wrapper for the aeon ShapeDTW time-series classifier.

    This class provides a consistent interface for the ShapeDTW classifier,
    including defining a hyperparameter search space. This classifier is
    intended for univariate time series only.

    Attributes:
        algorithm_implementation: An instance of the aeon ShapeDTW classifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: ShapeDTW
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the ShapeDTW_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation = ShapeDTW()
        self.method_name = "ShapeDTW"
        self.parameter_space = {
            "n_neighbours": [-1],
            "subsequence_length": ["sqrt(n_timepoints)"],
            "shape_descriptor_function": ["raw"],
            "params": [None],
            "shape_descriptor_functions": [["raw", "derivative"]],
            "metric_params": [None],
            "n_jobs": [n_jobs_model_val],
        }
