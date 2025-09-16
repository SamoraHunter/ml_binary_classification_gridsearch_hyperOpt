from typing import Any, Dict, List

from aeon.classification.distance_based import ShapeDTW
from ml_grid.pipeline.data import pipe


class ShapeDTW_class:
    """A wrapper for the aeon ShapeDTW time-series classifier."""

    def __init__(self, ml_grid_object: pipe):
        """Initializes the ShapeDTW_class.

        Note:
            This classifier is for univariate time series only.

        Args:
            ml_grid_object (pipe): The main data pipeline object, which contains
                data and global parameters.
        """

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation: ShapeDTW = ShapeDTW()

        self.method_name: str = "ShapeDTW"

        self.parameter_space: Dict[str, List[Any]] = {
            "n_neighbours": [-1],
            "subsequence_length": ["sqrt(n_timepoints)"],
            "shape_descriptor_function": ["raw"],
            "params": [None],
            "shape_descriptor_functions": [["raw", "derivative"]],
            "metric_params": [None],
            "n_jobs": [n_jobs_model_val],
        }
