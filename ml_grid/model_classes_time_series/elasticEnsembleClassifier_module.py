from typing import Any, Dict, List

from aeon.classification.distance_based import ElasticEnsemble
from ml_grid.pipeline.data import pipe


class ElasticEnsemble_class:
    """A wrapper for the aeon ElasticEnsemble time-series classifier."""

    def __init__(self, ml_grid_object: pipe):
        """Initializes the ElasticEnsemble_class.

        Args:
            ml_grid_object (pipe): The main data pipeline object, which contains
                data and global parameters.
        """

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation: ElasticEnsemble = ElasticEnsemble()

        self.method_name: str = "ElasticEnsemble"

        self.parameter_space: Dict[str, List[Any]] = {
            "proportion_of_param_options": [1.0, 0.8, 0.6],
            "proportion_train_in_param_finding": [1.0, 0.8, 0.6],
            "proportion_train_for_test": [1.0, 0.8, 0.6],
            "n_jobs": [n_jobs_model_val],
            "majority_vote": [False, True],
        }
