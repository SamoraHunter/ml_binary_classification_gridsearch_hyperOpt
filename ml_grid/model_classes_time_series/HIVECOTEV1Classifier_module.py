from typing import Any, Dict, List

from aeon.classification.hybrid._hivecote_v1 import HIVECOTEV1
from ml_grid.pipeline.data import pipe


class HIVECOTEV1_class:
    """A wrapper for the aeon HIVECOTEV1 time-series classifier."""

    def __init__(self, ml_grid_object: pipe):
        """Initializes the HIVECOTEV1_class.

        Args:
            ml_grid_object (pipe): The main data pipeline object, which contains
                data and global parameters.
        """
        random_state_val = ml_grid_object.global_params.random_state_val

        verbose_param = ml_grid_object.verbose

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation: HIVECOTEV1 = HIVECOTEV1()

        self.method_name: str = "HIVECOTEV1"

        self.parameter_space: Dict[str, List[Any]] = {
            "stc_params": [None],
            "tsf_params": [None],
            "rise_params": [None],
            "cboss_params": [None],
            "verbose": [verbose_param],
            "n_jobs": [n_jobs_model_val],
            "random_state": [random_state_val],
        }
