from typing import Any, Dict, List

from aeon.classification.feature_based._signature_classifier import SignatureClassifier
from ml_grid.pipeline.data import pipe


class SignatureClassifier_class:
    """A wrapper for the aeon SignatureClassifier time-series classifier."""

    def __init__(self, ml_grid_object: pipe):
        """Initializes the SignatureClassifier_class.

        Args:
            ml_grid_object (pipe): The main data pipeline object, which contains
                data and global parameters.
        """

        random_state_val = ml_grid_object.global_params.random_state_val

        self.algorithm_implementation: SignatureClassifier = SignatureClassifier()

        self.method_name: str = "SignatureClassifier"

        self.parameter_space: Dict[str, List[Any]] = {
            "random_state": [random_state_val],
        }
