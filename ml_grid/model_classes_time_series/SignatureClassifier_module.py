from typing import Any, Dict, List

from aeon.classification.feature_based._signature_classifier import SignatureClassifier

from ml_grid.pipeline.data import pipe


class SignatureClassifier_class:
    """A wrapper for the aeon SignatureClassifier time-series classifier.

    This class provides a consistent interface for the SignatureClassifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon SignatureClassifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: SignatureClassifier
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the SignatureClassifier_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """

        random_state_val = ml_grid_object.global_params.random_state_val

        self.algorithm_implementation = SignatureClassifier()
        self.method_name = "SignatureClassifier"
        self.parameter_space = {
            "random_state": [random_state_val],
        }
