from typing import Any, Dict, List

# Monkeypatch sklearn.utils.validation for aeon compatibility
import sklearn.utils.validation

if not hasattr(sklearn.utils.validation, "validate_data"):
    sklearn.utils.validation.validate_data = sklearn.utils.validation.check_X_y

from aeon.classification.deep_learning import TimeCNNClassifier

from ml_grid.pipeline.data import pipe
from ml_grid.util.param_space import ParamSpace


class CNNClassifier_class:
    """A wrapper for the aeon CNNClassifier time-series classifier.

    This class provides a consistent interface for the CNNClassifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon CNNClassifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: TimeCNNClassifier
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the CNNClassifier_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """

        random_state_val = ml_grid_object.global_params.random_state_val

        verbose_param = ml_grid_object.verbose

        param_space = ParamSpace(
            ml_grid_object.local_param_dict.get("param_space_size")
        )

        log_epoch = param_space.param_dict.get("log_epoch")

        self.algorithm_implementation = TimeCNNClassifier()
        self.method_name = "CNNClassifier"

        self.parameter_space = {
            #'n_layers': [2, 3, 4],
            #'kernel_size': [3, 5, 7],
            #'n_filters': [[6, 12], [8, 16], [10, 20]],
            #'avg_pool_size': [2, 3, 4],
            "activation": ["sigmoid", "relu"],
            "padding": ["valid"],
            #'strides': [1, 2],
            "dilation_rate": [1, 2],
            "use_bias": [True],
            "random_state": [random_state_val],
            "n_epochs": [log_epoch],
            "batch_size": [16, 32, 64],
            "verbose": [verbose_param],
            "loss": ["binary_crossentropy"],
            "metrics": ["accuracy"],
            #'save_best_model': [True, False],
            #'save_last_model': [True, False],
            #'best_file_name': ['best_model', 'top_model'],
            #'last_file_name': ['last_model', 'final_model'],
        }
