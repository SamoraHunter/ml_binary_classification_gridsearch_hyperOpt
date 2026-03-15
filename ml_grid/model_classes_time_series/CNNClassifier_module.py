from typing import Any, Dict, List

# Monkeypatch sklearn.utils.validation for aeon compatibility
import sklearn.utils.validation

if not hasattr(sklearn.utils.validation, "validate_data"):
    sklearn.utils.validation.validate_data = sklearn.utils.validation.check_X_y

from skopt.space import Categorical, Integer, Real
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

        gp = ml_grid_object.global_params
        test_mode = getattr(gp, "test_mode", False)
        if not test_mode and hasattr(gp, "__dict__"):
            test_mode = gp.__dict__.get("test_mode", False)

        if test_mode:
            self.parameter_space = {
                "activation": ["relu"],
                "padding": ["valid"],
                "dilation_rate": [1],
                "use_bias": [True],
                "random_state": [random_state_val],
                "n_epochs": [2],
                "batch_size": [16],
                "verbose": [0],
                "loss": ["binary_crossentropy"],
                "metrics": [("accuracy",)],
                "save_best_model": [False],
                "save_last_model": [False],
            }
        elif ml_grid_object.global_params.bayessearch:
            n_epochs_param = log_epoch
            if (
                isinstance(n_epochs_param, list)
                and len(n_epochs_param) >= 1
                and isinstance(n_epochs_param[0], (Categorical, Integer, Real))
            ):
                n_epochs_param = n_epochs_param[0]

            self.parameter_space = {
                "activation": Categorical(["sigmoid", "relu"]),
                "padding": Categorical(["valid"]),
                "dilation_rate": Categorical([1, 2]),
                "use_bias": [True],
                "random_state": [random_state_val],
                "n_epochs": n_epochs_param,
                "batch_size": Categorical([16, 32, 64]),
                "verbose": [verbose_param],
                "loss": Categorical(["binary_crossentropy"]),
                "metrics": Categorical(["accuracy"]),
                "save_best_model": [True],
                "save_last_model": [False],
            }
        else:
            self.parameter_space = {
                "activation": ["sigmoid", "relu"],
                "padding": ["valid"],
                "dilation_rate": [1, 2],
                "use_bias": [True],
                "random_state": [random_state_val],
                "n_epochs": log_epoch,
                "batch_size": [16, 32, 64],
                "verbose": [verbose_param],
                "loss": ["binary_crossentropy"],
                "metrics": ["accuracy"],
                "save_best_model": [True],
                "save_last_model": [False],
            }
