from typing import Any, Dict, List

import keras
from aeon.classification.deep_learning import MLPClassifier
from skopt.space import Categorical, Integer, Real

from ml_grid.pipeline.data import pipe


class MLPClassifier_class:
    """A wrapper for the aeon MLPClassifier time-series classifier.

    This class provides a consistent interface for the MLPClassifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon MLPClassifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: MLPClassifier
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the MLPClassifier_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """

        random_state_val = ml_grid_object.global_params.random_state_val

        verbose_param = ml_grid_object.verbose

        log_epoch = ml_grid_object.local_param_dict.get("log_epoch", [100])

        self.algorithm_implementation = MLPClassifier()
        self.method_name = "MLPClassifier"

        gp = ml_grid_object.global_params
        test_mode = getattr(gp, "test_mode", False)
        if not test_mode and hasattr(gp, "__dict__"):
            test_mode = gp.__dict__.get("test_mode", False)

        if test_mode:
            self.parameter_space = {
                "n_epochs": [2],
                "batch_size": [32],
                "random_state": [random_state_val],
                "verbose": [0],
                "loss": ["binary_crossentropy"],
                "save_best_model": [False],
                "save_last_model": [False],
                "optimizer": [keras.optimizers.Adam()],
                # Inner list is the param value; outer list is the search space.
                # Avoids tuple ambiguity that causes aeon's _metrics to never be set.
                "metrics": [["accuracy"]],
                "activation": ["relu"],
                "use_bias": [True],
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
                "n_epochs": n_epochs_param,
                "batch_size": Categorical([8, 16, 32]),
                "random_state": [random_state_val],
                "verbose": [verbose_param],
                "loss": Categorical(["binary_crossentropy"]),
                "save_best_model": [False],
                "save_last_model": [False],
                "optimizer": Categorical(
                    [keras.optimizers.Adadelta(), keras.optimizers.Adam()]
                ),
                # Inner list is the param value; outer list is the search space.
                # Avoids tuple ambiguity that causes aeon's _metrics to never be set.
                "metrics": [["accuracy"]],
                "activation": Categorical(["sigmoid", "relu"]),
                "use_bias": Categorical([True, False]),
            }
        else:
            self.parameter_space = {
                "n_epochs": [log_epoch],
                "batch_size": [8, 16, 32],
                "random_state": [random_state_val],
                "verbose": [verbose_param],
                "loss": ["binary_crossentropy"],
                "save_best_model": [False],
                "save_last_model": [False],
                "optimizer": [
                    keras.optimizers.Adadelta(),
                    keras.optimizers.Adam(),
                ],
                # Inner list is the param value; outer list is the search space.
                # Avoids tuple ambiguity that causes aeon's _metrics to never be set.
                "metrics": [["accuracy"]],
                "activation": ["sigmoid", "relu"],
                "use_bias": [True, False],
            }
