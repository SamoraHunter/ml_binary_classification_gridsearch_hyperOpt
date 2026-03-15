from typing import Any, Dict, List

import keras
from aeon.classification.deep_learning import ResNetClassifier
from skopt.space import Categorical, Integer, Real

from ml_grid.pipeline.data import pipe
from ml_grid.util.param_space import ParamSpace


class ResNetClassifier_class:
    """A wrapper for the aeon ResNetClassifier time-series classifier.

    This class provides a consistent interface for the ResNetClassifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon ResNetClassifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: ResNetClassifier
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the ResNetClassifier_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """

        verbose_param = ml_grid_object.verbose

        param_space = ParamSpace(
            ml_grid_object.local_param_dict.get("param_space_size")
        )

        log_epoch = param_space.param_dict.get("log_epoch")

        self.algorithm_implementation = ResNetClassifier()
        self.method_name = "ResNetClassifier"

        gp = ml_grid_object.global_params
        test_mode = getattr(gp, "test_mode", False)
        if not test_mode and hasattr(gp, "__dict__"):
            test_mode = gp.__dict__.get("test_mode", False)

        if test_mode:
            self.parameter_space = {
                "n_residual_blocks": [1],
                "n_conv_per_residual_block": [1],
                "n_filters": [16],
                "strides": [1],
                "dilation_rate": [1],
                "padding": ["same"],
                "activation": ["relu"],
                "use_bias": [True],
                "n_epochs": [2],
                "batch_size": [16],
                "use_mini_batch_size": [True],
                "callbacks": [None],
                "save_best_model": [False],
                "save_last_model": [False],
                "verbose": [0],
                "loss": ["binary_crossentropy"],
                "optimizer": [keras.optimizers.Adam()],
                "metrics": [("accuracy",)],
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
                "n_residual_blocks": Categorical(
                    [2, 3, 4]
                ),  # The number of residual blocks of ResNet's model
                "n_conv_per_residual_block": Categorical(
                    [2, 3, 4]
                ),  # The number of convolution blocks in each residual block
                "n_filters": Categorical(
                    [64, 128, 256]
                ),  # The number of convolution filters for all the convolution layers in the same residual block
                "strides": Categorical(
                    [1, 2]
                ),  # The strides of convolution kernels in each of the convolution layers in one residual block
                "dilation_rate": Categorical(
                    [1, 2]
                ),  # The dilation rate of the convolution layers in one residual block
                "padding": Categorical(
                    ["same", "valid"]
                ),  # The type of padding used in the convolution layers in one residual block
                "activation": Categorical(
                    ["relu", "tanh"]
                ),  # Keras activation used in the convolution layers in one residual block
                "use_bias": Categorical(
                    [True, False]
                ),  # Condition on whether or not to use bias values in the convolution layers in one residual block
                "n_epochs": n_epochs_param,  # The number of epochs to train the model
                "batch_size": Categorical(
                    [16, 32, 64]
                ),  # The number of samples per gradient update
                "use_mini_batch_size": Categorical(
                    [True, False]
                ),  # Condition on using the mini-batch size formula Wang et al.
                "callbacks": [None],  # List of tf.keras.callbacks.Callback objects
                "save_best_model": [False],  # Whether or not to save the best model
                "save_last_model": [False],  # Whether or not to save the last model
                "verbose": [verbose_param],  # Whether to output extra information
                "loss": ["binary_crossentropy"],  # Fit parameter for the keras model
                "optimizer": Categorical(
                    [keras.optimizers.Adadelta(), keras.optimizers.Adam()]
                ),  # Optimizer for the model
                # Use a tuple for metrics to make it hashable for skopt
                "metrics": [("accuracy",)],
            }
        else:
            self.parameter_space = {
                "n_residual_blocks": [2, 3, 4],
                "n_conv_per_residual_block": [2, 3, 4],
                "n_filters": [64, 128, 256],
                "strides": [1, 2],
                "dilation_rate": [1, 2],
                "padding": ["same", "valid"],
                "activation": ["relu", "tanh"],
                "use_bias": [True, False],
                "n_epochs": [log_epoch],
                "batch_size": [16, 32, 64],
                "use_mini_batch_size": [True, False],
                "callbacks": [None],
                "save_best_model": [False],
                "save_last_model": [False],
                "verbose": [verbose_param],
                "loss": ["binary_crossentropy"],
                "optimizer": [
                    keras.optimizers.Adadelta(),
                    keras.optimizers.Adam(),
                ],
                # Use a tuple for metrics to make it hashable
                "metrics": [("accuracy",)],
            }
