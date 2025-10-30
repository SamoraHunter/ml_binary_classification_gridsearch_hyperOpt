from typing import Any, Dict, List

import keras
from aeon.classification.deep_learning import ResNetClassifier

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

        random_state_val = ml_grid_object.global_params.random_state_val

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        verbose_param = ml_grid_object.verbose

        param_space = ParamSpace(
            ml_grid_object.local_param_dict.get("param_space_size")
        )

        log_epoch = param_space.param_dict.get("log_epoch")

        self.algorithm_implementation = ResNetClassifier()
        self.method_name = "ResNetClassifier"

        self.parameter_space = {
            "n_residual_blocks": [
                2,
                3,
                4,
            ],  # The number of residual blocks of ResNet's model
            "n_conv_per_residual_block": [
                2,
                3,
                4,
            ],  # The number of convolution blocks in each residual block
            "n_filters": [
                64,
                128,
                256,
            ],  # The number of convolution filters for all the convolution layers in the same residual block
            #'kernel_sizes': [3, 5, 7],                 # The kernel size of all the convolution layers in one residual block
            "strides": [
                1,
                2,
            ],  # The strides of convolution kernels in each of the convolution layers in one residual block
            "dilation_rate": [
                1,
                2,
            ],  # The dilation rate of the convolution layers in one residual block
            "padding": [
                "same",
                "valid",
            ],  # The type of padding used in the convolution layers in one residual block
            "activation": [
                "relu",
                "tanh",
            ],  # Keras activation used in the convolution layers in one residual block
            "use_bias": [
                True,
                False,
            ],  # Condition on whether or not to use bias values in the convolution layers in one residual block
            "n_epochs": [log_epoch],  # The number of epochs to train the model
            "batch_size": [16, 32, 64],  # The number of samples per gradient update
            "use_mini_batch_size": [
                True,
                False,
            ],  # Condition on using the mini-batch size formula Wang et al.
            "callbacks": [None],  # List of tf.keras.callbacks.Callback objects
            #'file_path': ['./', './models/'],          # File path when saving model_Checkpoint callback
            "save_best_model": [False],  # Whether or not to save the best model
            "save_last_model": [False],  # Whether or not to save the last model
            #'best_file_name': ['best_model', 'best_classifier'],  # Name of the file of the best model
            #'last_file_name': ['last_model', 'last_classifier'],  # Name of the file of the last model
            "verbose": [verbose_param],  # Whether to output extra information
            "loss": [
                "categorical_crossentropy"
            ],  # Fit parameter for the keras model # 'mean_squared_error',
            "optimizer": [
                keras.optimizers.Adadelta(),
                keras.optimizers.Adam(),
            ],  # Optimizer for the model
            "metrics": [["accuracy", "mae"]],  # List of strings for metrics
        }
