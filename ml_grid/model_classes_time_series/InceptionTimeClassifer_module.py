from typing import Any, Dict, List

import keras
from aeon.classification.deep_learning import InceptionTimeClassifier

from ml_grid.pipeline.data import pipe
from ml_grid.util.param_space import ParamSpace


class InceptionTimeClassifier_class:
    """A wrapper for the aeon InceptionTimeClassifier.

    This class provides a consistent interface for the InceptionTimeClassifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon
            InceptionTimeClassifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: InceptionTimeClassifier
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the InceptionTimeClassifier_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """

        random_state_val = ml_grid_object.global_params.random_state_val

        verbose_param = ml_grid_object.verbose

        param_space = ParamSpace(
            ml_grid_object.local_param_dict.get("param_space_size")
        )

        log_epoch = param_space.param_dict.get("log_epoch")

        self.algorithm_implementation = InceptionTimeClassifier()
        self.method_name = "InceptionTimeClassifier"

        self.parameter_space = {
            "n_classifiers": [
                3,
                5,
                7,
            ],  # Number of Inception models used for the Ensemble
            "depth": [4, 6, 8],  # Number of inception modules used
            "nb_filters": [
                32,
                64,
                128,
            ],  # Number of filters used in one inception module
            "nb_conv_per_layer": [
                3,
                4,
            ],  # Number of convolution layers in each inception module
            "kernel_size": [
                30,
                40,
                50,
            ],  # Head kernel size used for each inception module
            #         'use_max_pooling': [True, False],         # Whether to use max pooling layer in inception modules #will throw error
            "max_pool_size": [2, 3],  # Size of the max pooling layer
            "strides": [
                1,
                2,
            ],  # Strides of kernels in convolution layers for each inception module
            "dilation_rate": [
                1,
                2,
            ],  # Dilation rate of convolutions in each inception module
            "padding": [
                "same",
                "valid",
            ],  # Type of padding used for convolution for each inception module
            "activation": [
                "relu",
                "tanh",
            ],  # Activation function used in each inception module
            "use_bias": [True],  # Whether to use bias values in each inception module
            "use_residual": [
                True,
                False,
            ],  # Whether to use residual connections all over Inception
            "use_bottleneck": [
                True,
                False,
            ],  # Whether to use bottlenecks all over Inception
            "bottleneck_size": [
                16,
                32,
            ],  # Bottleneck size in case use_bottleneck = True
            "use_custom_filters": [
                True,
                False,
            ],  # Whether to use custom filters in the first inception module
            "batch_size": [32, 64],  # Number of samples per gradient update
            "use_mini_batch_size": [
                True,
                False,
            ],  # Whether to use the mini batch size formula Wang et al.
            "n_epochs": [log_epoch],  # Number of epochs to train the model
            "callbacks": [None],  # List of tf.keras.callbacks.Callback objects
            #         'file_path': ['./'],                      # File path when saving model_Checkpoint callback
            "save_best_model": [False],  # Whether or not to save the best model
            "save_last_model": [False],  # Whether or not to save the last model
            #         'best_file_name': ['best_model'],         # Name of the file of the best model
            #         'last_file_name': ['last_model'],         # Name of the file of the last model
            "random_state": [random_state_val],  # Seed for random actions
            "verbose": [verbose_param],  # Whether to output extra information
            "optimizer": [
                keras.optimizers.Adam(0.01),
                keras.optimizers.SGD(0.01),
            ],  # Keras optimizer
            "loss": ["categorical_crossentropy"],  # Keras loss
            "metrics": ["accuracy"],  # Keras metrics
        }
