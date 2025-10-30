from typing import Any, Dict, List

from aeon.classification.deep_learning import EncoderClassifier
from ml_grid.pipeline.data import pipe


class EncoderClassifier_class:
    """A wrapper for the aeon EncoderClassifier time-series classifier.

    This class provides a consistent interface for the EncoderClassifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon EncoderClassifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: EncoderClassifier
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the EncoderClassifier_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """
        random_state_val = ml_grid_object.global_params.random_state_val

        self.algorithm_implementation = EncoderClassifier()
        self.method_name = "EncoderClassifier"
        self.parameter_space = {
            "kernel_size": [
                [5],
                [11],
                [21],
                [5, 11],
                [5, 21],
                [11, 21],
                [5, 11, 21],
            ],  # Specifying the length of the 1D convolution windows
            "n_filters": [
                [128],
                [256],
                [512],
                [128, 256],
                [128, 512],
                [256, 512],
                [128, 256, 512],
            ],  # Specifying the number of 1D convolution filters used for each layer
            "max_pool_size": [2, 3],  # Size of the max pooling windows
            "activation": ["sigmoid", "relu", "tanh"],  # Keras activation function
            "dropout_proba": [0.0, 0.2, 0.5],  # Dropout layer probability
            "padding": ["same", "valid"],  # Type of padding used for 1D convolution
            "strides": [1, 2],  # Sliding rate of the 1D convolution filter
            "fc_units": [
                128,
                256,
                512,
            ],  # Number of units in the hidden fully connected layer
            #'file_path': ['./', './models/'],                # File path when saving the model_Checkpoint callback
            "save_best_model": [False],  # Whether or not to save the best model
            "save_last_model": [False],  # Whether or not to save the last model
            #'best_file_name': ['best_model', 'best_classifier'],  # Name of the file of the best model
            #'last_file_name': ['last_model', 'last_classifier'],  # Name of the file of the last model
            "random_state": [random_state_val],  # Seed for any needed random actions
        }
