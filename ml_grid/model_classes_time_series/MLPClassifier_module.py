from typing import Any, Dict, List

import keras
from aeon.classification.deep_learning import MLPClassifier

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

        # This seems to be trying to get a parameter that is not set in the time-series grid space.
        # It might be better to define it directly or add it to the GA grid.
        # For now, we'll default to a reasonable value if it's not found.
        log_epoch = ml_grid_object.local_param_dict.get("log_epoch", [100])

        self.algorithm_implementation = MLPClassifier()
        self.method_name = "MLPClassifier"

        self.parameter_space = {
            "n_epochs": [log_epoch],  # Number of epochs to train the model
            "batch_size": [8, 16, 32],  # Number of samples per gradient update
            "random_state": [random_state_val],  # Seed for random number generation
            "verbose": [verbose_param],  # Whether to output extra information
            "loss": [
                "binary_crossentropy"
            ],  # Fit parameter for the Keras model #must be binary? # 'mean_squared_error',
            #'file_path': ['./', '/models'],                         # File path when saving ModelCheckpoint callback
            "save_best_model": [False],  # Whether or not to save the best model
            "save_last_model": [False],  # Whether or not to save the last model
            #'best_file_name': ['best_model', 'top_model'],          # The name of the file of the best model
            #'last_file_name': ['last_model', 'final_model'],        # The name of the file of the last model
            "optimizer": [
                keras.optimizers.Adadelta(),
                keras.optimizers.Adam(),
            ],  # Keras optimizer
            "metrics": [
                ["accuracy"],
                ["accuracy", "mae"],
            ],  # List of strings for metrics
            "activation": [
                "sigmoid",
                "relu",
            ],  # Activation function used in the output linear layer
            "use_bias": [True, False],  # Whether the layer uses a bias vector
        }
