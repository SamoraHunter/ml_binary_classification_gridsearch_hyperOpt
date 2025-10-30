from typing import Any, Dict, List

from aeon.classification.deep_learning import (
    IndividualInceptionClassifier,
)

from ml_grid.pipeline.data import pipe
from ml_grid.util.param_space import ParamSpace


class IndividualInceptionClassifier_class:
    """A wrapper for the aeon IndividualInceptionClassifier.

    This class provides a consistent interface for the IndividualInceptionClassifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon
            IndividualInceptionClassifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: IndividualInceptionClassifier
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the IndividualInceptionClassifier_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """

        random_state_val = ml_grid_object.global_params.random_state_val

        verbose_param = ml_grid_object.verbose

        param_space = ParamSpace(
            ml_grid_object.local_param_dict.get("param_space_size")
        )

        log_epoch = param_space.param_dict.get("log_epoch")

        self.algorithm_implementation = IndividualInceptionClassifier()
        self.method_name = "IndividualInceptionClassifier"

        self.parameter_space = {
            "depth": [6, 8, 10],
            "nb_filters": [32, 64, 128],
            "nb_conv_per_layer": [3, 4, 5],
            "kernel_size": [30, 40, 50],
            "use_max_pooling": [True, False],
            "max_pool_size": [2, 3, 4],
            "strides": [1, 2],
            "dilation_rate": [1, 2],
            "padding": ["same", "valid"],
            "activation": ["relu", "elu"],
            "use_bias": [True, False],
            "use_residual": [True, False],
            "use_bottleneck": [True, False],
            "bottleneck_size": [16, 32, 64],
            "use_custom_filters": [True, False],
            "batch_size": [32, 64, 128],
            "use_mini_batch_size": [True, False],
            "n_epochs": [log_epoch],
            #'callbacks': [None, [ReduceOnPlateau(), ModelCheckpoint()]],
            #'file_path': ['./', '/path/to/save'],
            "save_best_model": [False],  # Whether or not to save the best model
            "save_last_model": [False],  # Whether or not to save the last model
            #'best_file_name': ['best_model', 'model_best'],
            #'last_file_name': ['last_model', 'model_last'],
            "random_state": [random_state_val],
            "verbose": [verbose_param],
            #'optimizer': [Adam(), RMSprop(), SGD()],
            "loss": ["categorical_crossentropy", "binary_crossentropy"],
            "metrics": ["accuracy"],
        }
