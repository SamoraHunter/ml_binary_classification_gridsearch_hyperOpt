"""Keras Neural Network Classifier.

This module contains the NeuralNetworkClassifier_class, which is a configuration
class for the NeuralNetworkClassifier (Keras wrapper). It provides parameter
spaces for grid search and Bayesian optimization.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ml_grid.model_classes.NeuralNetworkKerasClassifier import NeuralNetworkClassifier
from ml_grid.util import param_space

logging.getLogger("ml_grid").debug("Imported NeuralNetworkClassifier class")


class NeuralNetworkClassifier_class:
    """NeuralNetworkClassifier with a predefined parameter space."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        """Initializes the NeuralNetworkClassifier_class.

        Args:
            X (Optional[pd.DataFrame]): Feature matrix for training.
                Defaults to None.
            y (Optional[pd.Series]): Target vector for training.
                Defaults to None.
            parameter_space_size (Optional[str]): Size of the parameter space for
                optimization. This is not used in the current implementation
                as the parameter space is hardcoded. Defaults to None.

        Raises:
            ValueError: If `parameter_space_size` is not a valid key (though current
                implementation does not explicitly raise this).
        """
        self.X: Optional[pd.DataFrame] = X
        self.y: Optional[pd.Series] = y

        self.algorithm_implementation: NeuralNetworkClassifier = (
            NeuralNetworkClassifier()
        )
        self.method_name: str = "NeuralNetworkClassifier"

        self.parameter_vector_space: param_space.ParamSpace = param_space.ParamSpace(
            parameter_space_size
        )

        self.parameter_space: Union[List[Dict[str, Any]], Dict[str, Any]]


        from ml_grid.util.global_params import global_parameters

        if global_parameters.bayessearch:
            from skopt.space import Categorical, Integer, Real

            self.parameter_space = [
                {
                    # Changed from: Categorical([(8, 8), (16, 8), ...])
                    # To: Categorical(["(8, 8)", "(16, 8)", ...])
                    # Tuples encoded as strings to avoid skopt's .item() error
                    "hidden_layer_sizes": Categorical(
                        ["(8, 8)", "(16, 8)", "(32, 16, 8)", "(64, 32)"]
                    ),
                    "dropout_rate": Real(0.2, 0.4),
                    "learning_rate": Real(1e-4, 1e-2, prior="log-uniform"),
                    "activation_func": Categorical(["relu", "tanh", "sigmoid"]),
                    "epochs": Integer(5, 15),
                    "batch_size": Categorical([16, 32, 64]),
                }
            ]
        else:
            self.parameter_space = [
                {
                    "hidden_layer_sizes": [
                        (8, 8),
                        (16, 8),
                        (32, 16, 8),
                        (64, 32),
                    ],
                    "dropout_rate": [0.2, 0.3, 0.4],
                    "learning_rate": [1e-4, 1e-3, 1e-2],
                    "activation_func": ["relu", "tanh", "sigmoid"],
                    "epochs": [5, 10, 15],
                    "batch_size": [16, 32, 64],
                }
            ]
