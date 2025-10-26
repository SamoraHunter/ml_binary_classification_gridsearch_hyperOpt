from typing import Optional
import pandas as pd
from ml_grid.util import param_space
from ml_grid.model_classes.NeuralNetworkKerasClassifier import NeuralNetworkClassifier
import logging

logging.getLogger('ml_grid').debug("Imported NeuralNetworkClassifier class")


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
        """
        self.X = X
        self.y = y

        self.algorithm_implementation = NeuralNetworkClassifier()
        self.method_name = "NeuralNetworkClassifier"

        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        from ml_grid.util.global_params import global_parameters
        import logging
        if global_parameters.bayessearch:
            from skopt.space import Categorical, Integer, Real

            self.parameter_space = [
                {
                    # Changed from: Categorical([(8, 8), (16, 8), ...])
                    # To: Categorical(["(8, 8)", "(16, 8)", ...])
                    # Tuples encoded as strings to avoid skopt's .item() error
                    "hidden_layer_sizes": Categorical([
                        "(8, 8)",
                        "(16, 8)", 
                        "(32, 16, 8)",
                        "(64, 32)"
                    ]),
                    "dropout_rate": Real(0.2, 0.4),
                    "learning_rate": Real(1e-4, 1e-2, prior='log-uniform'),
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