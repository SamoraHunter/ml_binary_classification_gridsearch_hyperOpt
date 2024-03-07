"""Define NeuralNetworkClassifier class"""

from ml_grid.util import param_space

# from ml_grid.model_classes.nni_sklearn_wrapper import *
from ml_grid.model_classes.nni_sklearn_wrapper import NeuralNetworkClassifier

print("Imported NeuralNetworkClassifier class")


class NeuralNetworkClassifier_class:
    """NeuralNetworkClassifier."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        self.X = X
        self.y = y

        self.algorithm_implementation = NeuralNetworkClassifier()
        self.method_name = "NeuralNetworkClassifier"

        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        self.parameter_space = [
            {
                "hidden_units_1": [1, 2, 3],
                "hidden_units_2": [1, 2, 3],
                "dropout_rate": [0.2, 0.3, 0.4],
                "learning_rate": [1e-4, 1e-3, 1e-2],
                "activation_func": ["relu", "tanh", "sigmoid"],
                "epochs": [5, 10, 15],
                "batch_size": [1],
            }
        ]

        return None

        # print("init log reg class ", self.parameter_space)
