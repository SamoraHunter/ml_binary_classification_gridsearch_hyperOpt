"""Define MLPClassifier class"""

from ml_grid.util import param_space
from sklearn.neural_network import MLPClassifier

print("Imported MLPClassifier class")


class mlp_classifier_class:
    """MLPClassifier."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        self.X = X
        self.y = y

        self.algorithm_implementation = MLPClassifier()
        self.method_name = "MLPClassifier"

        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        # print(self.parameter_vector_space)

        self.parameter_space = {
            "activation": ["relu"],
            "alpha": self.parameter_vector_space.param_dict.get("log_small"),
            "batch_size": ["auto"],
            #'beta_1': log_small,
            #'beta_2': log_small,
            #'early_stopping': bool_param,
            #'epsilon': log_small,
            "hidden_layer_sizes": self.parameter_vector_space.param_dict.get(
                "log_large_long"
            ),
            "learning_rate": ["adaptive"],  # ["constant", "adaptive"],
            #'learning_rate_init': log_small,
            #'max_fun': 15000,
            #'max_iter': log_large_long,
            "momentum": self.parameter_vector_space.param_dict.get("lin_zero_one"),
            #'n_iter_no_change': log_large_long,
            #'nesterovs_momentum': [True],
            #'power_t': 0.5,
            "random_state": [None],
            # "shuffle": bool_param,
            #'solver': ['adam', 'lbfgs', 'sgd'],
            # "tol": log_small,
            "validation_fraction": [0.1],
            "verbose": [False],
            "warm_start": [False],
        }

        return None

        # print("init log reg class ", self.parameter_space)
