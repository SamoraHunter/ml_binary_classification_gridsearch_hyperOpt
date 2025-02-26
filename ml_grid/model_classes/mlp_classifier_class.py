from sklearn.neural_network import MLPClassifier
from skopt.space import Real, Integer, Categorical
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters

print("Imported MLPClassifier class")


class mlp_classifier_class:
    """MLPClassifier with support for both Bayesian and non-Bayesian parameter spaces."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """
        Initialize the mlp_classifier_class.

        Args:
            X (pd.DataFrame): Feature matrix for training (optional).
            y (pd.Series): Target vector for training (optional).
            parameter_space_size (int): Size of the parameter space for optimization.
        """
        self.X = X
        self.y = y

        # Initialize the MLPClassifier
        self.algorithm_implementation = MLPClassifier()
        self.method_name = "MLPClassifier"

        # Define the parameter vector space
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        if global_parameters.bayessearch:
            # Bayesian Optimization: Define parameter space using skopt.space
            self.parameter_space = {
                "activation": Categorical(["relu"]),  # Fixed value as per the original
                "alpha": self.parameter_vector_space.param_dict.get("log_small"),  # Logarithmic scale (param_dict)
                "batch_size": Categorical(["auto"]),  # Fixed value as per the original
                "hidden_layer_sizes": self.parameter_vector_space.param_dict.get("log_large_long"),  # Log scale for large long sizes (param_dict)
                "learning_rate": Categorical(["adaptive"]),  # Fixed value as per the original
                "momentum": self.parameter_vector_space.param_dict.get("lin_zero_one"),  # Linear scale from 0 to 1 (param_dict)
                "random_state": Categorical([None]),  # Fixed value as per the original
                "validation_fraction": Real(0.05, 0.2),  # Real value in a small range [0.1, 0.2]
                "verbose": Categorical([False]),  # Fixed value as per the original
                "warm_start": Categorical([False]),  # Fixed value as per the original
            }
        else:
            # Traditional Grid Search: Define parameter space using lists
            self.parameter_space = {
                "activation": ["relu"],  # ["relu", "tanh", "logistic"]
                "alpha": list(self.parameter_vector_space.param_dict.get("log_small")),  # (param_dict)
                "batch_size": ["auto"],  # ["auto", 32, 64]
                # "beta_1": list(self.parameter_vector_space.param_dict.get("log_small")),
                # "beta_2": list(self.parameter_vector_space.param_dict.get("log_small")),
                # "early_stopping": list(self.parameter_vector_space.param_dict.get("bool_param")),
                # "epsilon": list(self.parameter_vector_space.param_dict.get("log_small")),
                "hidden_layer_sizes": list(self.parameter_vector_space.param_dict.get("log_large_long")),  # (param_dict)
                "learning_rate": ["adaptive"],  # ["constant", "adaptive"]
                # "learning_rate_init": list(self.parameter_vector_space.param_dict.get("log_small")),
                # "max_fun": [20000],  # Example fixed value
                # "max_iter": list(self.parameter_vector_space.param_dict.get("log_large_long")),
                "momentum": list(self.parameter_vector_space.param_dict.get("lin_zero_one")),  # (param_dict)
                # "n_iter_no_change": list(self.parameter_vector_space.param_dict.get("log_med")),
                # "nesterovs_momentum": [True],
                # "power_t": list(self.parameter_vector_space.param_dict.get("log_small")),
                "random_state": [None],  # [None, 42]
                # "shuffle": [True, False],
                # "solver": ["adam", "lbfgs", "sgd"],
                # "tol": list(self.parameter_vector_space.param_dict.get("log_small")),
                "validation_fraction": [0.1],  # [0.1, 0.2]
                "verbose": [False],
                "warm_start": [False],
            }

        return None
