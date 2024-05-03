"""Define logistic regression class"""

from ml_grid.util import param_space
from sklearn.linear_model import LogisticRegression


print("Imported logistic regression class")


class LogisticRegression_class:
    """LogisticRegression."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """Initialize the LogisticRegression class.

        Args:
            X (numpy.ndarray): Features
            y (numpy.ndarray): Target variable
            parameter_space_size (int): Size of the parameter space
        """
        self.X = X
        self.y = y

        self.algorithm_implementation = LogisticRegression()
        self.method_name = "LogisticRegression"

        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        # print(self.parameter_vector_space)

        # self.parameter_space = {
        #     "C": self.parameter_vector_space.param_dict.get('log_small'),
        #     "class_weight": [None],
        #     "dual": [False],
        #     "fit_intercept": [True],
        #     "intercept_scaling": [1],
        #     "l1_ratio": [None],
        #     "max_iter": self.parameter_vector_space.param_dict.get('log_large_long'),
        #     "multi_class": ["auto"],
        #     "n_jobs": [None],
        #     "penalty": ["l2", "l1", "elasticnet"], #None
        #     "random_state": [None],
        #     "solver": ["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
        #     "tol": self.parameter_vector_space.param_dict.get('log_small'),
        #     "verbose": [0],
        #     "warm_start": [False],
        # }

        # Create a list of dictionaries with the parameter space
        # Each dictionary is a combination of parameters
        self.parameter_space = [
            {
                "C": self.parameter_vector_space.param_dict.get("log_small"),
                "class_weight": [None],
                "dual": [False],
                "fit_intercept": [True],
                "intercept_scaling": [1],
                "l1_ratio": [0.5],
                "max_iter": self.parameter_vector_space.param_dict.get(
                    "log_large_long"
                ),
                "multi_class": ["auto"],
                "n_jobs": [None],
                "penalty": [
                    "elasticnet",
                    "l1",
                    "l2",
                ],  # None
                "random_state": [None],
                "solver": ["saga"],
                "tol": self.parameter_vector_space.param_dict.get("log_small"),
                "verbose": [0],
                "warm_start": [False],
            },
            {
                "C": self.parameter_vector_space.param_dict.get("log_small"),
                "class_weight": [None],
                "dual": [False],
                "fit_intercept": [True],
                "intercept_scaling": [1],
                "l1_ratio": [None],
                "max_iter": self.parameter_vector_space.param_dict.get(
                    "log_large_long"
                ),
                "multi_class": ["auto"],
                "n_jobs": [None],
                "penalty": [
                    "l2",
                ],  # None
                "random_state": [None],
                "solver": ["newton-cg", "lbfgs"],
                "tol": self.parameter_vector_space.param_dict.get("log_small"),
                "verbose": [0],
                "warm_start": [False],
            },
        ]

        return None

    # docstring
    def __repr__(self):
        """Return the representation of the LogisticRegression class."""
        return "LogisticRegression class"

    # docstring
    def __str__(self):
        """Return the string representation of the LogisticRegression class."""
        return "LogisticRegression class"
