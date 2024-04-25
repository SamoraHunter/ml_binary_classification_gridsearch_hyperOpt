"""Define logistic regression class"""

from ml_grid.util import param_space
from sklearn.linear_model import LogisticRegression


print("Imported logistic regression class")


class LogisticRegression_class:
    """LogisticRegression."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
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

        self.parameter_space = [
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

        # print("init log reg class ", self.parameter_space)
