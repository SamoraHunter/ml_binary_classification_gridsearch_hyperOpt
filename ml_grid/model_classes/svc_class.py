"""Define SVC class"""

from ml_grid.util import param_space
from sklearn.svm import SVC

print("Imported SVC class")


class SVC_class:
    """SVC."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        self.X = X
        self.y = y

        self.algorithm_implementation = SVC()
        self.method_name = "SVC"

        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        # print(self.parameter_vector_space)

        self.parameter_space = {
            "C": self.parameter_vector_space.param_dict.get("log_small"),
            "break_ties": self.parameter_vector_space.param_dict.get("bool_param"),
            #'cache_size': [200],
            #'class_weight': [None, 'balanced'] + [{0: w} for w in [1, 2, 4, 6, 10]], # enumerate class weight
            "coef0": self.parameter_vector_space.param_dict.get("log_small"),
            "decision_function_shape": ["ovr"],  # , 'ovo'
            "degree": self.parameter_vector_space.param_dict.get("log_med"),
            "gamma": ["scale", "auto"],
            "kernel": ["rbf", "linear", "poly", "sigmoid"],
            "max_iter": self.parameter_vector_space.param_dict.get("log_large_long"),
            #'probability': [False],
            #'random_state': [None],
            "shrinking": self.parameter_vector_space.param_dict.get("bool_param"),
            "tol": self.parameter_vector_space.param_dict.get("log_small"),
            "verbose": [False],
        }

        return None

        # print("init log reg class ", self.parameter_space)
