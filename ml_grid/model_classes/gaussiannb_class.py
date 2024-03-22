"""Define adaboost class"""

from ml_grid.util import param_space
from sklearn.naive_bayes import GaussianNB

print("Imported gaussiannb class")


class GaussianNB_class:
    """gaussiannb."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        self.X = X
        self.y = y

        self.algorithm_implementation = GaussianNB()
        self.method_name = "GaussianNB"

        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        # print(self.parameter_vector_space)

        self.new_list = list(
            self.parameter_vector_space.param_dict.get("log_small")
        ).copy()
        self.new_list.append(1e-09)
        self.parameter_space = {
            "priors": [
                None,
                [0.1, 0.9],
                [0.9, 0.1],
                [0.7, 0.3],
                [0.3, 0.7],
                [0.5, 0.5],
                [0.6, 0.4],
                [0.4, 0.6],
            ],  # enumerate
            "var_smoothing": self.new_list,
        }

        return None

        # print("init log reg class ", self.parameter_space)
