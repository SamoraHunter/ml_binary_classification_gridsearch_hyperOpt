"""Define QuadraticDiscriminantAnalysis class"""

from ml_grid.util import param_space
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

print("Imported QuadraticDiscriminantAnalysis class")


class quadratic_discriminant_analysis_class:
    """QuadraticDiscriminantAnalysis."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        self.X = X
        self.y = y

        self.algorithm_implementation = QuadraticDiscriminantAnalysis()
        self.method_name = "QuadraticDiscriminantAnalysis"

        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        # print(self.parameter_vector_space)

        self.parameter_space = {
            "priors": [None],
            "reg_param": self.parameter_vector_space.param_dict.get("log_small"),
            "store_covariance": [False],
            "tol": self.parameter_vector_space.param_dict.get("log_small"),
        }

        return None

        # print("init log reg class ", self.parameter_space)
