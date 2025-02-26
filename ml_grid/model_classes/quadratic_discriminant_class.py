"""Define QuadraticDiscriminantAnalysis class"""

from ml_grid.util import param_space
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from ml_grid.util.global_params import global_parameters
from scipy.stats import uniform
import numpy as np
from skopt.space import Categorical

print("Imported QuadraticDiscriminantAnalysis class")


class quadratic_discriminant_analysis_class:
    """QuadraticDiscriminantAnalysis."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        global_params = global_parameters
        self.X = X
        self.y = y

        self.algorithm_implementation = QuadraticDiscriminantAnalysis()
        self.method_name = "QuadraticDiscriminantAnalysis"

        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        # print(self.parameter_vector_space)

        if(global_params.bayessearch):
            self.parameter_space = {
                "priors": Categorical([None]),  # Categorical: single option, None
                "reg_param": self.parameter_vector_space.param_dict.get("log_small"),  # Log-uniform between 1e-5 and 1e-2
                "store_covariance": Categorical([False]),  # Categorical: single option, False
                "tol": self.parameter_vector_space.param_dict.get("log_small"),  # Log-uniform between 1e-5 and 1e-2
            }

        else:
            self.parameter_space = {
                "priors": [None],
                "reg_param": self.parameter_vector_space.param_dict.get("log_small"),
                "store_covariance": [False],
                "tol": self.parameter_vector_space.param_dict.get("log_small"),
            }

        return None

        # print("init log reg class ", self.parameter_space)
