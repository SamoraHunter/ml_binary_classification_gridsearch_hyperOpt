"""Define GradientBoostingClassifier class"""

from ml_grid.util import param_space
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

print("Imported GradientBoostingClassifier class")


class GradientBoostingClassifier_class:
    """GradientBoostingClassifier."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        self.X = X
        self.y = y

        self.algorithm_implementation = GradientBoostingClassifier()
        self.method_name = "GradientBoostingClassifier"

        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        # print(self.parameter_vector_space)

        self.parameter_space = {
            "ccp_alpha": self.parameter_vector_space.param_dict.get("log_small"),
            "criterion": ["friedman_mse"],
            "init": [None],
            "learning_rate": self.parameter_vector_space.param_dict.get("log_small"),
            "loss": ["log_loss", "exponential"],
            #'max_depth': log_med,
            "max_features": ["sqrt", "log2"],
            #'max_leaf_nodes': log_large_long,
            #'min_impurity_decrease': log_small,
            #'min_samples_leaf': log_med,
            #'min_samples_split': log_med,
            #'min_weight_fraction_leaf': log_small,
            "n_estimators": self.parameter_vector_space.param_dict.get(
                "log_large_long"
            ),
            #'n_iter_no_change': log_large_long,
            "random_state": [None],
            "subsample": np.delete(
                self.parameter_vector_space.param_dict.get("lin_zero_one"), 0
            ),
            "tol": self.parameter_vector_space.param_dict.get("log_small"),
            "validation_fraction": [0.1],
            "verbose": [0],
            "warm_start": [0],
        }

        return None

        # print("init log reg class ", self.parameter_space)
