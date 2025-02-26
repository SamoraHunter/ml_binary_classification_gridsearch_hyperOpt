import numpy as np
from ml_grid.util import param_space
from sklearn.ensemble import RandomForestClassifier
from skopt.space import Real, Categorical, Integer

from ml_grid.util.global_params import global_parameters

class RandomForestClassifier_class:
    """RandomForestClassifier with support for both Bayesian and non-Bayesian parameter spaces."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """
        Initialize the RandomForestClassifier class.

        Args:
            X (pd.DataFrame): Feature matrix for training (optional).
            y (pd.Series): Target vector for training (optional).
            parameter_space_size (int): Size of the parameter space for optimization.
        """
        self.X = X
        self.y = y

        # Initialize the RandomForestClassifier
        self.algorithm_implementation = RandomForestClassifier()
        self.method_name = "RandomForestClassifier"

        # Define the parameter vector space
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        # Define the parameter space for Bayesian and traditional search
        if global_parameters.bayessearch:
            # Bayesian Optimization: Adjust parameters if traditional doesn't use param_dict
            self.parameter_space = {
                "bootstrap": Categorical([True, False]),
                "ccp_alpha": self.parameter_vector_space.param_dict.get("lin_zero_one"),
                "criterion": Categorical(["gini", "entropy", "log_loss"]),
                "max_depth": self.parameter_vector_space.param_dict.get("log_med"),
                "max_features": Categorical(["sqrt", "log2"]),
                "max_samples": Categorical([None]),
                "min_samples_leaf": self.parameter_vector_space.param_dict.get("log_med"),
                "min_samples_split": self._valid_min_samples_split(self.parameter_vector_space.param_dict.get("log_med")),  # patched line
                "n_estimators": self.parameter_vector_space.param_dict.get("log_large_long"),
                "n_jobs": Categorical([None]),
                "oob_score": Categorical([False]),
                "random_state": Categorical([None]),
                "verbose": Categorical([0]),
                "warm_start": Categorical([False]),
            }
        else:
            # Traditional Grid Search: Use lists directly from parameter vector space
            self.parameter_space = {
                "bootstrap": [True, False],
                "ccp_alpha": self.parameter_vector_space.param_dict.get("lin_zero_one"),
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": self.parameter_vector_space.param_dict.get("log_med"),
                "max_features": ["sqrt", "log2"],
                "max_samples": [None],
                "min_samples_leaf": self.parameter_vector_space.param_dict.get("log_med"),
                "min_samples_split": self._valid_min_samples_split(self.parameter_vector_space.param_dict.get("log_med")),  # patched line
                "n_estimators": self.parameter_vector_space.param_dict.get("log_large_long"),
                "n_jobs": [None],
                "oob_score": [False],
                "random_state": [None],
                "verbose": [0],
                "warm_start": [False],
            }

    def _valid_min_samples_split(self, values):
        """
        Ensure values for min_samples_split are valid integers (>=2) or floats (between 0.0 and 1.0).
        If necessary, modify values to ensure they are valid.
        """
        # Check if values is an Integer (this is common in Bayesian optimization parameter space)
        if isinstance(values, Integer):
            # Extract the valid range for the Integer type (min, max)
            min_val, max_val = values.bounds

            # Ensure min_val and max_val are integers (if they are floats, convert them)
            min_val = int(np.floor(min_val))  # Convert to int
            max_val = int(np.floor(max_val))  # Convert to int

            valid_values = [v for v in range(min_val, max_val + 1) if v >= 2]  # Ensure v >= 2 for integers
        else:
            # If values is iterable (list or array), apply filtering logic
            valid_values = [v for v in values if (isinstance(v, int) and v >= 2) or (isinstance(v, float) and 0.0 < v < 1.0)]

        # If no valid values were found, fallback to a default valid value (e.g., 2)
        if not valid_values:
            valid_values = [2]  # Fallback to a default valid value if no valid value found

        return valid_values

